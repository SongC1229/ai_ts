"""WavLM Large — Age & Sex classification model (adapted from vox-profile-release)"""

import os
import warnings
import torch
import loralib as lora
import transformers.models.wavlm.modeling_wavlm as wavlm

from torch import nn
from torch.nn import functional as F
from huggingface_hub import PyTorchModelHubMixin
from transformers import Wav2Vec2FeatureExtractor
from transformers import WavLMModel

from .revgrad import RevGrad

# 屏蔽 WavLM attention mask 类型不匹配的 deprecation warning
warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask and attn_mask")


def make_padding_masks(wav, wav_len=None):
    """Create attention masks from audio lengths (replaces speechbrain import).

    Args:
        wav: (batch, samples) raw audio tensor
        wav_len: (batch,) relative lengths in [0, 1], or None for full length
    Returns:
        (batch, samples) boolean mask — True = valid (attend), False = pad (ignore)
    """
    batch_size, max_len = wav.shape
    if wav_len is None:
        return torch.ones(batch_size, max_len, dtype=torch.bool, device=wav.device)
    abs_len = (wav_len * max_len).long()
    range_tensor = torch.arange(max_len, device=wav.device).unsqueeze(0).expand(batch_size, -1)
    return range_tensor < abs_len.unsqueeze(1)


class WavLMEncoderLayerStableLayerNorm(nn.Module):
    """WavLM encoder layer with StableLayerNorm and optional LoRA on later layers."""

    def __init__(self, layer_idx, config, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = wavlm.WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = wavlm.WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config

        if layer_idx > config.num_hidden_layers // 2:
            method = getattr(self.config, "finetune_method", "lora")
            if method == "lora" or method == "combined":
                rank = getattr(self.config, "lora_rank", 16)
                self.feed_forward.intermediate_dense = lora.Linear(
                    config.hidden_size, config.intermediate_size, r=rank
                )
                self.feed_forward.output_dense = lora.Linear(
                    config.intermediate_size, config.hidden_size, r=rank
                )

    def forward(self, hidden_states, attention_mask=None, position_bias=None,
                output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states, position_bias)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class WavLMWrapper(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/tiantiaf0627/vox-profile-release"
):
    """WavLM-Large with LoRA fine-tuning for age & sex classification.

    Adapted from vox-profile-release (https://github.com/tiantiaf0627/vox-profile-release).

    Input: raw audio waveform at 16kHz, shape (batch, samples), values in [-1, 1].
    Output: (age, sex_logits)
        - age: float tensor (batch, 1), value in [0, 1] — multiply by 100 for years
        - sex_logits: float tensor (batch, 2) — [Female, Male]
    """

    def __init__(
        self,
        pretrain_model="wavlm_large",
        hidden_dim=256,
        finetune_method="lora",
        lora_rank=16,
        freeze_params=True,
        output_class_num=2,
        use_conv_output=True,
        apply_gradient_reversal=False,
        num_dataset=4,
        num_age_bins=7,
        apply_reg=True,
    ):
        super(WavLMWrapper, self).__init__()

        self.pretrain_model = pretrain_model
        self.finetune_method = finetune_method
        self.apply_gradient_reversal = apply_gradient_reversal
        self.use_conv_output = use_conv_output

        # Load backbone WavLM-Large（仅需 config,权重由 from_pretrained 加载)
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 确定骨干网络 HuggingFace WavLM 配置来源
        # 优先：模型目录下的 wavlm_backbone_config.json（自包含部署,如 wavlm-large-age-sex-ja)
        # 后备：models/wavlm-large/config.json（原始骨干)
        import json as _json
        _model_dirs = [
            os.path.join(_root, "models", "wavlm-large-age-sex-ja"),
            os.path.join(_root, "models", "wavlm-large-age-sex"),
        ]
        _backbone_cfg_path = None
        for _d in _model_dirs:
            # 优先：专用 HuggingFace WavLM 配置（自包含部署）
            _p = os.path.join(_d, "wavlm_backbone_config.json")
            if os.path.exists(_p):
                _backbone_cfg_path = _p
                break
            # 后备：config.json（需确保是 HuggingFace WavLM config 而非自定义 11 字段配置）
            _has_pp = os.path.exists(os.path.join(_d, "preprocessor_config.json"))
            _has_cfg = os.path.exists(os.path.join(_d, "config.json"))
            if _has_pp and _has_cfg:
                # 检查是否是真正的 WavLM config（含 model_type 字段）
                try:
                    with open(os.path.join(_d, "config.json")) as _f:
                        if _json.load(_f).get("model_type") == "wavlm":
                            _backbone_cfg_path = os.path.join(_d, "config.json")
                            break
                except Exception:
                    pass
        if _backbone_cfg_path is None:
            # 后备：原始骨干路径
            _backbone_cfg_path = os.path.join(_root, "models", "wavlm-large", "config.json")

        from transformers import WavLMConfig
        with open(_backbone_cfg_path, 'r') as _f:
            _bb_config = WavLMConfig.from_dict(_json.load(_f))

        # processor 也从同一目录加载（preprocessor_config.json)
        _processor_path = os.path.dirname(_backbone_cfg_path)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            _processor_path, local_files_only=True
        )
        self.backbone_model = WavLMModel(_bb_config)

        # 骨干随机初始化；真正的预训练权重由外层 from_pretrained 用
        # model.safetensors 整体覆盖（该文件含完整骨干 + LoRA + 下游头)。
        # 这里捕获随机 state_dict 仅用于下一步：替换 encoder 层为 LoRA 版本后,
        # 用 load_state_dict(strict=False) 的 missing_keys 收集新增的 LoRA
        # 参数名（lora_A/lora_B),作为下面冻结逻辑判断"哪些可训练"的依据。
        state_dict = self.backbone_model.state_dict()
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method = finetune_method
        self.model_config.lora_rank = lora_rank

        # Replace encoder layers with LoRA-capable versions
        self.backbone_model.encoder.layers = nn.ModuleList(
            [
                WavLMEncoderLayerStableLayerNorm(
                    i, self.model_config, has_relative_position_bias=(i == 0)
                )
                for i in range(self.model_config.num_hidden_layers)
            ]
        )
        # missing_keys 即替换后新增的 LoRA 参数（骨干参数均命中,不算 missing)
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)

        # Freeze / unfreeze parameters
        self.freeze_params = freeze_params
        if self.freeze_params and finetune_method == "lora":
            for name, p in self.backbone_model.named_parameters():
                if name in msg.missing_keys:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        elif self.freeze_params:
            for _, p in self.backbone_model.named_parameters():
                p.requires_grad = False
        else:
            for _, p in self.backbone_model.named_parameters():
                p.requires_grad = True

        # Downstream 1D Conv layers
        self.model_seq = nn.Sequential(
            nn.Conv1d(self.model_config.hidden_size, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
        )

        if self.use_conv_output:
            num_layers = self.model_config.num_hidden_layers + 1
            self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        else:
            num_layers = self.model_config.num_hidden_layers
            self.weights = nn.Parameter(torch.zeros(num_layers))

        # Age head
        # apply_reg=True: sigmoid 回归头,输出 1 维（年龄 0~1,×100=岁)
        # apply_reg=False: 年龄分布分类头,输出 num_age_bins 维（7 个年龄段桶)
        #   注意 num_age_bins 与 num_dataset（数据集对抗头)是两个独立概念,不可混用
        self.apply_reg = apply_reg
        if apply_reg:
            self.age_dist_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.age_dist_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_age_bins),
            )

        # Sex head
        self.sex_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        if apply_gradient_reversal:
            self.dataset_layer = nn.Sequential(
                RevGrad(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_dataset),
            )

    def forward(self, x, length=None, return_feature=False, pred="age_dist_sex"):
        # 1. Feature extraction via processor
        with torch.no_grad():
            signal_list = []
            if length is not None:
                attention_mask = make_padding_masks(
                    x, wav_len=length / length.max()
                ).to(x.device)
            else:
                attention_mask = make_padding_masks(
                    x, wav_len=torch.tensor([1]).to(x.device)
                ).to(x.device)

            for idx in range(len(x)):
                inp = self.processor(
                    x[idx], sampling_rate=16_000, return_tensors="pt", padding=True
                )
                signal_list.append(inp["input_values"][0].to(x.device))
            signal = torch.stack(signal_list)

        # 2. Get length and mask
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            if signal.is_cuda:
                length = length.cuda()

        # 3. Transformer encoding
        x = self.backbone_model(
            signal, attention_mask=attention_mask, output_hidden_states=True
        ).hidden_states

        # 4. Weighted sum of hidden states
        if self.use_conv_output:
            stacked_feature = torch.stack(x, dim=0)
        else:
            stacked_feature = torch.stack(x, dim=0)[1:]

        _, *origin_shape = stacked_feature.shape
        if self.use_conv_output:
            stacked_feature = stacked_feature.view(
                self.backbone_model.config.num_hidden_layers + 1, -1
            )
        else:
            stacked_feature = stacked_feature.view(
                self.backbone_model.config.num_hidden_layers, -1
            )
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        features = weighted_feature.view(*origin_shape)

        # 5. 1D Conv
        features = features.transpose(1, 2)  # B x D x T
        features = self.model_seq(features)
        features = features.transpose(1, 2)  # B x T x D

        # 6. Pooling
        if length is not None:
            pooled = []
            for snt_id in range(features.shape[0]):
                actual_size = length[snt_id]
                pooled.append(torch.mean(features[snt_id, 0:actual_size, ...], dim=0))
            features = torch.stack(pooled)
        else:
            features = torch.mean(features, dim=1)

        # 7. Predictions
        age_predicted = self.age_dist_layer(features)
        sex_predicted = self.sex_layer(features)

        if return_feature:
            if self.apply_gradient_reversal:
                dataset_predicted = self.dataset_layer(features)
                return age_predicted, sex_predicted, dataset_predicted, features
            return age_predicted, sex_predicted, features
        if self.apply_gradient_reversal:
            dataset_predicted = self.dataset_layer(features)
            return age_predicted, sex_predicted, dataset_predicted
        return age_predicted, sex_predicted

    def get_feat_extract_output_lengths(self, input_length):
        """Compute output length of the convolutional layers."""
        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(
            self.backbone_model.config.conv_kernel,
            self.backbone_model.config.conv_stride,
        ):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length
