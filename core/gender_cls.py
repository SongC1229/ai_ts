"""JaesungHuh/voice-gender-classifier 封装

用法:
    from .gender_cls import VoiceGenderClassifier
    model = VoiceGenderClassifier.from_pretrained()
    gender, prob = model.predict_file("audio.wav")  
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 原始权重精确映射 ───────────────────────────────────

class GenderConvBlock(nn.Module):
    """单一层的精确结构（7 路并行 conv + 1x1 混洗 + SE）"""
    def __init__(self, n_groups=7):
        super().__init__()
        # 7 路并行分组卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(128, 128, 3, padding=1) for _ in range(n_groups)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(128) for _ in range(n_groups)
        ])
        # 1×1 通道混洗
        self.conv1 = nn.Conv1d(1024, 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.conv3 = nn.Conv1d(1024, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        # SE
        self.se_fc1 = nn.Conv1d(1024, 128, 1)
        self.se_fc3 = nn.Conv1d(128, 1024, 1)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        # 分组卷积：处理前 n_groups*128 通道
        B, C, T = x.shape
        n = len(self.convs)
        outs = []
        for i in range(n):
            xi = x[:, i*128:(i+1)*128]
            outs.append(F.relu(self.bns[i](self.convs[i](xi))))
        # 剩余通道直接拼接（补齐到 1024）
        if n * 128 < C:
            outs.append(x[:, n*128:])
        x = torch.cat(outs, dim=1)
        x = F.relu(self.bn3(self.conv3(x)))
        # SE
        w = F.adaptive_avg_pool1d(x, 1)
        w = F.relu(self.se_fc1(w))
        w = torch.sigmoid(self.se_fc3(w))
        x = x * w
        return F.relu(x + r)


class VoiceGenderClassifier(nn.Module):
    """JaesungHuh/voice-gender-classifier"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(80, 1024, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(1024)

        # attention: Conv1d(1024→4608, 1) → BN → Conv1d(1536→256, 1)
        # 但 attention.0.weight 是 [256, 4608, 1]，不是 [4608, 1024, 1]
        # 所以先降维再 attention
        # attention（按权重 shape 对应）
        self.attn_down = nn.Conv1d(4608, 256, 1)
        self.attn_bn = nn.BatchNorm1d(256)
        self.attn_up = nn.Conv1d(256, 1536, 1)

        # 投影：1024→4608(Q/K/V 展开)，未被原权重覆盖，走随机初始化
        self.attn_qkv = nn.Conv1d(1024, 4608, 1)     # 保留（兼容旧 checkpoint）
        self.attn_out_proj = nn.Conv1d(1536, 1024, 1)  # 保留（兼容旧 checkpoint）

        self.layer1 = GenderConvBlock()
        self.layer2 = GenderConvBlock()
        self.layer3 = GenderConvBlock()

        self.bn5 = nn.BatchNorm1d(3072)
        self.conv_out = nn.Conv1d(3072, 1536, 1)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
        self.fc7 = nn.Linear(192, 2)

    def forward(self, mel):
        """mel: (B, 80, T)"""
        x = F.relu(self.bn1(self.conv1(mel)))   # (B, 1024, T)

        # attention: qkv projection → bottleneck → restore
        x_pool = F.adaptive_avg_pool1d(x, 64)
        qkv = self.attn_qkv(x_pool)  # (B, 4608, 64)
        qkv = self.attn_down(qkv)    # (B, 256, 64)
        qkv = F.relu(self.attn_bn(qkv))
        qkv = self.attn_up(qkv)      # (B, 1536, 64)
        out = self.attn_out_proj(qkv)  # (B, 1024, 64)
        x = x + F.interpolate(out, size=x.shape[-1], mode='linear')

        # 3 层 block
        skips = []
        for layer in [self.layer1, self.layer2, self.layer3]:
            x = layer(x)
            skips.append(F.adaptive_avg_pool1d(x, 1).squeeze(-1))

        x = torch.cat(skips, dim=-1)             # (B, 3072)
        x = F.relu(self.bn5(x))
        x = F.relu(self.bn6(self.fc6(x)))
        x = self.fc7(x)
        return x

    @classmethod
    def from_pretrained(cls, model_dir: str = "") -> 'VoiceGenderClassifier':
        """从本地 models/voice-gender-classifier-ja/ 加载"""
        from safetensors.torch import load_file

        if not model_dir:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "models", "voice-gender-classifier-ja")
        if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
            raise FileNotFoundError(
                f"模型文件不存在: {model_dir}/model.safetensors\n"
                f"请将 model.safetensors 和 config.json 放入该目录")

        model = cls()
        ckpt = load_file(os.path.join(model_dir, "model.safetensors"))

        # 手动映射 checkpoint 到模型
        mapping = {
            'conv1.weight': 'conv1.weight', 'conv1.bias': 'conv1.bias',
            'bn1.weight': 'bn1.weight', 'bn1.bias': 'bn1.bias',
            'bn1.running_mean': 'bn1.running_mean', 'bn1.running_var': 'bn1.running_var',
            'bn5.weight': 'bn5.weight', 'bn5.bias': 'bn5.bias',
            'bn5.running_mean': 'bn5.running_mean', 'bn5.running_var': 'bn5.running_var',
            'bn6.weight': 'bn6.weight', 'bn6.bias': 'bn6.bias',
            'bn6.running_mean': 'bn6.running_mean', 'bn6.running_var': 'bn6.running_var',
            'fc6.weight': 'fc6.weight', 'fc6.bias': 'fc6.bias',
            'fc7.weight': 'fc7.weight', 'fc7.bias': 'fc7.bias',
            'layer4.weight': 'conv_out.weight', 'layer4.bias': 'conv_out.bias',
        }

        sd = model.state_dict()
        for ck, mk in mapping.items():
            if ck in ckpt and mk in sd and ckpt[ck].shape == sd[mk].shape:
                sd[mk] = ckpt[ck]

        # 三层 block 映射
        for li in range(1, 4):
            prefix = f'layer{li}'
            for gi in range(7):
                sd[f'{prefix}.convs.{gi}.weight'] = ckpt[f'{prefix}.convs.{gi}.weight']
                sd[f'{prefix}.convs.{gi}.bias'] = ckpt[f'{prefix}.convs.{gi}.bias']
                sd[f'{prefix}.bns.{gi}.weight'] = ckpt[f'{prefix}.bns.{gi}.weight']
                sd[f'{prefix}.bns.{gi}.bias'] = ckpt[f'{prefix}.bns.{gi}.bias']
                sd[f'{prefix}.bns.{gi}.running_mean'] = ckpt[f'{prefix}.bns.{gi}.running_mean']
                sd[f'{prefix}.bns.{gi}.running_var'] = ckpt[f'{prefix}.bns.{gi}.running_var']
            for sub in ['conv1', 'bn1', 'conv3', 'bn3']:
                for attr in ['weight', 'bias', 'running_mean', 'running_var']:
                    k = f'{prefix}.{sub}.{attr}'
                    if k in ckpt:
                        sd[f'{prefix}.{sub}.{attr}'] = ckpt[k]
            # SE（兼容旧格式 se.se.1 和新格式 se_fc1）
            for se_name, se_key in [('se_fc1', 'se.se.1'), ('se_fc3', 'se.se.3')]:
                if f'{prefix}.{se_key}.weight' in ckpt:
                    sd[f'{prefix}.{se_name}.weight'] = ckpt[f'{prefix}.{se_key}.weight']
                    sd[f'{prefix}.{se_name}.bias'] = ckpt[f'{prefix}.{se_key}.bias']
                elif f'{prefix}.{se_name}.weight' in ckpt:
                    sd[f'{prefix}.{se_name}.weight'] = ckpt[f'{prefix}.{se_name}.weight']
                    sd[f'{prefix}.{se_name}.bias'] = ckpt[f'{prefix}.{se_name}.bias']

        # attention 映射（原 checkpoint key 与当前定义不同）
        attn_map = [
            ('attention.0.weight', 'attn_down.weight'),
            ('attention.0.bias', 'attn_down.bias'),
            ('attention.2.weight', 'attn_bn.weight'),
            ('attention.2.bias', 'attn_bn.bias'),
            ('attention.2.running_mean', 'attn_bn.running_mean'),
            ('attention.2.running_var', 'attn_bn.running_var'),
            ('attention.4.weight', 'attn_up.weight'),
            ('attention.4.bias', 'attn_up.bias'),
        ]
        for ck_key, sd_key in attn_map:
            if ck_key in ckpt:
                sd[sd_key] = ckpt[ck_key]

        model.load_state_dict(sd, strict=False)
        model.eval()
        return model

    @torch.no_grad()
    def predict(self, audio, sr=16000) -> Tuple[str, float]:
        import numpy as np
        if isinstance(audio, str):
            import soundfile as sf
            audio, sr = sf.read(audio)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio[:48000] if len(audio) > 48000 else np.pad(audio, (0, max(0, 48000 - len(audio))))
        x = torch.from_numpy(audio).float().unsqueeze(0)  # (1, 48000)
        if next(self.parameters()).is_cuda:
            x = x.cuda()
        mel = self._mel_gpu(x)
        logits = self.forward(mel)
        probs = F.softmax(logits, dim=1).squeeze(0)
        is_male = probs[1] > probs[0]
        return ("male" if is_male else "female", max(probs[0].item(), probs[1].item()))

    @staticmethod
    def _mel_gpu(waveform: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """GPU 端计算 mel 谱图（与训练一致）"""
        B, T = waveform.shape
        n_fft, hop_length, n_mels = 512, 160, 80
        window = torch.hann_window(n_fft, device=waveform.device)
        spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                          win_length=n_fft, window=window, pad_mode='reflect',
                          return_complex=True)
        mag = spec.abs() ** 2  # 功率谱
        if not hasattr(VoiceGenderClassifier._mel_gpu, '_mel_basis'):
            import librosa
            mel_basis_np = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=20, fmax=8000)
            VoiceGenderClassifier._mel_gpu._mel_basis = torch.from_numpy(mel_basis_np).float()
        mel_basis = VoiceGenderClassifier._mel_gpu._mel_basis.to(waveform.device)
        mel_spec = mel_basis @ mag
        mel_spec = torch.clamp(mel_spec, min=1e-10)
        mel_db = 10 * torch.log10(mel_spec)
        max_val = mel_db.amax(dim=(1, 2), keepdim=True)
        mel_db = torch.clamp(mel_db, min=max_val - 80.0) / 80.0
        return mel_db


if __name__ == '__main__':
    import sys
    model = VoiceGenderClassifier.from_pretrained()
    g, p = model.predict(sys.argv[1] if len(sys.argv) > 1 else "test.wav")
    print(f'{g} ({p:.1%})')
