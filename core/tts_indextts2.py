"""IndexTTS 2.0 本地 TTS 引擎 — 单例生命周期管理 + 推理封装

特点:
  - 输入: 参考音频支持内存 bytes（无需磁盘文件)
  - 时长: 支持输出时长与输入音频自动对齐（语音拉伸/压缩)
  - 单例: 复用模型,避免重复加载
  - 显存: 退出时自动释放 GPU 显存

用法:
    from core.tts_engine import tts_synthesize, unload_tts_engine

    # 1. 参考音频来自文件
    wav_data = tts_synthesize("你好世界", ref_audio_path="voice.wav")

    # 2. 参考音频来自内存 bytes
    with open("voice.wav", "rb") as f:
        ref_bytes = f.read()
    wav_data = tts_synthesize("你好世界", ref_audio=ref_bytes)

    # 3. 指定目标时长（毫秒)
    wav_data = tts_synthesize("你好", ref_audio=ref_bytes, target_duration_ms=2000)

    # 用完释放
    unload_tts_engine()
"""

import os
import io
import time
import uuid
from typing import Optional

from .model_base import MLModelHolder
from .utils import resolve_device
import torch


def _get_checkpoint_dir() -> str:
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "indextts2_src", "checkpoints")
    return os.path.abspath(p)


def _get_src_dir() -> str:
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "indextts2_src")
    return os.path.abspath(p)


class _IndexTTSHolder(MLModelHolder):
    """IndexTTS2 模型的单例生命周期管理"""

    _model_dir = None
    _fp16 = True

    @classmethod
    def _load_impl(cls, device, dtype):
        import sys as _sys
        src_dir = _get_src_dir()
        if src_dir not in _sys.path:
            _sys.path.insert(0, src_dir)

        checkpoints = _get_checkpoint_dir()
        hub_dir = os.path.join(checkpoints, "hub")
        hf_cache = os.path.join(checkpoints, "hf_cache")
        os.environ.setdefault("HF_HOME", hub_dir)
        os.environ.setdefault("MODELSCOPE_CACHE", hf_cache)

        if device is None or device == "auto":
            device = resolve_device("auto")

        use_fp16 = cls._fp16
        if dtype == "float32":
            use_fp16 = False
        elif dtype == "float16":
            use_fp16 = True

        cfg_path = os.path.join(checkpoints, "config.yaml")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"IndexTTS2 模型配置未找到: {cfg_path}\n"
                f"请确保模型权重已下载到 {checkpoints}"
            )

        print(f"  [tts] 加载 IndexTTS2 模型 (FP16={use_fp16}, device={device})")
        t0 = time.time()

        from indextts.infer_v2 import IndexTTS2

        # Monkey-patch torchaudio.save -> scipy（torch 2.11 torchcodec 不兼容 FFmpeg)
        import torchaudio as _ta
        from scipy.io import wavfile as _wavfile
        def _save_wav(uri, wav, sr):
            wav_np = wav.cpu().numpy()
            if wav_np.ndim == 2:
                wav_np = wav_np.T
            _wavfile.write(uri, sr, wav_np)
        _ta.save = _save_wav

        model = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=checkpoints,
            use_fp16=use_fp16,
            use_cuda_kernel=False,
            use_deepspeed=False,
        )

        print(f"  [tts] 模型加载完成 ({time.time()-t0:.1f}s)")
        return model

    @classmethod
    def _unload_impl(cls):
        """将子模块移到 CPU 后再释放"""
        model = cls._model
        if model is None:
            return
        try:
            # 清空缓存张量
            for attr in ['cache_spk_cond', 'cache_s2mel_style', 'cache_s2mel_prompt',
                         'cache_emo_cond', 'cache_mel']:
                if hasattr(model, attr):
                    setattr(model, attr, None)
            # 将 nn.Module 子模块移到 CPU
            for name in ['gpt', 's2mel', 'semantic_codec', 'campplus_model', 'bigvgan',
                         'semantic_model', 'qwen_emo']:
                mod = getattr(model, name, None)
                if mod is not None and hasattr(mod, 'cpu'):
                    mod.cpu()
        except Exception:
            pass


# ── 便利入口函数 ──

_speaker_emb_cache = {}  # .pt路径 -> [1,192] tensor


def load_tts_engine(device: str = "auto", dtype: str = "float16"):
    return _IndexTTSHolder.load(device=device, dtype=dtype)


def unload_tts_engine():
    was_loaded = _IndexTTSHolder.is_loaded()
    _IndexTTSHolder.unload(move_to_cpu=False)
    if was_loaded:
        print("  [tts] 引擎已卸载,显存已释放")


def train_speaker_embedding(clip_paths: list, output_path: str,
                            device: str = "auto", dtype: str = "float16",
                            **kwargs) -> str:
    """拼接多条参考音频,分组提取并平均说话人特征集

    拼接后一次性提取 style + spk_cond_emb + ref_mel + prompt_condition,
    确保所有特征来自同一段语音,维度一致。
    """
    _log = kwargs.get("log_cb") or (lambda msg: print(msg, flush=True))
    import librosa
    import torchaudio
    import soundfile as sf

    model = _IndexTTSHolder.load(device=device, dtype=dtype)
    if not clip_paths:
        raise ValueError("clip_paths 为空")

    MAX_CLIPS_PER_GROUP = 10
    MAX_SEC = 20
    SR = 24000

    # 分组: 每组最多10条,不超过20s
    groups = []
    cur_group = []
    cur_samples = 0

    for path in clip_paths:
        if not os.path.exists(path):
            print(f"  [tts] 跳过不存在的音频: {path}")
            continue
        data, file_sr = sf.read(path)
        if file_sr != SR:
            data = librosa.resample(data, orig_sr=file_sr, target_sr=SR)

        if cur_samples + len(data) > MAX_SEC * SR or len(cur_group) >= MAX_CLIPS_PER_GROUP:
            if cur_group:
                groups.append(cur_group)
            cur_group = [data]
            cur_samples = len(data)
        else:
            cur_group.append(data)
            cur_samples += len(data)
    if cur_group:
        groups.append(cur_group)

    if not groups:
        raise RuntimeError("没有有效的音频文件用于训练")

    import numpy as np
    results = []

    for g_idx, group in enumerate(groups):
        combined = np.concatenate(group)
        _log(f"  [tts]  组{g_idx+1}: {len(group)} 条, {len(combined)/SR:.1f}s")
        audio_t = torch.from_numpy(combined).float().unsqueeze(0)

        audio_16k = torchaudio.transforms.Resample(SR, 16000)(audio_t)
        audio_22k = torchaudio.transforms.Resample(SR, 22050)(audio_t)

        inputs = model.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        spk_cond = model.get_emb(input_features, attention_mask)

        _, S_ref = model.semantic_codec.quantize(spk_cond)
        ref_m = model.mel_fn(audio_22k.to(model.device).float())
        tgt_len = torch.LongTensor([ref_m.size(2)]).to(ref_m.device)
        prompt = model.s2mel.models["length_regulator"](
            S_ref, ylens=tgt_len, n_quantizers=3, f0=None)[0]

        feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(model.device),
                                                 num_mel_bins=80, dither=0,
                                                 sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = model.campplus_model(feat.unsqueeze(0))

        results.append({
            "speaker_embedding": style.detach().cpu(),
            "spk_cond_emb": spk_cond.detach().cpu(),
            "ref_mel": ref_m.detach().cpu(),
            "prompt_condition": prompt.detach().cpu(),
        })

    out = {"speaker_embedding": torch.stack([r["speaker_embedding"] for r in results]).mean(dim=0)}
    for key in ("spk_cond_emb", "ref_mel", "prompt_condition"):
        out[key] = results[0][key]

    torch.save(out, output_path)
    _log(f"  [tts] 说话人特征集已保存 ({len(groups)} 组, {sum(len(g) for g in groups)} 条): {output_path}")
    return output_path


def _adjust_duration(
    wav_bytes: bytes,
    target_duration_ms: int,
    sample_rate: int = 22050,
    target_chars: int = 0,
) -> bytes:
    """通过时间拉伸调整音频到目标时长（动态限制拉伸比率)

    Args:
        wav_bytes: 输入 WAV 字节
        target_duration_ms: 目标时长（毫秒)
        sample_rate: 采样率
        target_chars: 文本字符数,用于估算自然语速下的合理时长

    Returns:
        时长对齐后的 WAV 字节
    """
    import librosa
    import soundfile as sf

    # 读取内存 WAV
    data, sr = sf.read(io.BytesIO(wav_bytes))

    current_duration_ms = len(data) / sr * 1000
    rate = current_duration_ms / target_duration_ms  # >1 加速, <1 减速

    # ── 估算自然语速下的合理时长,作为拉伸上限参考 ──
    # 汉语平均语速 ~4 字/s = 250ms/字
    _natural_ms = target_chars * 250 if target_chars > 0 else 0
    # 拉伸上限: 15%, 且拉伸后不低于自然语速的 60%
    _max_stretch = 1.15
    _min_stretch = 0.85
    if _natural_ms > 0:
        # 如果目标时长远小于自然语速应有时长,保守拉伸
        _implied_rate = _natural_ms / target_duration_ms
        if _implied_rate > _max_stretch:
            print(f"  [tts] 目标时长({target_duration_ms}ms)远小于自然语速({_natural_ms}ms),"
                  f"拉伸率{rate:.3f}已超上限{_max_stretch}, 回退到{_max_stretch}")
            rate = _max_stretch
    rate = max(_min_stretch, min(_max_stretch, rate))

    if abs(rate - 1.0) < 0.02:
        # 差异 < 2%,无需调整
        return wav_bytes

    # 时间拉伸
    y_stretched = librosa.effects.time_stretch(y=data, rate=rate)

    # 重采样到模型原始采样率
    if sr != sample_rate:
        y_stretched = librosa.resample(y_stretched, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    # 写入内存 bytes
    buf = io.BytesIO()
    sf.write(buf, y_stretched, int(sr), format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ── 主合成函数 ──

def tts_synthesize(
    text: str,
    ref_audio_path: Optional[str] = None,
    ref_audio: Optional[bytes] = None,
    target_duration_ms: Optional[int] = None,
    stretch_to_target: bool = True,       # True=拉伸对齐, False=仅用于 max_mel_tokens 估算
    # 情绪控制参数
    emo_audio_path: Optional[str] = None,
    emo_audio: Optional[bytes] = None,
    emo_alpha: float = 0.3,
    emo_vector: Optional[list] = None,
    use_emo_text: bool = False,
    emo_text: Optional[str] = None,
    # 其他
    device: str = "auto",
    dtype: str = "float16",
    work_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    **generation_kwargs,
) -> Optional[bytes]:
    """合成语音

    Args:
        text: 待合成文本
        ref_audio_path: 说话人参考音频文件路径（与 ref_audio 二选一)
        ref_audio: 说话人参考音频 WAV bytes（与 ref_audio_path 二选一)
        target_duration_ms: 可选,目标时长（毫秒)。用于估算 max_mel_tokens
        stretch_to_target: True 则拉伸音频对齐到该时长,False 仅用于生成参数估算

        情绪控制（三选一,优先级: emo_vector > emo_audio > use_emo_text):
            emo_audio_path: 情绪参考音频文件路径（例如 angry.wav)
            emo_audio: 情绪参考音频 WAV bytes
            emo_alpha: 情绪 blend 比例 0.0-1.0,1.0=完全按参考情绪
            emo_vector: 8 维情绪向量 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            use_emo_text: 是否从文本自动检测情绪
            emo_text: 用于情绪检测的文本（默认使用 text 本身)

        device: 推理设备
        dtype: 精度 ("float16" / "float32")
        output_path: 可选,指定输出 WAV 文件路径
        **generation_kwargs: 传递给底层模型的生成参数:
            - max_mel_tokens (int): 最大生成 token 数,默认 1500,每 token ≈ 20ms
            - length_penalty (float): 长度偏好,>0 倾向更长,<0 倾向更短
            - temperature (float): 采样温度,默认 0.8
            - top_p / top_k: 采样参数
            - repetition_penalty (float): 重复惩罚,默认 10.0
            - num_beams (int): beam 宽度,默认 3

    Returns:
        WAV 音频字节数据（16-bit mono),失败返回 None
    """
    if not text or not text.strip():
        print("  [tts] 跳过空文本")
        return None

    if not work_dir:
        raise ValueError("work_dir is required")

    # ── 打印函数入参 ──
    _emb_hint = generation_kwargs.get('_emb_path_hint', '')
    print(f"  [tts] 文本: {text[:40]}...", flush=True)
    print(f"  [tts] 参考音频: {ref_audio_path or 'None'}", flush=True)
    print(f"  [tts] 音色权重: {os.path.basename(_emb_hint) if _emb_hint else 'None'}", flush=True)
    print(f"  [tts] 情绪参考: {emo_audio_path or 'None'}", flush=True)

    # ── 确定参考音频路径(仅非 pt 模式需要) ──
    _temp_ref_path = None
    ref_path = ""
    if _emb_hint:
        # pt 模式: 不要求 ref_audio,用占位路径确保模型内部不出错
        ref_path = ref_audio_path or "__fixed_speaker__"
    else:
        if ref_audio is not None:
            _temp_ref_path = _wav_bytes_to_tempfile(ref_audio, work_dir=work_dir)
            ref_path = _temp_ref_path
        elif ref_audio_path is not None:
            if not os.path.exists(ref_audio_path):
                print(f"  [tts] 参考音频不存在: {ref_audio_path}")
                return None
            ref_path = ref_audio_path
        else:
            print("  [tts] 必须提供 ref_audio 或 ref_audio_path")
            return None

    try:
        # ── 说话人嵌入缓存加载(必须在模型加载前) ──
        if _emb_hint:
            generation_kwargs.pop('_emb_path_hint', None)
            if _emb_hint not in _speaker_emb_cache:
                try:
                    _data = torch.load(_emb_hint)
                    if "speaker_embedding" in _data:
                        _speaker_emb_cache[_emb_hint] = _data
                except Exception:
                    pass
            _cached = _speaker_emb_cache.get(_emb_hint)
            if _cached and "speaker_embedding" in _cached:
                model = _IndexTTSHolder.load(device=device, dtype=dtype)
                _dev = model.device if hasattr(model, 'device') else next(model.parameters()).device
                model.cache_spk_cond = _cached["spk_cond_emb"].to(_dev)
                model.cache_s2mel_style = _cached["speaker_embedding"].to(_dev)
                model.cache_s2mel_prompt = _cached["prompt_condition"].to(_dev)
                model.cache_mel = _cached["ref_mel"].to(_dev)
                model.cache_spk_audio_prompt = ref_path
            generation_kwargs.pop('_emb_path_hint', None)

        # 确定情绪参考音频路径(用于模型显存)
        _temp_emo_path = None
        emo_path = None
        if emo_audio is not None:
            _temp_emo_path = _wav_bytes_to_tempfile(emo_audio, work_dir=work_dir)
            emo_path = _temp_emo_path
        elif emo_audio_path is not None and os.path.exists(emo_audio_path):
            emo_path = emo_audio_path

        try:
            model = load_tts_engine(device=device, dtype=dtype)
        except Exception as e:
            print(f"  [tts] 模型加载失败: {e}")
            return None

        # 输出路径
        out_path = os.path.join(work_dir, f"tts_{uuid.uuid4().hex}.wav")

        # 构建 infer 参数
        infer_kwargs = dict(
            spk_audio_prompt=ref_path,
            text=text.strip(),
            output_path=out_path,
            verbose=False,
            emo_alpha=emo_alpha,
        )
        if emo_path:
            infer_kwargs["emo_audio_prompt"] = emo_path
        if emo_vector is not None:
            infer_kwargs["emo_vector"] = emo_vector
        if use_emo_text:
            infer_kwargs["use_emo_text"] = True
        if emo_text is not None:
            infer_kwargs["emo_text"] = emo_text
        infer_kwargs.update(generation_kwargs)

        # 合成
        try:
            result = model.infer(**infer_kwargs)
            if result is None:
                print("  [tts] model.infer 返回 None", flush=True)
                return None
        except Exception as e:
            import traceback
            print(f"  [tts] model.infer 异常: {e}", flush=True)
            traceback.print_exc()
            return None

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 100:
            print("  [tts] 合成返回空文件")
            return None

        with open(out_path, "rb") as f:
            wav_data = f.read()

        # 裁剪前导静音,保留约 150ms（混音阶段需参考原声再微调)
        try:
            import soundfile as sf
            import io, numpy as np
            _data, _sr = sf.read(io.BytesIO(wav_data))
            _th = max(np.max(np.abs(_data)) * 0.003, 1e-4)
            _non_silent = np.where(np.abs(_data) > _th)[0]
            if len(_non_silent) > 0:
                _lead_ms = _non_silent[0] / _sr * 1000
                if _lead_ms > 180:
                    _trim_start = int(_non_silent[0] - int(0.15 * _sr))
                    if _trim_start > 0:
                        _data = _data[_trim_start:]
                        _buf = io.BytesIO()
                        sf.write(_buf, _data, _sr, format="WAV", subtype="PCM_16")
                        wav_data = _buf.getvalue()
                        print(f"  [tts] 裁剪前导静音: {_lead_ms:.0f}ms → 150ms")
        except Exception as _trim_e:
            print(f"  [tts] 裁剪前导静音失败: {_trim_e}")

        # 时长对齐（仅在 stretch_to_target=True 时拉伸)
        if target_duration_ms is not None and target_duration_ms > 0 and stretch_to_target:
            wav_data = _adjust_duration(wav_data, target_duration_ms, target_chars=len(text))
            print(f"  [tts] 时长对齐到 {target_duration_ms}ms")

        # 输出到指定路径
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(wav_data)
            print(f"  [tts] 已保存到: {os.path.basename(output_path)}")

        return wav_data

    except Exception as e:
        print(f"  [tts] 合成失败: {e}")
        return None

    finally:
        if _temp_ref_path:
            try:
                os.unlink(_temp_ref_path)
            except Exception:
                pass
        if 'out_path' in locals() or 'out_path' in dir():
            try:
                if os.path.exists(out_path):
                    os.unlink(out_path)
            except Exception:
                pass
