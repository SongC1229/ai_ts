"""dots.tts 本地 TTS 引擎 — 单例生命周期管理 + 推理封装

依赖:
  - dots.tts: https://github.com/rednote-hilab/dots.tts (已适配 Python 3.13)
  - 模型: models/dots.tts-soar/ (从 G:/ai/dots_tts_rainfall/models 复制)

用法:
    from core.dots_tts_engine import tts_synthesize, unload_dots_engine

    wav_data = tts_synthesize("你好世界", ref_audio_path="voice.wav")
    wav_data = tts_synthesize("你好", ref_audio=ref_bytes, target_duration_ms=2000)
    unload_dots_engine()
"""

import io
import os
import time
import uuid
from typing import Optional

import soundfile as sf

from .model_base import MLModelHolder
from .utils import resolve_device

# 首次导入时安装 tn 桩（WeTextProcessing 不兼容 Python 3.13)
import core.tn_stub  # noqa: F401


def _get_model_dir() -> str:
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "dots.tts-soar")
    return os.path.abspath(p)


def _get_src_dir() -> str:
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "dots.tts", "src")
    return os.path.abspath(p)


class _DotsTtsHolder(MLModelHolder):
    """dots.tts 模型的单例生命周期管理"""

    @classmethod
    def _load_impl(cls, device, dtype):
        import sys as _sys
        src_dir = _get_src_dir()
        if src_dir not in _sys.path:
            _sys.path.insert(0, src_dir)

        if device is None or device == "auto":
            device = resolve_device("auto")

        precision = "float16" if dtype == "float16" else "bfloat16"

        model_dir = _get_model_dir()
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"dots.tts 模型目录未找到: {model_dir}\n"
                f"请确保模型权重已下载到 models/dots.tts-soar/"
            )

        print(f"  [dots_tts] 加载模型 (precision={precision}, device={device})")
        t0 = time.time()

        from dots_tts.runtime import DotsTtsRuntime
        runtime = DotsTtsRuntime.from_pretrained(
            model_dir,
            precision=precision,
            optimize=False,
        )

        print(f"  [dots_tts] 模型加载完成 ({time.time()-t0:.1f}s)")
        print(f"  [dots_tts] 参数: {sum(p.numel() for p in runtime.model.parameters())/1e6:.0f}M")
        return runtime

    @classmethod
    def _unload_impl(cls):
        runtime = cls._model
        if runtime is None:
            return
        try:
            # 将模型移到 CPU
            runtime.model.cpu()
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ── 便利入口函数 ──

def load_dots_engine(device: str = "auto", dtype: str = "bfloat16"):
    return _DotsTtsHolder.load(device=device, dtype=dtype)


def unload_dots_engine():
    was_loaded = _DotsTtsHolder.is_loaded()
    _DotsTtsHolder.unload(move_to_cpu=False)
    if was_loaded:
        print("  [dots_tts] 引擎已卸载,显存已释放")


# ── 工具函数 ──

def _wav_bytes_to_tempfile(wav_bytes: bytes, suffix=".wav", work_dir: str = "") -> str:
    """将内存 WAV bytes 写入临时文件,返回路径"""
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, f"ref_{uuid.uuid4().hex}{suffix}")
    with open(path, "wb") as f:
        f.write(wav_bytes)
    return path


def _adjust_duration(wav_bytes: bytes, target_duration_ms: int, sample_rate: int = 48000) -> bytes:
    """通过时间拉伸调整音频到目标时长"""
    import librosa
    data, sr = sf.read(io.BytesIO(wav_bytes))
    current_duration_ms = len(data) / sr * 1000
    rate = current_duration_ms / target_duration_ms
    rate = max(0.5, min(2.0, rate))

    if abs(rate - 1.0) < 0.02:
        return wav_bytes

    y_stretched = librosa.effects.time_stretch(y=data, rate=rate)
    if sr != sample_rate:
        y_stretched = librosa.resample(y_stretched, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    buf = io.BytesIO()
    sf.write(buf, y_stretched, int(sr), format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ── 主合成函数 ──

def tts_synthesize(
    text: str,
    ref_audio_path: Optional[str] = None,
    ref_audio: Optional[bytes] = None,
    target_duration_ms: Optional[int] = None,
    stretch_to_target: bool = True,
    # dots.tts 特有参数
    prompt_text: Optional[str] = None,
    num_steps: int = 10,
    guidance_scale: float = 1.2,
    speaker_scale: float = 1.5,
    # 通用
    device: str = "auto",
    dtype: str = "bfloat16",
    work_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    **generation_kwargs,
) -> Optional[bytes]:
    """合成语音

    Args:
        text: 待合成文本
        ref_audio_path: 说话人参考音频文件路径
        ref_audio: 说话人参考音频 WAV bytes
        target_duration_ms: 可选,目标时长（毫秒)
        stretch_to_target: True 则拉伸音频对齐到该时长
        prompt_text: 参考音频的转录文本（续读克隆,推荐带上以提升相似度)
        num_steps: 流匹配采样步数 (默认10,越高越好)
        guidance_scale: CFG 尺度 (默认1.2)
        speaker_scale: 说话人条件强度 (默认1.5)
        output_path: 可选,指定输出 WAV 文件路径
    """
    if not text or not text.strip():
        print("  [dots_tts] 跳过空文本")
        return None

    if not work_dir:
        raise ValueError("work_dir is required")

    _temp_ref_path = None
    try:
        if ref_audio is not None:
            _temp_ref_path = _wav_bytes_to_tempfile(ref_audio, work_dir=work_dir)
            ref_path = _temp_ref_path
        elif ref_audio_path is not None:
            if not os.path.exists(ref_audio_path):
                print(f"  [dots_tts] 参考音频不存在: {ref_audio_path}")
                return None
            ref_path = ref_audio_path
        else:
            print("  [dots_tts] 必须提供 ref_audio 或 ref_audio_path")
            return None

        print(f"  [dots_tts] 参考音频: {ref_path}")

        # 加载模型
        try:
            runtime = load_dots_engine(device=device, dtype=dtype)
        except Exception as e:
            print(f"  [dots_tts] 模型加载失败: {e}")
            return None

        # 合成
        t0 = time.time()
        result = runtime.generate(
            text=text.strip(),
            prompt_audio_path=ref_path,
            prompt_text=prompt_text or "",
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            speaker_scale=speaker_scale,
            **generation_kwargs,
        )

        audio = result["audio"].float().cpu().squeeze().numpy()
        sr = result["sample_rate"]
        elapsed = time.time() - t0
        audio_secs = len(audio) / sr
        print(f"  [dots_tts] 合成完成: {elapsed:.1f}s, 音频={audio_secs:.1f}s, RTF={elapsed/audio_secs:.2f}")

        # 写入 WAV bytes
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        wav_data = buf.getvalue()

        # 时长对齐
        if target_duration_ms is not None and target_duration_ms > 0 and stretch_to_target:
            wav_data = _adjust_duration(wav_data, target_duration_ms)

        # 输出到指定路径
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(wav_data)
            print(f"  [dots_tts] 已保存到: {os.path.basename(output_path)}")

        return wav_data

    except Exception as e:
        print(f"  [dots_tts] 合成失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        if _temp_ref_path:
            try:
                os.unlink(_temp_ref_path)
            except Exception:
                pass
