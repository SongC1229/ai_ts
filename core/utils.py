"""统一工具函数集 — 时间格式化、音频读写、波形降采样、设备检测

本模块集中存放各模块间重复的工具函数,避免四处复制。
"""

import os
from typing import Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 时间格式化
# ══════════════════════════════════════════════════════════════════════════════

def _ms_parts(ms: int) -> tuple:
    """毫秒拆解为 (h, m, s, ms3)"""
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms3 = ms % 1000
    return h, m, s, ms3


def fmt_time(ms: int, ms_sep: str = ".") -> str:
    """毫秒转 HH:MM:SS.mmm（用于日志和 UI 显示)"""
    if ms < 0:
        ms = 0
    h, m, s, ms3 = _ms_parts(ms)
    return f"{h:02d}:{m:02d}:{s:02d}{ms_sep}{ms3:03d}"


def fmt_time_adaptive(ms: int, total_duration_ms: int) -> str:
    """自适应时间格式：>1h 显示 HH:MM:SS, >1min 显示 MM:SS.ms, 否则 SS.ms"""
    if ms < 0:
        ms = 0
    h, m, s, ms3 = _ms_parts(ms)
    if total_duration_ms >= 3600000:
        return f"{h:02d}:{m:02d}:{s:02d}"
    elif total_duration_ms >= 60000:
        return f"{m:02d}:{s:02d}.{ms3 // 100}"
    else:
        return f"{s}.{ms3 // 100}s"


def read_wav_np(path: str, mono: bool = True):
    """读取 WAV 为 float64 numpy 数组 (默认 mono)"""
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(path, dtype='float64')
    if mono and y.ndim > 1:
        y = np.mean(y, axis=1)
    return y, sr


def write_wav_np(path: str, y, sr: int):
    """写入 float64 numpy 数组为 16-bit WAV"""
    import soundfile as sf
    sf.write(path, y, sr, subtype='PCM_16')


def read_wav_segment(path: str, start_ms: int, end_ms: int = None, mono: bool = True):
    """从 WAV 文件读取指定毫秒区间的音频, 返回 (float64 array, sample_rate)"""
    import numpy as np
    import soundfile as sf
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        total_frames = len(f)
        start_frame = int(start_ms * sr / 1000)
        end_frame = int(end_ms * sr / 1000) if end_ms is not None else total_frames
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(start_frame, min(end_frame, total_frames))
        if end_frame <= start_frame:
            return None, sr
        f.seek(start_frame)
        y = f.read(end_frame - start_frame, dtype='float64')
        if mono and y.ndim > 1:
            y = np.mean(y, axis=1)
        return y, sr


def downsample_waveform(y, duration_ms: int, target_n: int = 200, max_duration_ms: int = 0) -> list:
    """将音频数据降采样为波形峰值 (用于 UI 显示)"""
    import numpy as np
    if len(y) == 0:
        return []
    n = min(target_n, len(y))
    chunk = len(y) // n
    peaks = []
    for i in range(0, len(y) - chunk + 1, chunk):
        peaks.append(float(np.max(np.abs(y[i:i + chunk]))))
    if len(peaks) < target_n and len(y) > 0:
        peaks.append(float(np.max(np.abs(y[-chunk:]))))
    return peaks[:target_n]


# ══════════════════════════════════════════════════════════════════════════════
# 设备检测（程序生命周期内只检测一次)
# ══════════════════════════════════════════════════════════════════════════════

_resolved_device: Optional[str] = None


def resolve_device(device: str = "auto") -> str:
    """统一设备检测：cuda > cpu（首次调用后缓存结果,后续直接返回)

    Args:
        device: "auto" 自动检测；传入 "cuda"/"cpu" 则直接返回

    Returns:
        "cuda" 或 "cpu"
    """
    global _resolved_device
    if device != "auto":
        return device
    if _resolved_device is None:
        try:
            import torch
            _resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            _resolved_device = "cpu"
    return _resolved_device


def cleanup_cuda():
    """统一 GPU 显存清理：gc.collect + cuda.empty_cache"""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 日志 / 线程工具
# ══════════════════════════════════════════════════════════════════════════════

def make_logger(log_cb):
    """创建安全日志函数：log_cb 为 None 时返回静默函数"""
    if log_cb:
        def _log(msg):
            log_cb(msg)
        return _log
    return lambda msg: None


def get_threads(ctx, attr_name: str, max_workers: int = 8) -> int:
    """从 ctx 获取线程数,未配置时回退为 1 并记录日志

    Args:
        ctx: PipelineContext
        attr_name: 属性名（如 'tts_threads')
        max_workers: 上限值

    Returns:
        实际线程数 (1 ~ max_workers)
    """
    raw = getattr(ctx, attr_name, None)
    if not raw:
        if ctx.log_ui:
            ctx.log_ui(f"  ⚠️ {attr_name} 未配置,回退为 1")
        return 1
    return max(1, min(raw, max_workers))

def calibrate_times(first_ms, last_ms, orig_s, orig_e, win_s, win_e, left_pad=0, right_pad=0, asr_pad_ms=100):
    cs = max(win_s, first_ms - asr_pad_ms)
    ce = last_ms + asr_pad_ms
    if left_pad > 0:
        cs = max(cs, win_s)
    if right_pad > 0:
        ce = min(ce, win_e)
    ce = max(ce, orig_e)
    cs = min(cs, orig_s)
    if ce - cs <= 50:
        return orig_s, orig_e
    return cs, ce
