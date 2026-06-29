"""声纹相似度检测 — Resemblyzer 说话人嵌入对比

用法:
    from core.voice_similarity import compare_similarity
    result = compare_similarity("tts_output.wav", "reference_vocals.wav")
    # result = {"similarity": 0.85, "device": "cuda", "error": None}
"""

import os
import numpy as np

from core.utils import resolve_device

_encoder = None        # 全局缓存的 VoiceEncoder 实例
_device = None         # 实际使用的设备字符串


def _get_device() -> str:
    """自动检测可用设备（代理 resolve_device)"""
    global _device
    if _device is not None:
        return _device
    _device = resolve_device("auto")
    return _device


def _load_encoder():
    """惰性加载 Resemblyzer VoiceEncoder（仅首次调用时加载)"""
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        from resemblyzer import VoiceEncoder
        device = _get_device()
        _encoder = VoiceEncoder(device=device)
        return _encoder
    except ImportError as e:
        raise ImportError(
            "需要安装 resemblyzer: pip install resemblyzer\n"
            "注意: resemblyzer 依赖 torch,请确保 torch 已安装"
        ) from e


def _load_audio(path: str) -> np.ndarray:
    """加载音频并返回归一化波形数组"""
    from resemblyzer import preprocess_wav
    if not os.path.exists(path):
        raise FileNotFoundError(f"音频文件不存在: {path}")
    return preprocess_wav(path)


def compare_similarity(tts_wav: str, orin_wav: str) -> dict:
    """比较 TTS 音频与原始人声的声纹相似度

    Args:
        tts_wav: TTS 合成后的音频路径
        orin_wav: 原始人声参考音频路径

    Returns:
        dict: {
            "similarity": float,   # 余弦相似度 (0~1),None 表示失败
            "device": str,         # 实际使用的设备
            "error": str | None,   # 错误信息
        }
    """
    result = {"similarity": None, "device": _get_device(), "error": None}

    try:
        encoder = _load_encoder()
        tts_wav_array = _load_audio(tts_wav)
        orin_wav_array = _load_audio(orin_wav)
    except Exception as e:
        result["error"] = f"加载失败: {e}"
        return result

    # 检查音频是否有效
    if len(tts_wav_array) < 100:
        result["error"] = f"TTS 音频过短 ({len(tts_wav_array)} samples)"
        return result
    if len(orin_wav_array) < 100:
        result["error"] = f"参考音频过短 ({len(orin_wav_array)} samples)"
        return result

    try:
        tts_embed = encoder.embed_utterance(tts_wav_array)
        orin_embed = encoder.embed_utterance(orin_wav_array)

        # 余弦相似度
        tts_norm = tts_embed / np.linalg.norm(tts_embed)
        orin_norm = orin_embed / np.linalg.norm(orin_embed)
        similarity = float(np.dot(tts_norm, orin_norm))
        similarity = max(-1.0, min(1.0, similarity))  # 截断到合法范围

        result["similarity"] = similarity
    except Exception as e:
        result["error"] = f"推理失败: {e}"

    return result




