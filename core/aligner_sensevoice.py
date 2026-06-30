"""SenseVoice ASR 模型加载 — 供 pipeline _generate_asr_srt 使用"""

import os

import core.editdistance_fallback  # noqa: F401

_asr_model = None


def _get_models():
    """返回 (asr_model, None) — 仅加载 ASR, 不加载 VAD"""
    global _asr_model
    if _asr_model is None:
        from funasr import AutoModel
        _asr_model = AutoModel(
            model=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "models", "SenseVoiceSmall"),
            trust_remote_code=False, disable_update=True, device="cuda:0",
        )
    return _asr_model, None
