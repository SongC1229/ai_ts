"""SenseVoice 字幕对齐 — ASR 转写生成原声字幕文本

与 Qwen/Whisper 共享同一套规则: _compute_send_ranges 计算送检区间,
_extract_audio_clips 统一提取音频片段, calibrate_times 统一校准。
"""

import os
import re

import core.editdistance_fallback  # noqa: F401

_TAG_PATTERN = re.compile(r"<\|[^|]+\|>")
_asr_model = None
_vad_model = None


def _get_models():
    global _asr_model, _vad_model
    if _asr_model is None:
        from funasr import AutoModel
        _asr_model = AutoModel(
            model=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "models", "SenseVoiceSmall"),
            trust_remote_code=False, disable_update=True, device="cuda:0",
        )
    if _vad_model is None:
        _vad_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "models", "fsmn-vad")
        from funasr import AutoModel
        _vad_model = AutoModel(
            model=_vad_dir, trust_remote_code=False, disable_update=True, device="cuda:0",
        )
    return _asr_model, _vad_model


def align_subs(subs, clip_paths: dict, send_ranges: list, ctx, progress_cb=None):
    """SenseVoice ASR + FSMN VAD 对齐

    Args:
        subs: 字幕列表
        clip_paths: {sub.idx: clip_path} 预提取的送检音频
        send_ranges: _compute_send_ranges 输出
        ctx: PipelineContext
        progress_cb: (done, total)
    Returns:
        (corrected_times, seg_texts, has_changes)
    """
    import warnings
    warnings.filterwarnings('ignore')

    asr_model, vad_model = _get_models()
    total = len(subs)
    corrected_times = {}
    seg_texts = {}
    changed_count = 0

    for i, sub in enumerate(subs):
        if progress_cb:
            progress_cb(i + 1, total)
        orig_s, orig_e = sub.start_ms, sub.end_ms
        win_s, win_e, _left_pad, _right_pad = send_ranges[i]
        seg_path = clip_paths.get(sub.idx)
        if not seg_path or not os.path.exists(seg_path):
            corrected_times[i], seg_texts[i] = (orig_s, orig_e), ""
            continue

        # ASR
        try:
            result = asr_model.generate(
                input=seg_path, language="auto", ban_emo_unk=True, use_itn=False)
        except Exception:
            corrected_times[i], seg_texts[i] = (orig_s, orig_e), ""
            continue
        text = _TAG_PATTERN.sub("", str(result[0].get("text", ""))).strip() if result else ""
        seg_texts[i] = text

        # FSMN VAD (需 16kHz, 显式降采样)
        vad_s, vad_e = orig_s, orig_e
        try:
            import soundfile as _sf, librosa as _lr
            _data, _sr = _sf.read(seg_path)
            if _sr != 16000:
                _d16 = _lr.resample(_data, orig_sr=_sr, target_sr=16000)
                _p16 = seg_path.replace('.wav', '_16k.wav')
                _sf.write(_p16, _d16, 16000)
                _vr = vad_model.generate(input=_p16)
                try: os.remove(_p16)
                except: pass
            else:
                _vr = vad_model.generate(input=seg_path)
            if _vr and 'value' in _vr[0]:
                _segs = [s for s in _vr[0]['value'] if s[1] - s[0] >= 100]
                if _segs:
                    vad_s, vad_e = win_s + _segs[0][0], win_s + _segs[-1][1]
        except Exception:
            pass

        from .utils import calibrate_times as _ct
        cs, ce = _ct(vad_s, vad_e, orig_s, orig_e, win_s, win_e, _left_pad, _right_pad)
        if ce - cs > 50:
            corrected_times[i] = (cs, ce)
            changed_count += (cs != orig_s or ce != orig_e)

    return corrected_times, seg_texts, changed_count > 0
