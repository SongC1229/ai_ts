"""SenseVoice 字幕对齐 — ASR 转写生成原声字幕文本

流程:
  1. 对每条字幕,使用 _compute_send_ranges 预计算的扩展窗口提取音频
  2. SenseVoiceSmall ASR 识别文字
  3. FSMN VAD 检测语音起止区间用于时间校准
  4. 更新字幕文本 + 校准时间

与 Qwen/Whisper 共享同一套送检区间规则 (max_pad=500ms, safe_gap=200ms)。
"""

import os
import re
import time

import core.editdistance_fallback  # noqa: F401 (editdistance 纯 Python 回退)

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
            trust_remote_code=False,
            disable_update=True,
            device="cuda:0",
        )
    if _vad_model is None:
        from funasr import AutoModel
        _vad_model = AutoModel(
            model="fsmn-vad",
            trust_remote_code=False,
            disable_update=True,
            device="cuda:0",
        )
    return _asr_model, _vad_model


def align_subs(subs, vocals_path: str, send_ranges: list, ctx, progress_cb=None):
    """用 SenseVoice + FSMN VAD 对字幕做 ASR 转写 + VAD 校准

    Args:
        subs: 字幕列表 (SubtitleItem)
        vocals_path: 人声音频路径
        send_ranges: _compute_send_ranges 输出
        ctx: PipelineContext
        progress_cb: (done, total) 进度回调

    Returns:
        (corrected_times, seg_texts, has_changes)
    """
    from core.audio_tools import split_audio_np

    asr_model, vad_model = _get_models()
    total = len(subs)
    corrected_times = {}
    seg_texts = {}
    changed_count = 0
    work_dir = getattr(ctx, 'work_dir', os.path.join(os.getcwd(), "tmp"))

    for i, sub in enumerate(subs):
        if progress_cb:
            progress_cb(i + 1, total)

        orig_s = sub.start_ms
        orig_e = sub.end_ms
        idx = sub.idx

        win_s, win_e, _left_pad, _right_pad = send_ranges[i]

        seg_path = os.path.join(work_dir, f"_sensev_{idx:04d}.wav")
        try:
            split_audio_np(vocals_path, win_s, win_e, seg_path)
        except Exception:
            seg_texts[idx] = ""
            corrected_times[idx] = (orig_s, orig_e)
            continue

        if not os.path.exists(seg_path):
            seg_texts[idx] = ""
            corrected_times[idx] = (orig_s, orig_e)
            continue

        # ASR
        try:
            result = asr_model.generate(
                input=seg_path, language="auto",
                ban_emo_unk=True, use_itn=False,
            )
        except Exception:
            seg_texts[idx] = ""
            corrected_times[idx] = (orig_s, orig_e)
            continue

        text = _TAG_PATTERN.sub("", str(result[0].get("text", ""))).strip() if result else ""
        seg_texts[idx] = text

        # FSMN VAD (返回值 ms)
        cs, ce = orig_s, orig_e
        try:
            vad_result = vad_model.generate(input=seg_path)
            if vad_result and 'value' in vad_result[0]:
                segs = vad_result[0]['value']  # [[start_ms, end_ms], ...]
                if segs:
                    cs = win_s + segs[0][0]
                    ce = win_s + segs[-1][1]
        except Exception:
            pass

        # pad 约束 (与 Qwen/Whisper 一致)
        asr_pad_ms = getattr(ctx, 'asr_pad_ms', 100)
        cs = max(win_s, cs - asr_pad_ms)
        ce = min(win_e, ce + asr_pad_ms)
        if _left_pad > 0:
            cs = max(cs, orig_s - _left_pad)
        if _right_pad > 0:
            ce = min(ce, orig_e + _right_pad)

        ce = max(ce, orig_e)
        cs = min(cs, orig_s)

        dur_ms = ce - cs
        if dur_ms > 50:
            corrected_times[idx] = (cs, ce)
            changed_count += (cs != orig_s or ce != orig_e)

        try:
            os.remove(seg_path)
        except Exception:
            pass

    return corrected_times, seg_texts, changed_count > 0
