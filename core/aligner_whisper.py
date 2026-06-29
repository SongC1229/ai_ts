"""faster-whisper 字幕区间矫正

镜像 aligner_qwen.py 的接口风格: 懒加载 + 用完即卸, 与 Qwen/WavLM 的显存策略一致。

与 Qwen3-ForcedAligner 的区别:
- Qwen 需要原声字幕文本做「强制对齐」, 无 raw_src_path 时不可用
- Whisper 做 ASR + 时间戳, 不需要任何字幕文本, 在无原声字幕时仍可矫正区间

硬编码参数 (符合「尽量不新增配置」约定):
  model:        large-v3
  device:       auto (cuda→cuda, 否则 cpu)
  compute_type: cuda → fp16 (float16, 与 Qwen 同为半精度)
                cpu  → int8  (fp16 在 CPU 上无加速且易出错)
"""

import os

from .utils import resolve_device


# ── 模型参数 (硬编码, 不走 config) ──
_MODEL_NAME = "large-v3"
# 推理精度: CUDA 走 fp16 (与 Qwen 同为半精度); CPU 走 int8 (fp16 在 CPU 上无加速且易出错)
_COMPUTE_TYPE_CPU = "int8"
_COMPUTE_TYPE_CUDA = "float16"  # = fp16

# 本地模型目录优先 (与项目 models/ 约定一致), 不存在则从 HuggingFace 下载
def _model_path() -> str:
    _local = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "faster-whisper-large-v3"
    )
    # 仅当目录包含 model.bin 时视为有效本地模型
    if os.path.isdir(_local) and os.path.isfile(os.path.join(_local, "model.bin")):
        return _local
    return _MODEL_NAME


# ── 模型状态 ──
_model = None
_loaded = False
_log_func = print  # 默认 print, 可由 set_log_cb 覆盖
_load_time = 0.0   # 模型加载耗时，供 pipeline 读取
_model_name = ""   # 模型名称
_model_device = "" # 推理设备


def set_log_cb(cb):
    """设置日志回调, 用于写入 UI + 日志文件"""
    global _log_func
    _log_func = cb


def _log(msg: str):
    _log_func(f"  [whisper] {msg}")


def _resolve_compute_type(device: str) -> str:
    """设备 → 推理精度"""
    return _COMPUTE_TYPE_CUDA if device == "cuda" else _COMPUTE_TYPE_CPU


def _load_model():
    """懒加载 faster-whisper 模型 (单例)

    返回 True 表示就绪, 失败抛异常。
    """
    global _model, _loaded, _load_time, _model_name, _model_device
    if _loaded and _model is not None:
        _load_time = 0.0  # 已加载,不计时
        return True

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ImportError(
            "需要 faster-whisper 库: pip install faster-whisper\n"
            f"原始错误: {e}"
        )

    # 强制离线模式，避免 HuggingFace 网络请求（即使模型已缓存）
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    device = resolve_device("auto")
    compute_type = _resolve_compute_type(device)
    src = _model_path()
    _name = os.path.basename(src) if os.path.isdir(src) else src
    # 导出供 pipeline 读取
    _model_name = _name
    _model_device = device
    import time as _t
    _t0 = _t.time()
    # download_root 指向项目内避免污染用户目录; 本地路径时忽略
    if os.path.isdir(src):
        _model = WhisperModel(src, device=device, compute_type=compute_type)
    else:
        _dl_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models"
        )
        _model = WhisperModel(
            src, device=device, compute_type=compute_type,
            download_root=_dl_root,
        )
    _loaded = True
    _load_time = _t.time() - _t0
    _log(f"就绪 ({_load_time:.1f}s)")
    return True


def unload_model():
    """卸载模型释放显存 (用完即卸, 与 Qwen/WavLM 一致)"""
    global _model, _loaded
    if _model is None:
        return
    _model = None
    _loaded = False
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    _log("模型已卸载, 显存已释放")


def is_loaded() -> bool:
    """检查模型是否已加载"""
    return _loaded and _model is not None


def transcribe_full(
    audio_path: str,
    language: str = "ja",
    vad_filter: bool = False,
    vad_threshold: float = 0.35,
    vad_min_silence_ms: int = 500,
    vad_speech_pad_ms: int = 200,
    beam_size: int = 5,
) -> tuple:
    """一次转写整条人声轨, 返回 (词表, 句子段)

    用 word_timestamps=True 拿词级时间戳, 比 segment 级更精确,
    供 pipeline 按字幕窗口映射首尾词。
    句子段 (segments) 供生成原声 .ja.srt 字幕。

    Args:
        audio_path:           人声轨 WAV 路径 (Demucs 产物 vocals_orig.wav)
        language:             语言代码, 默认 ja (日语)
        vad_filter:           VAD 预过滤 (True=开启, 由 faster-whisper 的 Silero VAD 预分段)
        vad_threshold:        语音概率阈值 (默认 0.35, 低于 faster-whisper 默认 0.5,
                              因 Demucs 纯人声无背景音, 降低阈值避免耳语/气声被滤掉)
        vad_min_silence_ms:   静音持续多久才算分段 (默认 500, 放宽避免短停顿切碎句子)
        vad_speech_pad_ms:    语音段前后补白 (默认 200, 避免首尾词时间戳失真)
        beam_size:            beam search 宽度

    Returns:
        (words, segments):
          words: [{"word": str, "start_ms": int, "end_ms": int, "prob": float}, ...]
          segments: [{"text": str, "start_ms": int, "end_ms": int}, ...]
          转写失败返回 ([], [])。
    """
    if not os.path.exists(audio_path):
        _log(f"音频不存在: {audio_path}")
        return [], []
    if not _load_model():
        return [], []

    import time as _t
    _t0 = _t.time()
    try:
        # vad_filter 由调用方控制
        segments, _info = _model.transcribe(
            audio_path, language=language,
            word_timestamps=True, vad_filter=vad_filter,
            beam_size=beam_size,
            vad_parameters={
                "threshold": vad_threshold,
                "min_silence_duration_ms": vad_min_silence_ms,
                "speech_pad_ms": vad_speech_pad_ms,
            },
        )
        words = []
        segs = []
        for seg in segments:
            # 句子段 (供生成原声 .ja.srt)
            if seg.start is not None and seg.end is not None:
                _seg_text = (seg.text or "").strip()
                if _seg_text:
                    segs.append({
                        "text": _seg_text,
                        "start_ms": int(seg.start * 1000),
                        "end_ms": int(seg.end * 1000),
                    })
            # 词级时间戳 (供字幕区间映射)
            if seg.words:
                for w in seg.words:
                    if w.start is None or w.end is None:
                        continue
                    words.append({
                        "word": w.word.strip(),
                        "start_ms": int(w.start * 1000),
                        "end_ms": int(w.end * 1000),
                        "prob": float(getattr(w, "probability", 0.0) or 0.0),
                    })
    except Exception as e:
        _log(f"转写失败: {e}")
        return [], []

    # 按起始时间排序 (保险)
    words.sort(key=lambda x: x["start_ms"])
    segs.sort(key=lambda x: x["start_ms"])
    _log(f"转写: {len(words)} 词, {_t.time()-_t0:.1f}s")
    return words, segs


def align_subs(subs, send_ranges, ctx, words, segments):
    """用 Whiper 词/段时间戳拟合字幕区间

    Args:
        subs: 字幕列表
        send_ranges: _compute_send_ranges 输出
        ctx: PipelineContext
        words: 词级时间戳
        segments: 段级时间戳

    Returns:
        (corrected_times, seg_texts, has_changes)
    """
    import bisect
    from .utils import calibrate_times as _ct

    _pad_ms = ctx.asr_pad_ms
    _seg_starts = [s["start_ms"] for s in segments] if segments else []
    _word_starts = [w["start_ms"] for w in words]
    corrected_times = {}
    seg_texts = {}
    changed_count = 0
    _seg_hit = 0
    _word_hit = 0

    def _fit_by_segment(win_s, win_e):
        """完全被拟合窗口包裹的 segment"""
        if not segments:
            return None
        _lo = bisect.bisect_left(_seg_starts, win_s)
        _hi = bisect.bisect_right(_seg_starts, win_e)
        _hits = []
        for j in range(_lo, min(_hi, len(segments))):
            _s = segments[j]
            if _s["start_ms"] >= win_s and _s["end_ms"] <= win_e:
                _hits.append(_s)
        if not _hits:
            return None
        return _hits[0]["start_ms"], _hits[-1]["end_ms"], "".join(h["text"] for h in _hits)

    def _fit_by_words(bound_s, bound_e):
        """原字幕区间内首尾词兜底"""
        _lo = bisect.bisect_left(_word_starts, bound_s)
        _hi = bisect.bisect_right(_word_starts, bound_e)
        _ww = words[_lo:_hi]
        if _ww:
            return _ww[0]["start_ms"], _ww[-1]["end_ms"]
        return None

    for i, sub in enumerate(subs):
        orig_s, orig_e = sub.start_ms, sub.end_ms
        win_s, win_e, _left_pad, _right_pad = send_ranges[i]

        _fit = _fit_by_segment(win_s, win_e)
        if _fit is not None:
            _first_s, _last_e, _seg_text = _fit
            seg_texts[i] = _seg_text
            _seg_hit += 1
        else:
            _wf = _fit_by_words(orig_s, orig_e)
            if _wf is not None:
                _first_s, _last_e = _wf
                _word_hit += 1
            else:
                corrected_times[i] = (orig_s, orig_e)
                continue

        cs, ce = _ct(_first_s, _last_e, orig_s, orig_e,
                      win_s, win_e, _left_pad, _right_pad, _pad_ms)
        dur_ms = ce - cs
        if dur_ms > 50:
            corrected_times[i] = (cs, ce)
            changed_count += (cs != orig_s or ce != orig_e)

    return corrected_times, seg_texts, changed_count > 0
