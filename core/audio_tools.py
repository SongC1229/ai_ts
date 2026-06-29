"""音频工具模块 — ffmpeg 音频提取/分割/混音"""

import json
import subprocess
import os
import re
from dataclasses import dataclass

from .utils import read_wav_np, write_wav_np

# 预编译正则（ffmpeg silencedetect 回退路径)
_SILENCE_END_RE = re.compile(r'silence_end:\s*([\d.]+)')
_SILENCE_START_RE = re.compile(r'silence_start:\s*([\d.]+)')


@dataclass(frozen=True)
class AudioInfo:
    """音频文件元信息（由 get_audio_info 返回)"""
    duration_ms: int   # 时长（毫秒)
    sample_rate: int   # 采样率（Hz)
    channels: int      # 声道数（0=未知)


def get_audio_info(filepath: str, max_rate: int = 48000) -> AudioInfo:
    """一次性获取音频 duration_ms / sample_rate / channels

    - WAV 文件直接从头部读取,无需 ffprobe
    - 非 WAV 或头部读取失败时回退到 ffprobe
    - 结果按 (filepath, mtime, max_rate) 缓存,避免重复探测

    Returns:
        AudioInfo：duration_ms=0 / sample_rate=0 / channels=0 表示探测失败
    """
    _cache = getattr(get_audio_info, '_cache', {})
    try:
        _mtime = os.path.getmtime(filepath)
    except OSError:
        _mtime = None
    _key = (filepath, _mtime, max_rate)
    if _key in _cache:
        return _cache[_key]

    duration_ms = 0
    sample_rate = 0
    channels = 0

    # WAV：直接从头部读取
    if filepath.lower().endswith('.wav'):
        try:
            import wave as _w
            with _w.open(filepath, 'rb') as wf:
                sr = wf.getframerate()
                n = wf.getnframes()
                ch = wf.getnchannels()
            if sr > 0:
                duration_ms = int(n * 1000 / sr)
                sample_rate = sr
                channels = ch
        except Exception:
            pass

    # 非 WAV 或 WAV 头读取失败 → ffprobe 一次获取
    if sample_rate <= 0 or duration_ms <= 0:
        try:
            # 用一条 ffprobe 同时取 duration 和 sample_rate、channels
            cmd_dur = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration:stream=sample_rate,channels',
                '-of', 'default=noprint_wrappers=1:nokey=0',
                filepath
            ]
            r = _run_ffmpeg(cmd_dur)
            _out = r.stdout or ""
            _sr_found = 0
            _ch_found = 0
            _dur_found = 0.0
            for _line in _out.splitlines():
                _line = _line.strip()
                if _line.startswith('duration='):
                    try:
                        _dur_found = float(_line.split('=', 1)[1])
                    except ValueError:
                        pass
                elif _line.startswith('sample_rate='):
                    try:
                        _sr_found = int(_line.split('=', 1)[1])
                    except ValueError:
                        pass
                elif _line.startswith('channels='):
                    try:
                        _ch_found = int(_line.split('=', 1)[1])
                    except ValueError:
                        pass
            if _dur_found > 0:
                duration_ms = int(_dur_found * 1000)
            if _sr_found > 0:
                sample_rate = _sr_found
            if _ch_found > 0:
                channels = _ch_found
        except Exception:
            pass

    # 采样率上限截断（保持与原 detect_audio_sample_rate 行为一致)
    if sample_rate > max_rate:
        sample_rate = max_rate
    # 采样率兜底：探测失败默认 48kHz（与原 detect_audio_sample_rate 一致)
    if sample_rate <= 0:
        sample_rate = 48000

    info = AudioInfo(duration_ms=duration_ms, sample_rate=sample_rate, channels=channels)
    _cache[_key] = info
    get_audio_info._cache = _cache
    return info


def _run_ffmpeg(cmd):
    """运行 ffmpeg 命令,失败时抛出详细错误"""
    try:
        # 添加 -hide_banner 屏蔽 MP3 时长估算等冗余警告
        if cmd[0] == 'ffmpeg':
            cmd = [cmd[0], '-hide_banner'] + cmd[1:]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True,
                                encoding='utf-8', errors='replace')
        return result
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr[:2000] if e.stderr else str(e)
        raise RuntimeError(
            f"ffmpeg 命令执行失败 (exit {e.returncode}):\n"
            f"  {' '.join(cmd[:6])}...\n"
            f"  {err_msg}"
        ) from e


def split_audio_by_time(
    audio_path: str,
    start_ms: int,
    end_ms: int,
    output_path: str,
    channels: int = None
) -> str:
    """按时间裁剪音频片段（保持原始采样率,上限 48kHz)

    Args:
        channels: 输出声道数,None 表示保持输入原始声道数
    """
    start_sec = start_ms / 1000.0
    duration_sec = (end_ms - start_ms) / 1000.0
    sr = get_audio_info(audio_path, max_rate=48000).sample_rate
    if channels is None:
        # 自动检测输入声道数
        try:
            r = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json',
                         '-show_streams', audio_path],
                        capture_output=True, text=True, timeout=10)
            info = json.loads(r.stdout)
            channels = info['streams'][0].get('channels', 1)
        except Exception:
            channels = 1
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_sec),    # 放在 -i 前面,快速 seek
        '-i', audio_path,
        '-t', str(duration_sec),
        '-acodec', 'pcm_s16le',
        '-ar', str(sr),
        '-ac', str(channels),
        output_path
    ]
    _run_ffmpeg(cmd)
    return output_path


def split_audio_np(audio_path: str, start_ms: int, end_ms: int, output_path: str):
    """按时间裁剪音频片段,使用 soundfile seek 只读需要的部分"""
    import soundfile as sf
    with sf.SoundFile(audio_path) as f:
        sr = f.samplerate
        s = int(start_ms * sr / 1000)
        e = int(end_ms * sr / 1000)
        s = max(0, min(s, len(f)))
        e = max(s, min(e, len(f)))
        if e <= s:
            raise ValueError(f"空片段: {audio_path} [{start_ms}, {end_ms})")
        f.seek(s)
        data = f.read(e - s)
    sf.write(output_path, data, sr)


def pad_audio_np(input_path: str, output_path: str, front_ms: int = 0, back_ms: int = 0):
    """在音频前后添加静音,使用 soundfile + numpy 代替 ffmpeg"""
    if front_ms <= 0 and back_ms <= 0:
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path
    import soundfile as sf
    import numpy as np
    data, sr = sf.read(input_path)
    if front_ms > 0:
        pad_len = int(sr * front_ms / 1000)
        nch = data.shape[1] if data.ndim > 1 else 1
        pad = np.zeros((pad_len, nch) if nch > 1 else pad_len, dtype=data.dtype)
        data = np.concatenate([pad, data])
    if back_ms > 0:
        pad_len = int(sr * back_ms / 1000)
        nch = data.shape[1] if data.ndim > 1 else 1
        pad = np.zeros((pad_len, nch) if nch > 1 else pad_len, dtype=data.dtype)
        data = np.concatenate([data, pad])
    sf.write(output_path, data, sr)
    return output_path




def vad_trim_silence(
    input_path: str,
    output_path: str,
    silence_thresh: str = "-45dB",
    ref_audio_path: str = "",
    pre_speech_start_ms: int = 0,
) -> tuple:
    """联合 add_leading_silence 处理前导静音：只裁 TTS 比参考多出的前导静音

    - 检测 TTS 和参考音频的前导静音长度
    - 仅裁剪 TTS 比参考多出的那部分（保留与参考等长的前导静音)
    - 尾部静音不裁剪,完整保留
    - 如果提供 pre_speech_start_ms,跳过 TTS 的 VAD 检测,直接使用该值

    Returns:
        (output_path, info_dict) info_dict 包含:
            orig_duration_ms, speech_start_ms, speech_end_ms,
            trim_start_ms, trim_end_ms,
            leading_trimmed_ms, trailing_cut_ms (=0 不截尾)
    """
    _info = {
        "orig_duration_ms": 0, "speech_start_ms": 0, "speech_end_ms": 0,
        "trim_start_ms": 0, "trim_end_ms": 0,
        "leading_trimmed_ms": 0, "trailing_cut_ms": 0,
    }
    try:
        if pre_speech_start_ms > 0:
            # 使用预先提供的语音起始位置,跳过 VAD 检测
            dur_ms = get_audio_info(input_path).duration_ms
            tts_start_ms = pre_speech_start_ms
        else:
            info = vad_detect_speech(input_path, silence_thresh)
            if not isinstance(info, dict) or info.get("speech_ms", 0) < 50:
                import shutil
                shutil.copy2(input_path, output_path)
                return output_path, _info
            dur_ms = info.get("duration_ms", 0)
            tts_start_ms = info.get("start_ms", 0)

        # 检测参考音频的前导静音长度
        ref_start_ms = 0
        if ref_audio_path and os.path.exists(ref_audio_path):
            try:
                ref_info = vad_detect_speech(ref_audio_path, silence_thresh)
                ref_start_ms = ref_info.get("start_ms", 0) if isinstance(ref_info, dict) else 0
            except Exception:
                pass

        # 只裁 TTS 比参考多出的前导静音
        trim_ms = max(0, tts_start_ms - ref_start_ms)
        start_sec = trim_ms / 1000.0

        # speech_end_ms / trim_end_ms: 不截尾,语音尾部延伸到音频末尾
        _speech_end = dur_ms
        _info.update({
            "orig_duration_ms": dur_ms,
            "speech_start_ms": tts_start_ms,
            "speech_end_ms": _speech_end,
            "trim_start_ms": trim_ms,
            "trim_end_ms": _speech_end,
            "leading_trimmed_ms": trim_ms,
            "trailing_cut_ms": 0,
        })
    except Exception:
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path, _info
    sr = get_audio_info(input_path).sample_rate
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-af', f'atrim=start={start_sec}',
        '-acodec', 'pcm_s16le', '-ar', str(sr), '-ac', '1',
        output_path
    ]
    _run_ffmpeg(cmd)
    return output_path, _info


def vad_detect_speech(
    input_path: str,
    silence_thresh: str = "-45dB"
) -> dict:
    """检测音频中语音的起始和结束位置（毫秒)

    WAV 文件使用 numpy 能量阈值检测（无子进程),非 WAV 回退到 ffmpeg。

    Returns:
        {"start_ms": int, "end_ms": int, "duration_ms": int, "speech_ms": int,
         "segments": [(start_ms, end_ms), ...]} — segments 为所有语音段列表
    """
    info = {"start_ms": 0, "end_ms": 0, "duration_ms": 0, "speech_ms": 0, "segments": []}

    # 获取总时长
    try:
        info["duration_ms"] = int(get_audio_info(input_path).duration_ms)
    except Exception:
        pass

    # 解析 dB 阈值为线性值
    try:
        _db = float(silence_thresh.replace("dB", ""))
    except (ValueError, AttributeError):
        _db = -35.0
    _lin_thresh = 10 ** (_db / 20.0)  # 振幅阈值

    # WAV 文件用 numpy 能量检测（无子进程)
    if input_path.lower().endswith('.wav'):
        try:
            speech_segs = _vad_numpy(input_path, _lin_thresh, info["duration_ms"])
        except Exception:
            return info
    else:
        # 非 WAV 回退到 ffmpeg silencedetect
        try:
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-af', f'silencedetect=noise={silence_thresh}:d=0.05',
                '-f', 'null', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
            silence_ends = [float(m) for m in _SILENCE_END_RE.findall(result.stderr)]
            silence_starts = [float(m) for m in _SILENCE_START_RE.findall(result.stderr)]
        except Exception:
            return info

        # 从静音起止重建语音段
        speech_segs = []
        if silence_starts and silence_ends:
            _marks = []
            _si, _ei = 0, 0
            while _si < len(silence_starts) and _ei < len(silence_ends):
                if silence_starts[_si] <= silence_ends[_ei]:
                    _marks.append(('start', silence_starts[_si]))
                    _si += 1
                else:
                    _marks.append(('end', silence_ends[_ei]))
                    _ei += 1
            for i in range(_si, len(silence_starts)):
                _marks.append(('start', silence_starts[i]))
            for i in range(_ei, len(silence_ends)):
                _marks.append(('end', silence_ends[i]))
            _last_end = 0.0
            for _kind, _t in _marks:
                if _kind == 'start' and _t > _last_end:
                    speech_segs.append((_last_end, _t))
                elif _kind == 'end':
                    _last_end = max(_last_end, _t)
            _total_end = info.get("duration_ms", 0) / 1000.0
            if _last_end < _total_end:
                speech_segs.append((_last_end, _total_end))
        elif not silence_starts:
            speech_segs.append((0, info.get("duration_ms", 0) / 1000.0))

    # 合并或删除过短片段
    if speech_segs:
        merged = [speech_segs[0]]
        for seg in speech_segs[1:]:
            if seg[0] - merged[-1][1] < 0.15:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(seg)
        speech_segs = [s for s in merged if (s[1] - s[0]) * 1000 >= 50]

    info["segments"] = [(int(s[0] * 1000), int(s[1] * 1000)) for s in speech_segs]
    if speech_segs:
        info["start_ms"] = int(speech_segs[0][0] * 1000)
        info["end_ms"] = int(speech_segs[-1][1] * 1000)
        info["speech_ms"] = info["end_ms"] - info["start_ms"]
    else:
        info["start_ms"] = 0
        info["end_ms"] = 0
        info["speech_ms"] = 0

    return info


def _vad_numpy(wav_path: str, lin_thresh: float, duration_ms: int) -> list:
    """用 numpy 能量阈值检测语音段（仅适用于 WAV 文件)

    Args:
        wav_path: WAV 文件路径
        lin_thresh: 线性振幅阈值（-35dB ≈ 0.0178)
        duration_ms: 文件总时长（毫秒)

    Returns:
        [(start_sec, end_sec), ...] 语音段列表
    """
    import numpy as np

    y, sr = read_wav_np(wav_path)
    if len(y) == 0:
        return []

    # 分帧计算 RMS 能量（20ms 帧)
    frame_len = int(sr * 0.020)
    n_frames = len(y) // frame_len
    if n_frames == 0:
        return [(0, duration_ms / 1000.0)] if duration_ms > 0 else []

    frames = y[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    is_speech = rms > lin_thresh

    # 从 bool 数组提取语音段
    speech_segs = []
    in_speech = False
    start_frame = 0
    for i, s in enumerate(is_speech):
        if s and not in_speech:
            in_speech = True
            start_frame = i
        elif not s and in_speech:
            in_speech = False
            speech_segs.append((start_frame * frame_len / sr, i * frame_len / sr))
    if in_speech:
        speech_segs.append((start_frame * frame_len / sr, len(y) / sr))

    if not speech_segs:
        # 无静音段,整个文件都是语音
        return [(0, duration_ms / 1000.0)] if duration_ms > 0 else []

    return speech_segs


def add_leading_silence(
    input_path: str,
    output_path: str,
    ref_audio_path: str = "",
    mode: str = "原声对齐",
    margin_ms: int = 50,
) -> tuple[str, int]:
    """在 TTS 音频前添加前导静音,对齐原声起始偏移

    Args:
        input_path: VAD 修剪后的 TTS 音频
        output_path: 输出路径
        ref_audio_path: 原始人声参考音频（vocals_clip),用于检测原声起始
        mode: "原声对齐" — 参考原声前导静音长度补足
              "字幕对齐" — 固定补充 margin_ms 的前导静音
        margin_ms: 字幕对齐模式下的固定留白毫秒数

    Returns:
        (output_path, pad_ms) — pad_ms 为实际补充的静音毫秒数
    """
    import shutil
    if mode == "字幕对齐":
        # 固定补充 margin_ms 前导静音
        if margin_ms <= 0:
            shutil.copy2(input_path, output_path)
            return output_path, 0
        sr = get_audio_info(input_path).sample_rate
        silence_samples = int(sr * margin_ms / 1000)
        import wave as _w
        with _w.open(input_path, 'rb') as wf:
            params = wf.getparams()
            frames = wf.readframes(wf.getnframes())
        frame_bytes = params.sampwidth * params.nchannels
        silence = b'\x00' * (silence_samples * frame_bytes)
        with _w.open(output_path, 'wb') as wf:
            wf.setparams(params)
            wf.writeframes(silence + frames)
        return output_path, margin_ms

    # 原声对齐：检测原始音频前导静音长度,补足到相同
    if not ref_audio_path or not os.path.exists(ref_audio_path):
        shutil.copy2(input_path, output_path)
        return output_path, 0

    ref_info = vad_detect_speech(ref_audio_path)
    tts_info = vad_detect_speech(input_path)
    if not ref_info or ref_info.get("speech_ms", 0) < 50:
        shutil.copy2(input_path, output_path)
        return output_path, 0
    if not tts_info or tts_info.get("speech_ms", 0) < 50:
        shutil.copy2(input_path, output_path)
        return output_path, 0

    # 原声前导静音长度 = 语音起始位置
    ref_start_ms = ref_info.get("start_ms", 0)
    tts_start_ms = tts_info.get("start_ms", 0)
    pad_needed_ms = ref_start_ms - tts_start_ms

    sr = get_audio_info(input_path).sample_rate
    import wave as _w
    with _w.open(input_path, 'rb') as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())
    frame_bytes = params.sampwidth * params.nchannels

    if pad_needed_ms > 0:
        # TTS 起始早于原声 → 在前部补充静音
        silence_samples = int(sr * pad_needed_ms / 1000)
        silence = b'\x00' * (silence_samples * frame_bytes)
        with _w.open(output_path, 'wb') as wf:
            wf.setparams(params)
            wf.writeframes(silence + frames)
        return output_path, pad_needed_ms
    else:
        # 无需处理（vad_trim_silence 已保证 TTS 前导不超出原始)
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path, 0


# ── 新流程: 片段级混音与拼接 ──────────────────────────────────


def match_rms_gain(tts_path: str, ref_vocal_path: str, output_path: str) -> tuple:
    """调整 TTS 增益使其 RMS 与原始人声严格匹配（误差 < 0.5dB)

    Args:
        tts_path: TTS 合成音频（VAD 对齐后)
        ref_vocal_path: 原始人声片段（音量参考)
        output_path: 输出路径

    Returns:
        (output_path, info_dict)  info_dict 包含:
            gain_db   — 施加的增益（dB),0 表示未调整
            rms_tts   — TTS 原始 RMS
            rms_ref   — 参考人声 RMS
            applied   — 是否实际写入了增益调整后的音频
    """
    import shutil as _shutil
    _empty = {"gain_db": 0.0, "rms_tts": 0.0, "rms_ref": 0.0, "applied": False}
    import numpy as np
    try:
        y_tts, sr_tts = read_wav_np(tts_path)
        y_ref, _ = read_wav_np(ref_vocal_path)

        if len(y_tts) == 0 or len(y_ref) == 0:
            _shutil.copy2(tts_path, output_path)
            return output_path, dict(_empty)

        # 截取重叠长度,确保 RMS 在同一时间区间上比较
        _min_len = min(len(y_tts), len(y_ref))
        y_tts_trim = y_tts[:_min_len]
        y_ref_trim = y_ref[:_min_len]

        # 各自去除静音段,仅在有声部分上对比 RMS
        # 阈值沿用 VAD 检测的 -35dB（≈ 线性幅值 0.0178)
        y_tts_active, y_ref_active = y_tts_trim, y_ref_trim
        _vad_thresh = 10 ** (-35 / 20)  # -35dB → 线性幅值
        for _y, _store in [(y_tts_trim, "tts"), (y_ref_trim, "ref")]:
            _mask = np.abs(_y) > _vad_thresh
            if np.any(_mask):
                _idx = np.where(_mask)[0]
                _seg = _y[_idx[0]:_idx[-1] + 1]
                if _store == "tts":
                    y_tts_active = _seg
                else:
                    y_ref_active = _seg

        rms_tts = float(np.sqrt(np.mean(y_tts_active ** 2))) if len(y_tts_active) > 0 else 0.0
        rms_ref = float(np.sqrt(np.mean(y_ref_active ** 2))) if len(y_ref_active) > 0 else 0.0

        if rms_tts < 1e-8 or rms_ref < 1e-8:
            _shutil.copy2(tts_path, output_path)
            return output_path, {"gain_db": 0.0, "rms_tts": rms_tts, "rms_ref": rms_ref, "applied": False}

        gain_db = 20 * np.log10(rms_ref / rms_tts)

        if abs(gain_db) > 0.3:
            y_out = y_tts * (10 ** (gain_db / 20))
            write_wav_np(output_path, y_out, sr_tts)
            return output_path, {"gain_db": float(gain_db), "rms_tts": rms_tts, "rms_ref": rms_ref, "applied": True}
        else:
            _shutil.copy2(tts_path, output_path)
            return output_path, {"gain_db": float(gain_db), "rms_tts": rms_tts, "rms_ref": rms_ref, "applied": False}
    except Exception as e:
        import sys
        print(f"[match_rms_gain] ERROR: {e}", file=sys.stderr)
        _shutil.copy2(tts_path, output_path)
        return output_path, dict(_empty)



def mix_segment_clip(
    tts_path: str,
    background_clip_path: str,
    output_path: str,
    edge_ms: int = 0,
) -> str:
    """混合单条 TTS 与背景音频片段（直接叠加,不做边界平滑)

    边界交叉淡变统一由 splice_segments_into_base 在拼接时处理。
    TTS 为单声道,背景保留原始声道数（立体声/单声道)。

    Args:
        tts_path: 增益已匹配的 TTS 片段
        background_clip_path: 从 background.wav 截取的对应片段
        output_path: 输出路径

    Returns:
        output_path
    """
    import numpy as np
    try:
        y_tts, sr = read_wav_np(tts_path)           # TTS → mono (1D)
        y_bg, sr_bg = read_wav_np(background_clip_path, mono=False)  # BG → 保留声道

        if len(y_tts) == 0 or len(y_bg) == 0:
            fallback_sr = get_audio_info(background_clip_path).sample_rate if os.path.exists(background_clip_path) else get_audio_info(tts_path).sample_rate
            cmd = [
                'ffmpeg', '-y',
                '-i', tts_path, '-i', background_clip_path,
                '-filter_complex',
                '[0:a][1:a]amix=inputs=2:duration=shortest[out]',
                '-map', '[out]',
                '-acodec', 'pcm_s16le', '-ar', str(fallback_sr), '-ac', '2',
                output_path
            ]
            _run_ffmpeg(cmd)
            return output_path

        # 统一采样率（以背景为准,TTS 升采样到背景采样率)
        if sr != sr_bg:
            tmp_resample = output_path + '.resample.tmp.wav'
            _run_ffmpeg([
                'ffmpeg', '-y', '-i', tts_path,
                '-ar', str(sr_bg), '-ac', '1', '-acodec', 'pcm_s16le',
                tmp_resample
            ])
            y_tts, sr = read_wav_np(tmp_resample)
            os.remove(tmp_resample)

        # 统一长度：以较长的一方为准
        # （TTS 短 → 尾部补静音,保留完整背景,确保混音片段≥字幕区间时长)
        bg_len = len(y_bg)
        tts_len = len(y_tts)
        if bg_len < tts_len:
            # 背景补零到 TTS 长度（保持声道数)
            pad_shape = (tts_len - bg_len,) + (y_bg.shape[1:] if y_bg.ndim > 1 else ())
            y_bg = np.concatenate([y_bg, np.zeros(pad_shape)])
        elif bg_len > tts_len:
            # TTS 尾部补静音到背景长度（保留尾部背景音乐)
            pad_shape = (bg_len - tts_len,)
            y_tts = np.concatenate([y_tts, np.zeros(pad_shape)])

        # 如果背景是立体声,将单声道 TTS 广播到双声道再叠加
        if y_bg.ndim == 2 and y_bg.shape[1] == 2:
            y_tts_stereo = np.column_stack([y_tts, y_tts])
            y_mixed = y_tts_stereo + y_bg
        else:
            y_mixed = y_tts + y_bg  # 双方都是 mono

        write_wav_np(output_path, y_mixed, sr)
        return output_path
    except Exception as e:
        import sys
        print(f"[mix_segment_clip] ERROR: {e}", file=sys.stderr)
        # 回退（以背景采样率为准)
        fallback_sr = get_audio_info(background_clip_path).sample_rate if os.path.exists(background_clip_path) else get_audio_info(tts_path).sample_rate
        cmd = [
            'ffmpeg', '-y',
            '-i', tts_path, '-i', background_clip_path,
            '-filter_complex',
            '[0:a][1:a]amix=inputs=2:duration=shortest[out]',
            '-map', '[out]',
            '-acodec', 'pcm_s16le', '-ar', str(fallback_sr), '-ac', '2',
            output_path
        ]
        _run_ffmpeg(cmd)
        return output_path


def splice_segments_into_base(
    base_audio_path: str,
    segments: list,
    output_path: str,
    crossfade_ms: int = 40,
    crossfade_info: list = None,
) -> str:
    """以原始音频为基底,将处理后的片段替换到对应区间,边界交叉淡变

    替换段前后各 crossfade_ms 做原始音频与替换片段的双向交叉淡变（实际宽度会被
    seg_len//4 与相邻片段间距夹断)。

    Args:
        crossfade_info: 若传入列表,函数会向其追加每段实际淡变信息
            (start_ms, end_ms, L_in_ms, L_out_ms),供调用方打印真实淡变宽度。
    """
    import numpy as np
    import shutil

    if not segments:
        shutil.copy2(base_audio_path, output_path)
        return output_path

    try:
        y_base, sr = read_wav_np(base_audio_path, mono=False)
        if len(y_base) == 0:
            shutil.copy2(base_audio_path, output_path)
            return output_path
        nch = y_base.ndim

        y_result = y_base  # 直接复用,无需 copy（y_base 不再单独使用)
        del y_base  # 释放引用,帮助 GC
        L = int(crossfade_ms * sr / 1000.0)

        sorted_segments = sorted(segments, key=lambda x: x[0])
        prev_end = 0

        for i, (start_ms, end_ms, clip_path) in enumerate(sorted_segments):
            if not clip_path or not os.path.exists(clip_path):
                continue

            y_clip, sr_clip = read_wav_np(clip_path, mono=False)
            if len(y_clip) == 0:
                continue

            if sr_clip != sr:
                tmp = output_path + '.splice_resample.tmp.wav'
                _run_ffmpeg([
                    'ffmpeg', '-y', '-i', clip_path,
                    '-ar', str(sr), '-ac', str(max(1, nch)), '-acodec', 'pcm_s16le',
                    tmp
                ])
                y_clip, _ = read_wav_np(tmp, mono=False)
                os.remove(tmp)

            s = int(start_ms * sr / 1000.0)
            e = int(end_ms * sr / 1000.0)
            s = max(0, min(s, len(y_result)))
            e = max(s, min(e, len(y_result)))
            seg_len = e - s
            if seg_len <= 0:
                continue

            # 将 clip 裁剪/填充到 seg_len
            clip_len = len(y_clip)
            if clip_len > seg_len:
                y_clip = y_clip[:seg_len]
            elif clip_len < seg_len:
                pad_shape = (seg_len - clip_len,) + (y_clip.shape[1:] if nch > 1 else ())
                y_clip = np.concatenate([y_clip, np.zeros(pad_shape)])

            # 计算安全的淡变长度,避免与相邻片段重叠
            L_in = min(L, seg_len // 4)
            if s - L_in < prev_end:
                L_in = max(0, s - prev_end)

            next_start = len(y_result)
            for j in range(i + 1, len(sorted_segments)):
                ns = int(sorted_segments[j][0] * sr / 1000.0)
                if ns >= e:
                    next_start = ns
                    break
            L_out = min(L, seg_len // 4)
            if e + L_out > next_start:
                L_out = max(0, next_start - e)

            # 记录实际淡变宽度（ms),供调用方打印
            if crossfade_info is not None:
                crossfade_info.append((start_ms, end_ms,
                                       round(L_in * 1000.0 / sr),
                                       round(L_out * 1000.0 / sr)))

            # 保存替换区间前后的原始音频（含边界交叉淡变用)
            orig_head = y_result[s - L_in:s + L_in].copy() if L_in > 0 and s >= L_in else np.array([])
            orig_tail = y_result[e - L_out:e + L_out].copy() if L_out > 0 and e + L_out <= len(y_result) else np.array([])

            # 声道适配：clip 是单声道但基底是立体声时复制声道
            if nch > 1 and y_clip.ndim == 1:
                y_clip = np.column_stack([y_clip, y_clip])
            elif nch == 1 and y_clip.ndim > 1:
                y_clip = y_clip.mean(axis=1)

            # ── 写入 clip ──
            y_result[s:e] = y_clip

            # ── 段前交叉淡变 [s-L_in, s+L_in)：原始淡出 + clip 淡入 ──
            if len(orig_head) > 1:
                fade_out = np.linspace(1.0, 0.0, len(orig_head)).reshape(-1, 1) if nch > 1 else np.linspace(1.0, 0.0, len(orig_head))
                fade_in  = np.linspace(0.0, 1.0, len(orig_head)).reshape(-1, 1) if nch > 1 else np.linspace(0.0, 1.0, len(orig_head))
                y_result[s - L_in:s + L_in] = orig_head * fade_out + y_result[s - L_in:s + L_in] * fade_in

            # ── 段后交叉淡变 [e-L_out, e+L_out)：clip 淡出 + 原始淡入 ──
            if len(orig_tail) > 1:
                fade_out = np.linspace(1.0, 0.0, len(orig_tail)).reshape(-1, 1) if nch > 1 else np.linspace(1.0, 0.0, len(orig_tail))
                fade_in  = np.linspace(0.0, 1.0, len(orig_tail)).reshape(-1, 1) if nch > 1 else np.linspace(0.0, 1.0, len(orig_tail))
                y_result[e - L_out:e + L_out] = y_result[e - L_out:e + L_out] * fade_out + orig_tail * fade_in

            # prev_end 包含尾淡变区域,防止下一片段头淡变与之重叠
            prev_end = e + L_out

        write_wav_np(output_path, y_result, sr)
        return output_path
    except Exception as e:
        import sys
        print(f"[splice_segments_into_base] ERROR: {e}", file=sys.stderr)
        shutil.copy2(base_audio_path, output_path)
        return output_path


def is_low_energy(audio_path: str, threshold: float = 0.005) -> bool:
    """检测音频是否低能量（静音/近乎静音)

    使用 VAD 检测语音段,仅在有声区间计算 RMS 能量,
    避免前导/尾部静音拉低整体 RMS。

    Args:
        audio_path: 音频文件路径（仅 WAV)
        threshold: 语音段 RMS 能量阈值,低于此值视为静音（默认 0.005)

    Returns:
        True 表示低能量（静音),False 表示有声音活动
    """
    import numpy as _np
    try:
        speech_segs = _vad_numpy(audio_path, 10 ** (-45 / 20.0), 0)
        if not speech_segs:
            return True  # 无语音段 → 静音

        data, sr = read_wav_np(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)

        # 仅在有声区间计算 RMS
        speech_parts = []
        for seg_start, seg_end in speech_segs:
            s = int(seg_start * sr)
            e = int(seg_end * sr)
            speech_parts.append(data[s:e])

        speech_data = _np.concatenate(speech_parts) if len(speech_parts) > 1 else speech_parts[0]
        rms = _np.sqrt(_np.mean(speech_data ** 2))
        return rms < threshold
    except Exception:
        return False


def get_rms_db(audio_path: str) -> float:
    """计算音频 RMS 分贝值（仅 VAD 有声区间)

    Args:
        audio_path: 音频文件路径（仅 WAV)

    Returns:
        RMS 分贝值（dBFS),静音返回 -inf
    """
    import numpy as _np
    try:
        speech_segs = _vad_numpy(audio_path, 10 ** (-45 / 20.0), 0)
        if not speech_segs:
            return float('-inf')

        data, sr = read_wav_np(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)

        speech_parts = []
        for seg_start, seg_end in speech_segs:
            s = int(seg_start * sr)
            e = int(seg_end * sr)
            speech_parts.append(data[s:e])

        speech_data = _np.concatenate(speech_parts) if len(speech_parts) > 1 else speech_parts[0]
        rms = _np.sqrt(_np.mean(speech_data ** 2))
        if rms <= 0:
            return float('-inf')
        return float(20 * _np.log10(rms))
    except Exception:
        return float('-inf')
