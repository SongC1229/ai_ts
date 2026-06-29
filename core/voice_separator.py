"""人声分离模块 — Demucs（流式处理)"""

import os
import sys
from pathlib import Path
from typing import Optional, Callable


def _run_demucs(audio_path, output_dir, model, device, threads, segment, overlap, progress_callback=None):
    """对单个音频运行 Demucs（超长音频自动分片以避免 OOM)"""
    import torch as _t
    _t.set_num_threads(threads)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)

    audio_name = Path(audio_path).stem
    demucs_out = os.path.join(output_dir, model, audio_name)
    os.makedirs(demucs_out, exist_ok=True)

    # 一次性探测采样率和时长（合并两次 ffprobe 为一次)
    import subprocess as _sp
    _info = _sp.run(
        ['ffprobe', '-v', '0', '-of', 'csv=p=0',
         '-select_streams', 'a:0',
         '-show_entries', 'stream=sample_rate',
         '-show_entries', 'format=duration',
         audio_path],
        capture_output=True, text=True
    )
    _lines = _info.stdout.strip().split('\n')
    try:
        _input_sr = int(_lines[0].strip())
    except (ValueError, TypeError, IndexError):
        _input_sr = 44100
    try:
        _total_sec = float(_lines[1].strip() if len(_lines) > 1 else '0')
    except (ValueError, TypeError, IndexError):
        _total_sec = 0

    _chunk_sec = 3600  # 1小时一片
    _overlap_sec = 2   # 拼接重叠 2s 用于交叉淡变

    if _total_sec > _chunk_sec:
        import tempfile
        import math
        import shutil
        import numpy as np
        import soundfile as _sf
        _total_frames = int(_total_sec * _input_sr + 0.5)  # 总采样数
        _chunk_frames = _chunk_sec * _input_sr              # 每块采样数
        _overlap_frames = _overlap_sec * _input_sr          # 重叠采样数
        _chunk_dir = tempfile.mkdtemp(prefix="demucs_chunks_")
        try:
            _n_chunks = math.ceil(_total_frames / _chunk_frames)
            # 将每片 Demucs 结果写入临时 WAV（避免全部驻留内存)
            _v_chunk_paths = []
            _nv_chunk_paths = []
            _v_sr = _input_sr  # 将在循环中更新
            for _ci in range(_n_chunks):
                _cs_frame = max(0, _ci * _chunk_frames - (_overlap_frames if _ci > 0 else 0))
                _ce_frame = min(_total_frames + _overlap_frames, (_ci + 1) * _chunk_frames + _overlap_frames)
                _cs = _cs_frame / _input_sr
                _ce = _ce_frame / _input_sr
                _chunk_path = os.path.join(_chunk_dir, f"chunk_{_ci:04d}.wav")
                _sp.run(['ffmpeg', '-y', '-i', audio_path,
                         '-ss', str(_cs), '-to', str(_ce),
                         '-acodec', 'pcm_s16le', '-ar', str(_input_sr), _chunk_path],
                        check=True, capture_output=True)
                _vp, _nvp = _run_demucs_single(
                    _chunk_path, output_dir, model, device, threads, segment, overlap,
                    _input_sr, None
                )
                if progress_callback:
                    progress_callback((_ci + 1) * 100 // _n_chunks)
                _v_data, _v_sr = _sf.read(_vp, dtype='float64')
                _nv_data, _ = _sf.read(_nvp, dtype='float64')
                # 立体声转单声道（Demucs 输出为立体声)
                if _v_data.ndim > 1:
                    _v_data = _v_data.mean(axis=1)
                if _nv_data.ndim > 1:
                    _nv_data = _nv_data.mean(axis=1)
                # 去重叠：取有效区间（使用 Demucs 输出实际采样率)
                _clip_start_s = _overlap_sec * _v_sr if _ci > 0 else 0
                _clip_end_s = -_overlap_sec * _v_sr if _ci < _n_chunks - 1 else None
                _v_data = _v_data[int(_clip_start_s):_clip_end_s] if _clip_end_s is not None else _v_data[int(_clip_start_s):]
                _nv_data = _nv_data[int(_clip_start_s):_clip_end_s] if _clip_end_s is not None else _nv_data[int(_clip_start_s):]
                # 写入临时文件而非驻留内存
                _v_cp = os.path.join(_chunk_dir, f"v_{_ci:04d}.wav")
                _nv_cp = os.path.join(_chunk_dir, f"nv_{_ci:04d}.wav")
                _sf.write(_v_cp, _v_data, _v_sr)
                _sf.write(_nv_cp, _nv_data, _v_sr)
                _v_chunk_paths.append(_v_cp)
                _nv_chunk_paths.append(_nv_cp)
                del _v_data, _nv_data  # 立即释放内存
                # 日志：每片 Demucs 前后的时长变化
                _in_dur = (_ce_frame - _cs_frame) / _input_sr
                _log_dur = int(_sf.info(_v_cp).frames / _v_sr) if _v_sr > 0 else 0
                print(f"  chunk {_ci}: {_cs:.0f}s-{_ce:.0f}s, Demucs {_in_dur:.0f}s→{_log_dur}s ({_log_dur - int(_in_dur):+.0f}s)")
                if progress_callback:
                    progress_callback((_ci + 1) * 100 // _n_chunks)
                # 清理 chunk 文件
                os.remove(_chunk_path)
            # 从磁盘读取并拼接（带交叉淡变)
            _vocals_full = _stitch_chunks_from_files(_v_chunk_paths, _v_sr)
            _novocals_full = _stitch_chunks_from_files(_nv_chunk_paths, _v_sr)
            # 校验并修正时长偏差：确保每段在原始坐标上对齐
            _target_frames = _total_frames  # 原始音频总采样数（含重叠修正)
            # 实际目标帧数需要转换为 _v_sr 域（Demucs 输出采样率 = 44100)
            # 重叠切除已在对齐时去除,最终输出应等于总时长
            _target_dur = _total_sec  # 原始时长（秒)
            _out_dur = len(_vocals_full) / _v_sr if _v_sr > 0 else 0
            _diff = _out_dur - _target_dur
            if abs(_diff) > 0.05:
                print(f"  ⚠️ 拼接后时长 {_out_dur:.3f}s ≠ 原始 {_target_dur:.3f}s（偏差 {_diff:+.3f}s),正在修正对齐...")
                _target_len = int(_target_dur * _v_sr)
                if len(_vocals_full) > _target_len:
                    _vocals_full = _vocals_full[:_target_len]
                    _novocals_full = _novocals_full[:_target_len]
                elif len(_vocals_full) < _target_len:
                    _pad = np.zeros(_target_len - len(_vocals_full), dtype=np.float64)
                    _vocals_full = np.concatenate([_vocals_full, _pad])
                    _novocals_full = np.concatenate([_novocals_full, _pad])
                print(f"  修正后: {len(_vocals_full)/_v_sr:.3f}s")
            else:
                print(f"  拼接对齐: 输入 {_target_dur:.0f}s → 输出 {_out_dur:.0f}s")
            _vp_path = os.path.join(demucs_out, 'vocals.wav')
            _nvp_path = os.path.join(demucs_out, 'no_vocals.wav')
            # 写入前重采样到原始采样率（Demucs 固定输出 44100Hz)
            if _v_sr != _input_sr:
                _tmp_v = _vp_path + ".44k.wav"
                _tmp_nv = _nvp_path + ".44k.wav"
                _sf.write(_tmp_v, _vocals_full, _v_sr)
                _sf.write(_tmp_nv, _novocals_full, _v_sr)
                for _src, _dst in [(_tmp_v, _vp_path), (_tmp_nv, _nvp_path)]:
                    _sp.run(['ffmpeg', '-y', '-i', _src, '-ar', str(_input_sr), _dst],
                            check=True, capture_output=True)
                    os.remove(_src)
            else:
                _sf.write(_vp_path, _vocals_full, _input_sr)
                _sf.write(_nvp_path, _novocals_full, _input_sr)
            _log = progress_callback or print
            _log(f"  Demucs 分片拼接完成: 输入 {_total_sec:.0f}s → 输出 {len(_vocals_full)/_input_sr:.0f}s")
            return _vp_path, _nvp_path
        finally:
            shutil.rmtree(_chunk_dir, ignore_errors=True)

    return _run_demucs_single(
        audio_path, output_dir, model, device, threads, segment, overlap, _input_sr, progress_callback)


def _stitch_chunks_from_files(chunk_paths: list, sr: int, crossfade_ms: int = 20):
    """从磁盘流式读取 chunk 文件并拼接（避免全部驻留内存)

    每次只在内存中保留 2 个 chunk（前一块尾部 + 当前块),
    对边界做线性交叉淡变,其余部分直接拷贝到预分配输出数组。

    Args:
        chunk_paths: WAV 文件路径列表（按顺序)
        sr: 采样率
        crossfade_ms: 交叉淡变毫秒数（默认 20ms)

    Returns:
        np.ndarray: 拼接后的单声道 float64 音频
    """
    import soundfile as _sf_read
    import numpy as np

    if not chunk_paths:
        return np.array([], dtype=np.float64)
    if len(chunk_paths) == 1:
        data, _ = _sf_read.read(chunk_paths[0], dtype='float64')
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data

    cf_len = int(crossfade_ms * sr / 1000)

    # 第一遍：只读元信息（帧数),不读数据
    chunk_lens = []
    for p in chunk_paths:
        info = _sf_read.info(p)
        chunk_lens.append(info.frames)

    # 计算交叉淡变重叠量与精确输出长度
    overlaps = [min(cf_len, chunk_lens[i], chunk_lens[i + 1])
                for i in range(len(chunk_lens) - 1)]
    total_out = sum(chunk_lens) - sum(overlaps)
    out = np.zeros(total_out, dtype=np.float64)

    # 第二遍：逐块读取,流式拼接（内存中最多同时 2 块)
    pos = 0
    prev_chunk = None
    for i, p in enumerate(chunk_paths):
        cur, _ = _sf_read.read(p, dtype='float64')
        if cur.ndim > 1:
            cur = cur.mean(axis=1)

        if i == 0:
            out[pos:pos + len(cur)] = cur
            pos += len(cur)
        else:
            cf = min(cf_len, len(prev_chunk), len(cur))
            if cf > 0:
                fade_out = np.linspace(1.0, 0.0, cf)
                fade_in = np.linspace(0.0, 1.0, cf)
                tail_start = pos - cf
                out[tail_start:pos] = out[tail_start:pos] * fade_out + cur[:cf] * fade_in
            rest = len(cur) - cf
            if rest > 0:
                out[pos:pos + rest] = cur[cf:]
                pos += rest

        del prev_chunk  # 释放前一块
        prev_chunk = cur

    del prev_chunk
    return out[:pos]




def _run_demucs_single(audio_path, output_dir, model, device, threads, segment, overlap, input_sr, progress_callback):
    """对单个音频片段运行 Demucs（内部函数,不分片)"""
    import torch as _t
    _t.set_num_threads(threads)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)

    audio_name = Path(audio_path).stem
    demucs_out = os.path.join(output_dir, model, audio_name)
    os.makedirs(demucs_out, exist_ok=True)

    # monkey-patch
    import soundfile as sf
    import demucs.audio as _da
    def _ps(wav, path, samplerate=None, sr=None, **kw):
        rate = samplerate or sr or input_sr
        sf.write(str(path), wav.cpu().numpy().T, rate)
    _da.save_audio = _ps
    import torchaudio as _ta
    def _pts(uri, audio, sample_rate=None, sr=None, **kw):
        rate = sample_rate or sr or input_sr
        sf.write(str(uri), audio.cpu().numpy().T, rate)
    _ta.save = _pts

    args = ['--two-stems', 'vocals', '-n', model,
            '--shifts', '1', '--segment', str(segment), '--overlap', str(overlap), '-o', output_dir]
    if device == "cpu":
        args.extend(['-d', 'cpu'])
    elif device == "cuda":
        args.extend(['-d', 'cuda'])
    args.append(audio_path)

    if progress_callback:
        import tqdm as _tq
        _oi = _tq.tqdm.__init__
        def _pi(self, it=None, **kw):
            _oi(self, it, **kw)
            self._tot = len(it) if hasattr(it, '__len__') else None
        _tq.tqdm.__init__ = _pi
        _ou = _tq.tqdm.update
        def _pu(self, n=1):
            _ou(self, n)
            if hasattr(self, '_tot') and self._tot:
                progress_callback(min(int(self.n * 100 / self._tot), 100))
        _tq.tqdm.update = _pu

    from demucs import separate
    old_argv = sys.argv
    try:
        sys.argv = ['demucs.separate'] + args
        separate.main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv
        from .utils import cleanup_cuda
        cleanup_cuda()

    vp = os.path.join(demucs_out, 'vocals.wav')
    nvp = os.path.join(demucs_out, 'no_vocals.wav')
    if not os.path.exists(vp) or not os.path.exists(nvp):
        raise RuntimeError("Demucs output not found")
    return vp, nvp


def separate_vocals(
    audio_path: str,
    output_dir: str,
    model: str = "htdemucs",
    device: str = "auto",
    threads: int = 4,
    segment: int = 7,
    overlap: float = 0.25,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> tuple[str, str]:
    """使用 Demucs 分离人声和背景音（流式处理)

    Args:
        audio_path: 输入音频路径
        output_dir: 输出目录
        model: Demucs 模型名
        device: 设备
        threads: CPU 线程数
        segment: Demucs 内部处理窗口长度（秒),htdemucs 最大 7
        overlap: 片段间重叠比例
        progress_callback: 进度回调

    Returns:
        (vocals_path, no_vocals_path)
    """
    audio_name = Path(audio_path).stem

    # 检查缓存
    demucs_out = os.path.join(output_dir, model, audio_name)
    vp = os.path.join(demucs_out, 'vocals.wav')
    nvp = os.path.join(demucs_out, 'no_vocals.wav')
    if os.path.exists(vp) and os.path.exists(nvp):
        return vp, nvp

    return _run_demucs(audio_path, output_dir, model, device, threads, segment, overlap, progress_callback)


