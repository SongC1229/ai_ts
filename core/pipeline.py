"""配音流水线 — 解耦分步架构

职责: 完整流水线 + 单步执行 + 单条 TTS 重生成 + 重新混音

每步独立：缓存检查 → 依赖检查 → 执行 → 标记完成。
部分缓存时自动跳过已完成的条目。

步骤:
  1. ExtractAudio   提取原始混合音频
  2. SeparateVocals Demucs 人声分离
  3. SubtitleProcess  字幕处理 + 性别标记 (Qwen对齐 或 直接使用字幕时间)
  4. TTSSynthesis   TTS API 合成（纯调用,不做音频处理)
  5. AudioMixAndMerge  混音 + 全长拼接
  6. VideoMerge      合并回视频

用法:
  ctx = PipelineContext(video_path, dst_srt_path, output_dir, cache_root)
  ctx.raw_src_path = raw_src_path  # 可选,用于时间对齐
  orchestrator = PipelineOrchestrator(ctx, progress_callback)
  orchestrator.run()
"""

import os
import json
import socket as _socket
import shutil
import time
import traceback
import subprocess as sp
import tempfile
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, as_completed
from urllib.parse import urlparse
import soundfile as _sf
import numpy as _np

from .cache_manager import CacheManager
from config import cfg
from .audio_tools import (
    split_audio_np,
    vad_trim_silence, vad_detect_speech,
    match_rms_gain, mix_segment_clip,
    add_leading_silence, pad_audio_np,
    get_audio_info,
    splice_segments_into_base,
    is_low_energy,
    get_rms_db,
)
from .srt_parser import parse_srt, SubtitleItem
from .voice_separator import separate_vocals
from .voice_similarity import compare_similarity
from .utils import resolve_device, cleanup_cuda, make_logger, get_threads, fmt_time
from .tts_client import TTSClient
from .cache_manager import Step

# ── 进度回调类型 ──────────────────────────────────────────
ProgressCB = Callable[[int, int, str], None]  # (step_index, 百分比 0-100, 状态文字)


class CancelledError(Exception):
    """流水线取消"""
    pass


# ── 上下文 ───────────────────────────────────────────────

class PipelineContext:
    """流水线共享上下文：输入路径、缓存管理器、运行时状态"""

    def __init__(self, video_path: str, dst_srt_path: str, output_dir: str, cache_root: str):
        self.video_path = video_path
        self.dst_srt_path = dst_srt_path
        self.output_dir = output_dir

        # 可选：原声字幕（用于 Qwen 强制对齐)
        self.raw_src_path = ""

        # 缓存
        self.cache = CacheManager(video_path, cache_root)
        self.keep_temp = False
        self.work_dir = ""  # 由 Orchestrator 在 run() 中设置

        # 运行时路径（由各 step 填充)
        self.audio_path = ""            # 原始混合音频
        self.vocals_path = ""           # 人声轨
        self.bg_path = ""               # 背景轨
        self.sample_rate: Optional[int] = None
        self.subs = []              # List[SubtitleItem] 目标字幕 (start_ms/end_ms=SRT原始, calib_*=校准, gender=性别)
        self.raw_subs = []           # List[SubtitleItem] 原声字幕（同上）
        self.progress_cb = None          # 进度回调 (step_idx, pct, text)
        self.on_subs_ready = None        # 回调(subs) — 字幕处理完成后调用
        self.on_raw_subs_ready = None    # 回调(raw_subs) — 原声字幕就绪
        self.on_mix_done = None          # 回调(path) — 混音完成
        self.tts_segments = []          # [(start_ms, end_ms, mixed_clip_path), ...]
        self.final_audio_path = ""      # 最终音频

        # ── 统一从 cfg 加载所有配置字段 ──
        _skip = {'__dict__', '__doc__', '__module__', '__init__'}
        self.__dict__.update({
            k: v for k, v in cfg.__dict__.items()
            if not k.startswith('_') and k not in _skip
        })

        # 取消标记
        self.cancelled = False

        # 日志回调（由 UI 设置)
        self.log_ui = None
        self.log_file = None  # 仅写入日志文件,不显示到 UI

        # TTS 错误回调（由 UI 设置,用于逐条询问重试/跳过/终止)
        self.on_tts_error_cb = None

    def check_cancelled(self):
        if self.cancelled:
            raise CancelledError()


# ── 步骤基类 ─────────────────────────────────────────────


class CacheStatus(Enum):
    NONE = auto()      # 无缓存
    PARTIAL = auto()   # 部分命中
    FULL = auto()      # 全部命中


class BaseStep:
    """步骤基类"""

    name: str = ""
    """步骤显示名"""

    step_index: int = 0
    """步骤序号（0-based)"""

    cache_key: str = ""
    """CacheManager 中对应的缓存 key（空串表示不记录完成标记)"""

    dependencies: list[str] = []
    """前置步骤 cache_key 列表"""

    @property
    def target(self) -> str:
        """本步骤产出的目标描述"""
        return self.name

    def get_target_files(self, ctx: PipelineContext) -> list[str]:
        """返回本步骤期望产出的目标文件路径列表（子类重写)"""
        return []

    def check_cache(self, ctx: PipelineContext) -> CacheStatus:
        """检查缓存状态：FULL=全部命中, PARTIAL=部分命中, NONE=无缓存"""
        targets = self.get_target_files(ctx)
        if not targets:
            # 无目标文件定义 → 依赖 is_step_completed
            return CacheStatus.FULL if ctx.cache.is_step_completed(self.cache_key) else CacheStatus.NONE

        hit = sum(1 for p in targets if os.path.exists(p))
        if hit == len(targets):
            return CacheStatus.FULL
        elif hit > 0:
            return CacheStatus.PARTIAL
        else:
            return CacheStatus.NONE

    def mark_completed(self, ctx: PipelineContext):
        if self.cache_key:
            ctx.cache.mark_completed(self.cache_key)


    def run(self, ctx: PipelineContext):
        """执行本步骤（由子类实现)"""
        raise NotImplementedError

    def _progress(self, ctx: PipelineContext, pct: int, text: str = ""):
        ctx.progress_cb(self.step_index, pct, text or self.name)


# ── Step 1: 提取原始混合音频 ────────────────────────────

class ExtractAudioStep(BaseStep):
    name = "提取音频"
    step_index = 0
    cache_key = "extract"
    dependencies = []

    def get_target_files(self, ctx: PipelineContext) -> list[str]:
        return [ctx.cache.get_path(Step.EXTRACT, "mix_orig.wav")]

    def run(self, ctx: PipelineContext):
        status = self.check_cache(ctx)
        if status == CacheStatus.FULL:
            ctx.audio_path = ctx.cache.get_path(Step.EXTRACT, "mix_orig.wav")
            if ctx.sample_rate is None:
                ctx.sample_rate = ctx.cache.get_meta("sample_rate")
            if ctx.sample_rate is None:
                ctx.sample_rate = get_audio_info(ctx.video_path).sample_rate
                ctx.cache.set_meta("sample_rate", ctx.sample_rate)
            self._progress(ctx, 100)
            return

        self._progress(ctx, 10, "提取音频 (0/1)")
        # 自动探测原始音频采样率（上限 48kHz)
        if ctx.sample_rate is None:
            ctx.sample_rate = get_audio_info(ctx.video_path).sample_rate
        output = ctx.cache.get_path(Step.EXTRACT, "mix_orig.wav")
        cmd = [
            'ffmpeg', '-y', '-i', ctx.video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(ctx.sample_rate),
            '-ac', '2', output
        ]
        self._run_ffmpeg(cmd, ctx, "提取音频", "(0/1)")
        ctx.audio_path = output
        ctx.cache.set_meta("sample_rate", ctx.sample_rate)
        self.mark_completed(ctx)
        self._progress(ctx, 100)

    def _run_ffmpeg(self, cmd: list, ctx: PipelineContext, label: str, count_suffix: str = ""):
        suffix = f" {count_suffix}" if count_suffix else ""
        self._progress(ctx, 10, f"正在{label}...{suffix}")
        try:
            sp.run(cmd, check=True, capture_output=True, timeout=600)
        except sp.CalledProcessError as e:
            raise RuntimeError(f"⛔{label} 失败: {e.stderr.decode('utf-8', errors='replace')[:200]}")


# ── Step 2: Demucs 人声分离 ─────────────────────────────

class SeparateVocalsStep(BaseStep):
    name = "分离人声"
    step_index = 1
    cache_key = "demucs"
    dependencies = ["extract"]

    def get_target_files(self, ctx: PipelineContext) -> list[str]:
        return [
            ctx.cache.get_path(Step.DEMUCS, "vocals_orig.wav"),
            ctx.cache.get_path(Step.DEMUCS, "background.wav"),
        ]

    def run(self, ctx: PipelineContext):
        status = self.check_cache(ctx)
        if status == CacheStatus.FULL:
            ctx.vocals_path = ctx.cache.get_path(Step.DEMUCS, "vocals_orig.wav")
            ctx.bg_path = ctx.cache.get_path(Step.DEMUCS, "background.wav")
            self._progress(ctx, 100)
            return

        self._progress(ctx, 0, "分离人声 (0/2)")
        demucs_out = ctx.cache.get_step_dir("demucs")

        def _on_progress(p: int):
            self._progress(ctx, p, "分离人声 (0/2)")

        try:
            if not ctx.audio_path or not os.path.exists(ctx.audio_path):
                # 尝试从缓存重建（缓存标记存在但文件被删除的情况)
                _recovered = ctx.cache.get_path(Step.EXTRACT, "mix_orig.wav")
                if os.path.exists(_recovered):
                    ctx.audio_path = _recovered
                else:
                    raise RuntimeError(f"音频文件不存在: {ctx.audio_path or '空'},请重新运行第一步")
            vp, nvp = separate_vocals(
                ctx.audio_path, demucs_out,
                model=ctx.demucs_model,
                device=ctx.demucs_device,
                threads=ctx.demucs_threads,
                segment=ctx.demucs_segment,
                overlap=ctx.demucs_overlap,
                progress_callback=_on_progress,
            )
        except Exception as e:
            raise RuntimeError(f"⛔Demucs 分离失败: {e}")

        # 获取缓存目标路径
        ctx.check_cancelled()
        out_vp = ctx.cache.get_path(Step.DEMUCS, "vocals_orig.wav")
        out_nvp = ctx.cache.get_path(Step.DEMUCS, "background.wav")

        # 转人声为单声道并移到缓存
        os.makedirs(os.path.dirname(out_vp), exist_ok=True)
        self._progress(ctx, 50, "分离人声 (1/2)")
        sp.run(
            ['ffmpeg', '-y', '-i', vp, '-ac', '1', '-acodec', 'pcm_s16le', out_vp],
            check=True, capture_output=True
        )
        # 复制背景音到缓存
        shutil.copy2(nvp, out_nvp)
        self._progress(ctx, 75, "分离人声 (2/2)")
        # 日志输出时长对比
        try:
            _in = _sf.SoundFile(ctx.audio_path)
            _out = _sf.SoundFile(out_vp)
            ctx.log_ui(f"  Demucs: 输入 {_in.frames/_in.samplerate:.0f}s → 人声 {_out.frames/_out.samplerate:.0f}s ({(abs(_in.frames-_out.frames)/_in.frames*100) if _in.frames else 0:.1f}%)")
            _in.close()
            _out.close()
        except Exception:
            pass

        # 移动 Demucs 输出到缓存目录
        demucs_subdir = os.path.join(demucs_out, ctx.demucs_model)
        if os.path.exists(demucs_subdir):
            shutil.rmtree(demucs_subdir, ignore_errors=True)

        # 释放 Demucs 占用的显存
        cleanup_cuda()

        ctx.vocals_path = out_vp
        ctx.bg_path = out_nvp
        self.mark_completed(ctx)
        self._progress(ctx, 100)


# ── Step 3: 字幕处理（对齐 + 性别标记）───────────────────

class SubStep(BaseStep):
    name = "字幕处理"
    step_index = 2
    cache_key = "subs"
    dependencies = ["extract", "demucs"]

    def get_target_files(self, ctx: PipelineContext) -> list[str]:
        return [
            ctx.cache.get_path(Step.SUBS, "genders_cache.json"),
            ctx.cache.get_path(Step.SUBS, "calib.json"),
            ctx.cache.get_path(Step.SUBS, "asr.srt"),
        ]

    def run(self, ctx: PipelineContext):
        """执行字幕处理 — 缓存检查→对齐→性别检测→构建 subs"""
        # 1. 加载字幕
        subs = self._normalize_subs(ctx, None)
        total = len(subs)
        if total == 0:
            raise RuntimeError("字幕为空")

        # 2. 进度
        self._progress(ctx, 0, f"字幕处理 (0/{total})")

        # 3.0 原声 ASR 字幕 (独立检查, 仅文本不改时间)
        _asr_exists, _asr_path, _asr_rel = ctx.cache.file_info(Step.SUBS, "asr.srt")
        if _asr_exists:
            from .srt_parser import parse_srt
            ctx.raw_subs = parse_srt(_asr_path)
            ctx.log_ui(f"  从缓存加载 ASR 原声字幕: {_asr_rel}")
        else:
            asr_subs = self._generate_asr_srt(ctx, subs)
            if asr_subs:
                from .srt_parser import write_srt
                write_srt(asr_subs, _asr_path)
                ctx.raw_subs = asr_subs
                ctx.log_ui(f"  ASR 原声字幕已生成: {_asr_rel}")

        _has_calib, _, _ = ctx.cache.file_info(Step.SUBS, "calib.json")
        if _has_calib:
            ctx.log_ui("  检测到校准缓存,跳过对齐,直接从缓存恢复")
            ctx.cache.load_calib_cache(subs, ctx.raw_subs)
        else:
            if self._do_alignment(ctx, subs):
                ctx.subs = subs
                self._save_calib_cache(ctx)

        # 保存校准后的 ASR 字幕 (asr_calib.srt, 供调试对比)
        if ctx.raw_subs:
            _calib_entries = [
                (s.calib_start_ms or s.start_ms,
                 s.calib_end_ms or s.end_ms,
                 s.text)
                for s in ctx.raw_subs
            ]
            from .srt_parser import write_srt as _ws
            _ws(_calib_entries, ctx.cache.get_path(Step.SUBS, "asr_calib.srt"))

        # 4. 性别检测（独立缓存检查)
        _has_gender, _, _ = ctx.cache.file_info(Step.SUBS, "genders_cache.json")
        if _has_gender:
            ctx.cache.load_gender_cache(subs)
            ctx.log_ui("  从缓存加载性别标记")
        else:
            genders = self._detect_genders(ctx, subs)
            ctx.cache.save_gender_cache(genders)
            ctx.log_ui(f"  性别检测完成: {len(genders)} 条")

        # 5. 最终写入
        ctx.subs = subs
        if ctx.on_subs_ready:
            ctx.on_subs_ready(subs)
        if ctx.on_raw_subs_ready and ctx.raw_subs:
            ctx.on_raw_subs_ready(ctx.raw_subs)

        # 打印 VAD 偏移日志
        _vad_items = [s for s in subs if s.calib_vad_ms >= 0]
        if _vad_items:
            _parts = [f"#{s.idx}={s.calib_vad_ms}" for s in _vad_items]
            _vad_lines = []
            for i in range(0, len(_parts), 8):
                _prefix = "  VAD偏移(ms): " if i == 0 else "               "
                _vad_lines.append(_prefix + " ".join(_parts[i:i+8]))
            for _line in _vad_lines:
                if ctx.log_file:
                    ctx.log_file(_line)

        # 6. 完成
        self.mark_completed(ctx)
        self._progress(ctx, 100)

    def _save_calib_cache(self, ctx: PipelineContext):
        """保存校准字幕到 calib.json（subs 和 raw_subs 共用一份校准时间）"""
        calib_list = [s.to_cache_dict() for s in ctx.subs if s.is_calibrated]
        if calib_list:
            with open(ctx.cache.get_path(Step.SUBS, "calib.json"), 'w', encoding='utf-8') as f:
                json.dump(calib_list, f, indent=2, ensure_ascii=False)

    def _restore_from_cache(self, ctx: PipelineContext, subs):
        """从缓存恢复校准和性别到 subs（原地修改)"""
        if not subs:
            subs = parse_srt(ctx.dst_srt_path)
        ctx.cache.load_gender_cache(subs)
        # 先恢复 ASR 原声字幕文本
        _asr_exists, _asr_path, _ = ctx.cache.file_info(Step.SUBS, "asr.srt")
        if _asr_exists:
            ctx.raw_subs = parse_srt(_asr_path)
        # 再加载校准（同时写入 subs 和 raw_subs）
        ctx.cache.load_calib_cache(subs, ctx.raw_subs)
        ctx.subs = subs

    # ── 路径 A: 原声对齐 ────────────────────────────────

    def _run_qwen_align(self, ctx: PipelineContext,
                              dst_subs) -> bool:
        """Qwen 强制对齐, 返回 has_calib_changes"""

        if not ctx.raw_subs:
            ctx.raw_subs = parse_srt(ctx.raw_src_path)

        total = min(len(dst_subs), len(ctx.raw_subs))
        if total == 0:
            raise RuntimeError("目标字幕或原声字幕为空")

        # 1. 加载 Qwen 模型
        self._progress(ctx, 0, "原声对齐...")
        ctx.log_ui("  Qwen 开始强制对齐...")
        # 设置 Qwen API 日志回调（写入文件+UI)
        from .aligner_qwen import set_log_cb as _qwen_set_log
        _qwen_set_log(ctx.log_ui)
        try:
            from .aligner_qwen import _load_model as _qwen_load, set_prefer_uv as _qwen_set_uv
            _qwen_set_uv()
            _qwen_load()
        except Exception as e:
            ctx.log_ui(f"  Qwen 模型加载失败 ({e}),跳过对齐,直接使用 SRT 时间")
            corrected_times = {s.idx: (s.start_ms, s.end_ms) for s in dst_subs}
            return False

        BATCH_SIZE = ctx.qwen_batch_size or 8
        ctx.log_ui(f"  Qwen 模型已就绪,开始对齐 {total} 条字幕(每批 {BATCH_SIZE} 条)")

        corrected_times = {}
        changed_count = 0

        # 预计算所有送检区间 (一次性, 避免每条重复计算)
        _send_ranges = SubStep._compute_send_ranges(ctx.raw_subs, ctx)

        # 统一提取送检音频片段（复用 _extract_audio_clips)
        _clip_paths = SubStep._extract_audio_clips(ctx.raw_subs, _send_ranges, ctx.vocals_path, ctx.work_dir)

        # Phase 1: 准备 Qwen 对齐参数
        def _prepare_one(i):
            ctx.check_cancelled()
            src_sub = ctx.raw_subs[i]
            txt = src_sub.text.replace('\n', ' ').strip()
            orig_s, orig_e = src_sub.start_ms, src_sub.end_ms
            if not txt:
                return i, None, orig_s, orig_e, orig_s, orig_e, "空文本"

            win_s, win_e, _left_pad, _right_pad = _send_ranges[i]
            seg_path = _clip_paths.get(src_sub.idx)
            if not seg_path or not os.path.exists(seg_path):
                return i, None, orig_s, orig_e, win_s, win_e, "无音频片段"

            return i, (seg_path, txt), orig_s, orig_e, win_s, win_e, None

        _qwen_threads = get_threads(ctx, 'qwen_aligner_threads')
        with ThreadPoolExecutor(max_workers=_qwen_threads) as ex:
            futs = {ex.submit(_prepare_one, i): i for i in range(total)}
            prepared = [None] * total
            for f in as_completed(futs):
                i, item, orig_s, orig_e, win_s, win_e, err = f.result()
                prepared[i] = (item, orig_s, orig_e, win_s, win_e, err)

        # Phase 2: 批量 Qwen 对齐（占用显存,完后立即释放)
        _t0 = time.time()
        try:
            from .aligner_qwen import align_batch as _qwen_batch
            def _fmt(d):
                return f"{d:+5d}"
            self._diff_h = lambda o,c: _fmt(o - c)  # c<o(更早)→正, c>o(更晚)→负
            self._diff_t = lambda o,c: _fmt(c - o)  # c>o(更晚)→正, c<o(更早)→负

            # 按时长排序分桶,减少 padding 浪费
            _valid = []  # [(duration_ms, orig_index), ...]
            for j in range(total):
                item = prepared[j][0]
                orig_s, orig_e = prepared[j][1], prepared[j][2]
                if item is not None:
                    _valid.append((orig_e - orig_s, j))
            _valid.sort(key=lambda x: x[0])  # 按时长升序,相近时长的进同一批
            _sorted_indices = [idx for _, idx in _valid]

            _log_msgs = []  # 收集日志,最后排序打印
            _aligned = set()  # 成功校准的索引集合（reason=="OK"）
            _qwen_vad_cache = {}  # idx -> VAD起始偏移
            for batch_pos in range(0, len(_sorted_indices), BATCH_SIZE):
                ctx.check_cancelled()
                batch_indices = _sorted_indices[batch_pos:batch_pos + BATCH_SIZE]
                batch_items = [prepared[j][0] for j in batch_indices]

                if batch_items:
                    all_words = _qwen_batch(batch_items, language="ja")
                    for k, words in zip(batch_indices, all_words):
                        item = prepared[k][0]
                        orig_s, orig_e, win_s = prepared[k][1], prepared[k][2], prepared[k][3]
                        win_e = prepared[k][4]
                        _k_idx = dst_subs[k].idx
                        _qwen_fs = 0
                        le = 0
                        # 以下四个差值仅在 words 非空且校准成功时赋值,
                        # 预先置 0 避免 NameError 及跨迭代残留导致日志错位
                        _qwen_head_diff = _qwen_tail_diff = 0
                        _qvad_hd = _qvad_td = 0
                        if words:
                            fs = int(words[0]["start_ms"])
                            le = int(words[-1]["end_ms"])
                            _le_raw = le  # 保存 VAD 纠正前的原始值
                            _qwen_fs = fs
                            _qwen_head_diff = win_s + fs - orig_s  # Qwen 起始偏离原始字幕
                            cs = win_s + fs            # Qwen 检测到的绝对起始
                            ce = win_s + le            # Qwen 检测到的绝对结束
                            _pad_ms = ctx.asr_pad_ms        # Qwen 结果安全区
                            cs = max(win_s, cs - _pad_ms)  # 头部裁剪少裁/扩展多扩

                            # VAD 后处理：Qwen 可能把尾部无声也吞了,用 VAD 纠正
                            if item and item[0] and os.path.exists(item[0]):
                                _vad_info = vad_detect_speech(item[0], silence_thresh="-45dB")
                                if _vad_info and _vad_info.get('segments'):
                                    _vad_end = int(_vad_info['segments'][-1][1])
                                    if _vad_end < le - _pad_ms:
                                        le = _vad_end + _pad_ms * 2
                                        ce = win_s + le

                            # Qwen 尾部（VAD 修正后）偏离原始字幕
                            _qwen_tail_diff = win_s + le - orig_e

                            ce = min(win_e, ce + _pad_ms)      # 尾部仅允许扩展
                            ce = max(ce, orig_e)                 # 不低于原始结束（禁止裁剪)
                            cs = min(cs, orig_s)                 # 起始不晚于原字幕（禁止头部裁剪)
                            dur_ms = ce - cs
                            if dur_ms > 50:
                                rate = len(item[1]) * 1000 / dur_ms
                                if rate < 15:
                                    _final_s = min(cs, win_e - 50)
                                    _final_e = max(ce, orig_e)
                                    _qvad_hd = (win_s + _qwen_fs) - _final_s
                                    _qvad_td = (win_s + le) - _final_e
                                    corrected_times[_k_idx] = (_final_s, _final_e)
                                    _qwen_vad_cache[_k_idx] = _qvad_hd
                                    changed_count += (_final_s != orig_s or _final_e != orig_e)
                                    _aligned.add(_k_idx)
                                    reason = "OK"
                                else:
                                    corrected_times[_k_idx] = (orig_s, orig_e)
                                    reason = f"语速过快({rate:.1f}字/s)"
                            else:
                                corrected_times[_k_idx] = (orig_s, orig_e)
                                reason = "无对齐结果"
                        else:
                            corrected_times[_k_idx] = (orig_s, orig_e)
                            reason = "无对齐结果"

                        s, e = corrected_times[_k_idx]
                        # 只当 Qwen 有调整时才打印
                        if _qwen_fs > 20 or le > 20:
                            _hd = self._diff_h(orig_s, s)
                            _td = self._diff_t(orig_e, e)
                            _log_msgs.append((_k_idx, _hd, _td, _qwen_head_diff, _qwen_tail_diff, _qvad_hd, _qvad_td, reason))

                for j in batch_indices:
                    _j_idx = dst_subs[j].idx
                    if _j_idx not in corrected_times:
                        orig_s, orig_e = prepared[j][1], prepared[j][2]
                        win_s_fb = prepared[j][3] if len(prepared[j]) > 3 else orig_s
                        win_e_fb = prepared[j][4] if len(prepared[j]) > 4 else orig_e
                        err = prepared[j][5] if len(prepared[j]) > 5 else ''
                        corrected_times[_j_idx] = (win_s_fb, win_e_fb)
                        if err:
                            ctx.log_ui(f"  第{_j_idx:>4d}条: 跳过 ({err})")

                done = batch_pos + len(batch_indices)
                if done % 10 == 0 or done == len(_sorted_indices):
                    self._progress(ctx, int(done * 40 / total), f"强制对齐 ({done}/{total})")

            # 处理跳过的条目（item 为 None 的)
            for j in range(total):
                _j_idx = dst_subs[j].idx
                if _j_idx not in corrected_times:
                    orig_s, orig_e = prepared[j][1], prepared[j][2]
                    win_s_fb = prepared[j][3] if len(prepared[j]) > 3 else orig_s
                    win_e_fb = prepared[j][4] if len(prepared[j]) > 4 else orig_e
                    err = prepared[j][5] if len(prepared[j]) > 5 else ''
                    corrected_times[_j_idx] = (win_s_fb, win_e_fb)
                    if err:
                        ctx.log_ui(f"  第{_j_idx:>4d}条: 跳过 ({err})")

            # 按索引排序打印 Qwen 校准日志
            for _k, _hd, _td, _qwen_hd, _qwen_td, _qvad_hd, _qvad_td, _reason in sorted(_log_msgs, key=lambda x: x[0]):
                def _fmt(d):
                    return f"{d:+5d}"
                _qwen_info = f"qwen=[{_fmt(_qwen_hd)}→{_fmt(_qwen_td)}]"
                _qvad_info = f"qvad=[{_fmt(_qvad_hd)}→{_fmt(_qvad_td)}]"
                if _reason == "OK":
                    ctx.log_file(f"  第{_k:>4d}条(ms): head={_hd} tail={_td} {_qwen_info} {_qvad_info}")
                else:
                    ctx.log_ui(f"  第{_k:>4d}条(ms): head={_hd} tail={_td} {_qwen_info} {_qvad_info} ({_reason})")

            ctx.log_ui(f"  Qwen 对齐完成: {total} 条中 {len(_aligned)} 条成功校准, {changed_count} 条有变动, 耗时{time.time()-_t0:.1f}s")
        finally:
            # Qwen 对齐完毕,立即释放显存,后续性别检测不占用
            from .aligner_qwen import unload_model as _qwen_unload
            _qwen_unload()
            ctx.log_ui("  Qwen 模型已卸载,显存已释放")

        # 清理临时片段文件
        import glob as _glob
        for _p in _glob.glob(os.path.join(ctx.work_dir, "_align_clip_*.wav")):
            try: os.remove(_p)
            except: pass

        # 将校准时间写入目标字幕和原声字幕（原地修改)
        # 仅对成功校准的条目（_aligned, reason=="OK"）写入 calib_ 字段;
        # API 失败/无对齐结果回退原时间时保持 0（未校准),避免误判已校准。
        for sub in dst_subs:
            s, e = corrected_times.get(sub.idx, (sub.start_ms, sub.end_ms))
            sub.calib_start_ms = s
            sub.calib_end_ms = e
        for sub in ctx.raw_subs:
            s, e = corrected_times.get(sub.idx, (sub.start_ms, sub.end_ms))
            sub.calib_start_ms = s
            sub.calib_end_ms = e

        # 从 Qwen 词级时间戳提取 VAD 起始偏移 = 第一个词起始 - 校准窗口起始
        for _sub in dst_subs:
            _vad = _qwen_vad_cache.get(_sub.idx, -1)
            if _vad >= 0:
                _sub.calib_vad_ms = _vad

        has_calib_changes = bool(_aligned)
        return has_calib_changes

    # ── 路径 C: Whisper 转写对齐 (无原声字幕时可用) ────────

    def _run_whisper_align(self, ctx: PipelineContext, subs) -> bool:
        """用 faster-whisper 转写 → align_subs 拟合 (segment/word 兜底封装在 aligner 内部)"""
        total = len(subs)
        if total == 0:
            raise RuntimeError("目标字幕为空")

        from . import aligner_whisper as _wa
        _wa.set_log_cb(ctx.log_ui or print)

        self._progress(ctx, 0, "Whisper 转写...")
        _t0 = time.time()
        try:
            try:
                words, segments = _wa.transcribe_full(
                    ctx.vocals_path, language="ja",
                    vad_filter=ctx.whisper_vad_filter,
                    vad_threshold=ctx.whisper_vad_threshold,
                    vad_min_silence_ms=ctx.whisper_vad_min_silence_ms,
                    vad_speech_pad_ms=ctx.whisper_vad_speech_pad_ms,
                    beam_size=ctx.whisper_beam_size,
                )
            except Exception as e:
                ctx.log_ui(f"  ⛔Whisper 转写失败 ({e}), 回退直接使用 SRT 时间")
                return False
            if not words:
                ctx.log_ui("  ⚠️Whisper 无转写结果, 回退直接使用 SRT 时间")
                return False

            # 保存原始 ASR
            try:
                from .srt_parser import write_srt as _rws
                _rws([(s["start_ms"], s["end_ms"], s["text"]) for s in segments],
                     os.path.join(ctx.cache.cache_dir, "whisper.ja.srt"))
            except Exception:
                pass

            # 对齐 (segment/word 拟合 + 共用 calibrate_times)
            _send_ranges = SubStep._compute_send_ranges(subs, ctx)
            corrected_times, seg_texts, has_changes = _wa.align_subs(
                subs, _send_ranges, ctx, words, segments)

            # 用 Whisper 词级文本更新 raw_subs.text
            try:
                import bisect as _b
                _ws_starts = [w["start_ms"] for w in words]
                for sub in ctx.raw_subs:
                    win_s, win_e, _, _ = _send_ranges[sub.idx - 1]
                    _lo = _b.bisect_left(_ws_starts, win_s)
                    _hi = _b.bisect_right(_ws_starts, win_e)
                    _ww = words[_lo:_hi]
                    if _ww:
                        sub.text = "".join(w["word"] for w in _ww)
            except Exception:
                pass


            # 校准时间写回目标字幕
            for sub in subs:
                cs, ce = corrected_times.get(sub.idx - 1, (sub.start_ms, sub.end_ms))
                sub.calib_start_ms, sub.calib_end_ms = cs, ce

            # 从 Whisper 词级时间戳提取 VAD 起始偏移
            import bisect as _b
            _ws_starts = [w["start_ms"] for w in words]
            for _sub in subs:
                if _sub.is_calibrated:
                    i = _sub.idx - 1
                    win_s, _, _, _ = _send_ranges[i]
                    _final_s = _sub.calib_start_ms
                    _lo = _b.bisect_left(_ws_starts, win_s)
                    _hi = _b.bisect_right(_ws_starts, _sub.calib_end_ms)
                    _ww = words[_lo:_hi]
                    if _ww:
                        _first_ms = _ww[0]["start_ms"]
                        _vad = _first_ms - _final_s
                        if _vad >= 0:
                            _sub.calib_vad_ms = _vad

            # 构建 raw_subs
            ctx.raw_subs = []
            for sub in subs:
                i = sub.idx - 1
                cs, ce = corrected_times.get(i, (sub.start_ms, sub.end_ms))
                ctx.raw_subs.append(SubtitleItem(
                    idx=sub.idx, start_ms=sub.start_ms, end_ms=sub.end_ms,
                    text=seg_texts.get(i, "").replace('\n', ' ').strip(),
                    calib_start_ms=cs, calib_end_ms=ce,
                ))

            ctx.log_ui(f"  Whisper 对齐完成: {total} 条, 耗时 {time.time()-_t0:.1f}s")
            return True
        finally:
            _wa.unload_model()

    # ── ASR 原声字幕生成 ────────────────────────────

    def _generate_asr_srt(self, ctx: PipelineContext, subs) -> list:
        """用 SenseVoice ASR 对目标字幕区间生成原文字幕文本"""
        if not ctx.vocals_path or not os.path.exists(ctx.vocals_path):
            ctx.log_ui("  无人声文件,跳过 ASR 字幕生成")
            return []
        import time as _t; _t0 = _t.time()
        try:
            from .aligner_sensevoice import _get_models
            _asr_model, _ = _get_models()
        except Exception as e:
            ctx.log_ui(f"  SenseVoice ASR 加载失败: {e}")
            return []
        _ranges = SubStep._compute_send_ranges(subs, ctx)
        _clips = SubStep._extract_audio_clips(subs, _ranges, ctx.vocals_path, ctx.work_dir)
        if not _clips:
            ctx.log_ui("  音频片段提取失败,跳过 ASR")
            return []
        import re
        entries = []
        _skip_pattern = re.compile(r'^[(\[（【]')
        for i, sub in enumerate(subs):
            txt = sub.text or ""
            # 跳过背景解说（括号内的注释)
            if _skip_pattern.match(txt.strip()):
                entries.append(SubtitleItem(sub.idx, sub.start_ms, sub.end_ms, ""))
                continue
            clip = _clips.get(sub.idx)
            if not clip or not os.path.exists(clip):
                entries.append(SubtitleItem(sub.idx, sub.start_ms, sub.end_ms, ""))
                continue
            try:
                res = _asr_model.generate(input=clip, language="auto", ban_emo_unk=True, use_itn=False)
                txt = re.sub(r"<\|[^|]+\|>", "", str(res[0].get("text",""))).strip() if res else ""
                entries.append(SubtitleItem(sub.idx, sub.start_ms, sub.end_ms, txt))
            except Exception:
                entries.append(SubtitleItem(sub.idx, sub.start_ms, sub.end_ms, ""))
        ctx.log_ui(f"  SenseVoice ASR: {len([e for e in entries if e.text])}/{len(entries)} 条, "
                   f"耗时 {_t.time()-_t0:.1f}s")
        return entries

    def _do_alignment(self, ctx: PipelineContext, subs) -> bool:
        """字幕校准对齐 — Qwen / Whisper / 无对齐, 返回 has_calib_changes"""
        _align_mode = getattr(ctx, 'align_mode', 'qwen')

        if _align_mode == 'whisper':
            _ret = self._run_whisper_align(ctx, subs)
            from .aligner_whisper import _load_time, _model_name, _model_device
            if _load_time > 0:
                ctx.log_ui(f"  faster-whisper-large-v3 {_model_device} float16（load {_load_time:.1f}s）")
            return _ret

        # Qwen 强制对齐 (已通过 ASR 步骤生成 ctx.raw_subs)
        if ctx.raw_subs:
            ctx.log_ui(f"  Qwen 强制对齐 (ASR 原声, {len(ctx.raw_subs)} 条)")
            return self._run_qwen_align(ctx, subs)

        ctx.log_ui("  未提供原声字幕,直接使用 SRT 时间")
        return False

    @staticmethod
    def _normalize_subs(ctx: PipelineContext, subs) -> list:
        """加载并复位字幕的校准/性别字段"""
        if not subs:
            subs = parse_srt(ctx.dst_srt_path)
        for s in subs:
            s.calib_start_ms = 0
            s.calib_end_ms = 0
            s.gender = ''
        return subs

    @staticmethod
    def _compute_send_ranges(subs: list, ctx: PipelineContext):
        """一次性预计算所有字幕的送检音频时间区间

        500ms max pad / 200ms safe_gap (与 Qwen _prepare_one / Whisper 拟合窗口一致):
        - 两字幕间隔 ≤200ms 不扩展
        - 否则 (gap-200)/2, 上限 500ms, 两边平分
        - 首条 prev=0, 末条无下一条 → right_pad 取满额 _max_pad

        Returns:
            [(win_s, win_e, left_pad, right_pad), ...]  按 subs 顺序
            left_pad/right_pad 供后续校准后 pad 约束复用, 避免重复计算
        """
        _safe_gap = ctx.asr_safe_gap
        _max_pad = ctx.asr_max_pad
        total = len(subs)

        def _calc_pad(gap_ms):
            if gap_ms <= _safe_gap:
                return 0
            return min((gap_ms - _safe_gap) // 2, _max_pad)

        ranges = []
        for i, sub in enumerate(subs):
            orig_s, orig_e = sub.start_ms, sub.end_ms
            _prev_e = subs[i - 1].end_ms if i > 0 else 0
            _left_pad = _calc_pad(orig_s - _prev_e)
            if i + 1 < total:
                _right_pad = _calc_pad(subs[i + 1].start_ms - orig_e)
            else:
                _right_pad = _max_pad  # 末条无下一条, 满额扩展
            win_s = max(0, orig_s - _left_pad)
            win_e = orig_e + _right_pad
            ranges.append((win_s, win_e, _left_pad, _right_pad))
        return ranges

    @staticmethod
    def _extract_audio_clips(subs: list, send_ranges: list, vocals_path: str, work_dir: str) -> dict:
        """统一提取送检音频片段（多线程)"""
        from .audio_tools import split_audio_np
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _extract_one(i, sub):
            win_s, win_e, _, _ = send_ranges[i]
            clip_path = os.path.join(work_dir, f"_align_clip_{sub.idx:04d}.wav")
            try:
                split_audio_np(vocals_path, win_s, win_e, clip_path)
                if os.path.exists(clip_path):
                    return sub.idx, clip_path
            except Exception:
                pass
            return sub.idx, None

        clips = {}
        with ThreadPoolExecutor(max_workers=min(len(subs), 8)) as ex:
            futs = [ex.submit(_extract_one, i, sub) for i, sub in enumerate(subs)]
            for f in as_completed(futs):
                idx, path = f.result()
                if path:
                    clips[idx] = path
        return clips

    def _detect_genders(self, ctx: PipelineContext, subs) -> dict:
        """性别检测"""
        genders = {}
        total = len(subs)
        BATCH_SIZE = ctx.gender_batch_size or 6
        _mode = getattr(ctx, 'gender_detect_mode', 'wavlm')

        if _mode == "gender_cls":
            ctx.log_ui(f"  性别检测: {total} 条(voice-gender-classifier, {resolve_device()})")
            from .gender_cls import VoiceGenderClassifier as _new_cls
            try:
                _new_model = _new_cls.from_pretrained()
                _new_model = _new_model.to(resolve_device())
            except Exception as e:
                ctx.log_ui(f"  ⛔模型加载失败: {e},跳过性别检测")
                return genders
            clips = [None] * total
            def _extract_one_gc(i):
                ctx.check_cancelled()
                sub = subs[i]
                clip = ctx.cache.ref_path(sub)
                try:
                    if not os.path.exists(clip):
                        split_audio_np(ctx.vocals_path, sub.eff_start_ms, sub.eff_end_ms, clip)
                    return i, clip
                except Exception:
                    return i, None
            _extract_threads = get_threads(ctx, 'qwen_aligner_threads')
            with ThreadPoolExecutor(max_workers=_extract_threads) as ex:
                futs = {ex.submit(_extract_one_gc, i): i for i in range(total)}
                for f in as_completed(futs):
                    ctx.check_cancelled()
                    i, clip = f.result()
                    clips[i] = clip
            _t0 = time.time()
            try:
                for i, clip in enumerate(clips):
                    ctx.check_cancelled()
                    if clip and os.path.exists(clip):
                        try:
                            gender, conf = _new_model.predict(clip)
                            subs[i].gender = gender
                            genders[str(subs[i].idx)] = gender
                            ctx.log_file(f"  ✅ 第{i+1}条 性别={gender} (置信度={conf:.1%})")
                        except Exception as e:
                            ctx.log_file(f"  ⚠️ 第{i+1}条 检测失败: {e}")
                    else:
                        ctx.log_file(f"  ⚠️ 第{i+1}条 音频片段不可用")
                    self._progress(ctx, 40 + min(int((i+1) * 60 / total), 59), f"性别检测 ({i+1}/{total})")
                ctx.log_ui(f"  推理完成 ({time.time()-_t0:.1f}s)")
            finally:
                _new_model = None
                ctx.log_ui("  模型已卸载")
            return genders

        ctx.log_ui(f"  性别检测: {total} 条(WavLM 声纹模型, 每批 {BATCH_SIZE} 条, {resolve_device()})")

        # 在主线程预加载模型
        try:
            from .gender_wavlm import detect_genders_batch as _wavlm_batch, _load_model as _wavlm_load, unload_model as _wavlm_unload
            _wavlm_load()
        except Exception as e:
            ctx.log_ui(f"  ⛔WavLM 模型加载失败: {e},跳过性别检测")
            return genders

        # Phase 1: 并行提取音频片段
        clips = [None] * total
        _skip_info = {}
        _skip_lock = __import__('threading').Lock()
        def _extract_one(i):
            ctx.check_cancelled()
            sub = subs[i]
            clip = ctx.cache.ref_path(sub)
            try:
                if not os.path.exists(clip):
                    split_audio_np(ctx.vocals_path, sub.eff_start_ms, sub.eff_end_ms, clip)
                # 静音/低能量片段跳过性别检测
                if is_low_energy(clip):
                    _db = get_rms_db(clip)
                    with _skip_lock:
                        _skip_info[i] = _db
                    return i, None
                return i, clip
            except Exception:
                return i, None

        _extract_threads = get_threads(ctx, 'qwen_aligner_threads')
        with ThreadPoolExecutor(max_workers=_extract_threads) as ex:
            futs = {ex.submit(_extract_one, i): i for i in range(total)}
            for f in as_completed(futs):
                ctx.check_cancelled()
                i, clip = f.result()
                clips[i] = clip

        # Phase 2: 批量模型推理
        _t0 = time.time()
        try:
            _wavlm_pass = []
            _wavlm_fail = []
            _wavlm_skip = []  # 静音跳过
            for batch_start in range(0, total, BATCH_SIZE):
                ctx.check_cancelled()
                batch = clips[batch_start:batch_start + BATCH_SIZE]
                batch_results = _wavlm_batch(batch)
                done = min(batch_start + BATCH_SIZE, total)
                for j, result in enumerate(batch_results):
                    idx = batch_start + j
                    g = result.get("gender", "")
                    conf = result.get("confidence", 0)
                    _raw_g = g
                    if g and conf < 0.75:
                        g = ""
                    if g:
                        subs[idx].gender = g
                        genders[str(idx + 1)] = g
                    if g:
                        _wavlm_pass.append((idx + 1, g, conf))
                    elif _raw_g:
                        _wavlm_fail.append((f"  ⚠️ 第{idx+1}条 WavLM检测={_raw_g} (置信度={conf:.1%}),低于阈值已置空",))
                    else:
                        err = result.get('error', '')
                        if err == "音频无效":
                            _wavlm_skip.append(idx + 1)  # 静音跳过,末尾汇总
                        elif err:
                            _wavlm_fail.append((f"  ⚠️ 第{idx+1}条 WavLM性别=未定 (置信度={conf:.1%}) {err}",))
                self._progress(ctx, 40 + min(int(done * 60 / total), 59), f"性别检测 ({done}/{total})")
            # 合并打印: 每行 8 条
            _batch = [f"{'♀️' if g=='female' else '♂️'}#{i:02d} ({100*conf:5.1f}%)" for i, g, conf in _wavlm_pass]
            for i in range(0, len(_batch), 8):
                ctx.log_file(" ".join(_batch[i:i+8]))
            if _wavlm_pass:
                ctx.log_file(f"  ✅ok {len(_wavlm_pass)}")
            for _msg in _wavlm_fail:
                ctx.log_file(_msg[0])
            ctx.log_ui(f"  WavLM 推理完成 ({time.time()-_t0:.1f}s)")
            if _wavlm_fail:
                ctx.log_ui(f"  ⚠️ {len(_wavlm_fail)} 条低于置信度阈值")
            if _wavlm_skip:
                _skip_lines = []
                for i in range(0, len(_wavlm_skip), 8):
                    _chunk = _wavlm_skip[i:i+8]
                    _parts = []
                    for s in _chunk:
                        _db = _skip_info.get(s - 1, None)
                        if _db is not None and _db != float('-inf'):
                            _parts.append(f"{s:>4d}({_db:.0f}dB)")
                        else:
                            _parts.append(f"{s:>4d}(-)")
                    _skip_lines.append("  ".join(_parts))
                ctx.log_ui(f"  🔇 {len(_wavlm_skip)} 条静音跳过:")
                for _line in _skip_lines:
                    ctx.log_ui(f"     {_line}")
        finally:
            _wavlm_unload()
            ctx.log_ui("  WavLM 模型已卸载,显存已释放")
        return genders



# ── Step 4: TTS API 合成（纯调用,不做音频处理)────────────

class TTSSynthesisStep(BaseStep):
    """仅调用 TTS API 生成原始音频,不做任何修剪/对齐/混音

    TTS 文件命名: tts_{idx}_{s_HH}_{s_MM}_{s_SS}_{s_MS}-{e_HH}_{e_MM}_{e_SS}_{e_MS}.wav
    例: tts_0001_00_00_12_340-00_00_16_780.wav
    带时间范围确保字幕校准后缓存不会误用。
    """
    name = "TTS合成"
    step_index = 3
    cache_key = "tts"
    dependencies = ["extract", "demucs", "subs"]

    def get_target_files(self, ctx: PipelineContext) -> list[str]:
        return [
            ctx.cache.tts_path(sub)
            for sub in ctx.subs
        ]

    def run(self, ctx: PipelineContext, on_item=None):
        """执行 TTS 合成

        Args:
            on_item: 可选,每条 TTS 完成后的回调 on_item(idx, tts_path)
        """
        status = self.check_cache(ctx)
        if status == CacheStatus.FULL:
            if ctx.tts_similarity_threshold > 0:
                self._progress(ctx, 99, "声纹校验...")
            else:
                self.mark_completed(ctx)
                self._progress(ctx, 100)
                return

        # CPU 时本地引擎不可用,强制切换为 API 模式
        if ctx.use_local_tts:
            from core.utils import resolve_device
            if resolve_device("auto") == "cpu":
                ctx.log_ui("  ⚠️ CPU 模式下本地 IndexTTS2 引擎不可用,自动切换为 API 模式")
                ctx.use_local_tts = False
                ctx.log_ui(f"  使用 API: {ctx.tts_api_url} ({ctx.tts_mode})")

        need_tts = status != CacheStatus.FULL
        total = len(ctx.subs)

        if need_tts:
            self._progress(ctx, 0, f"TTS (0/{total})")

            # 预收集需要合成的任务
            tasks = []
            for sub in ctx.subs:
                if os.path.exists(ctx.cache.tts_path(sub)):
                    continue
                txt = sub.text.strip()
                if (txt.startswith('(') and txt.endswith(')')) or \
                   (txt.startswith('（') and txt.endswith(')')):
                    if on_item:
                        on_item(sub.idx, "")
                    continue
                if not sub.gender:
                    if ctx.log_ui:
                        ctx.log_ui(f"  ⏭️ 第{sub.idx:4d}条 性别未定,跳过 TTS")
                    if on_item:
                        on_item(sub.idx, "")
                    continue
                tasks.append(sub)

            if tasks:
                # ── 多端口自动探测与分配 ──
                max_workers = get_threads(ctx, 'tts_threads', max_workers=4)
                base_url = ctx.tts_api_url.rstrip('/')
                _parsed = urlparse(base_url)
                _base_no_port = f"{_parsed.scheme}://{_parsed.hostname}"
                _base_path = _parsed.path
                _base_port = _parsed.port or 9000
                available_ports = []
                for _port in range(_base_port, _base_port + max_workers):
                    _sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                    _sock.settimeout(0.05)
                    _result = _sock.connect_ex((_parsed.hostname or '127.0.0.1', _port))
                    _sock.close()
                    if _result == 0:
                        available_ports.append(_port)
                if not available_ports:
                    available_ports = [_base_port]
                _port_urls = [f"{_base_no_port}:{p}{_base_path}" for p in available_ports]
                _port_idx = 0
                tasks_with_port = []
                for t in tasks:
                    tasks_with_port.append((t, _port_urls[_port_idx % len(_port_urls)]))
                    _port_idx += 1

                done_count = total - len(tasks)

                _tts_ui_counter = [0]  # 每 10 条才显示到窗口

                def _do_tts(sub, api_url):
                    nonlocal _tts_ui_counter
                    try:
                        tts_cfg = _ctx_to_tts_config(ctx)
                        tts_cfg.api_url = api_url  # 可能被端口分配覆盖
                        # 从原声字幕取文本作为 prompt_text
                        if tts_cfg.use_fixed_ref:
                            tts_cfg.prompt_text = ctx.fixed_ref_text_female if sub.gender == "female" else ctx.fixed_ref_text_male
                        elif hasattr(ctx, 'raw_subs') and ctx.raw_subs:
                            for _rs in ctx.raw_subs:
                                if _rs.idx == sub.idx:
                                    tts_cfg.prompt_text = _rs.text or ""
                                    break
                        tts_path = synthesize_tts_segment(
                            sub,
                            cache=ctx.cache, work_dir=ctx.work_dir,
                            tts_cfg=tts_cfg,
                            log_cb=ctx.log_ui, check_cancelled=ctx.check_cancelled,
                        )
                        if tts_path:
                            if on_item:
                                on_item(sub.idx, tts_path)
                            _tts_ui_counter[0] += 1
                            if _tts_ui_counter[0] % 10 == 0 and ctx.log_ui:
                                ctx.log_ui(f"  TTS 完成 {_tts_ui_counter[0]}/{total}")
                            return ("ok", sub.idx, tts_path, sub.text)
                        return ("error", sub.idx, "TTS 返回空数据", sub.text)
                    except Exception as e:
                        return ("error", sub.idx, str(e), sub.text)


                failed_items = []
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = set()
                    remaining_tasks = list(tasks_with_port)
                    for _ in range(max_workers):
                        if not remaining_tasks:
                            break
                        ctx.check_cancelled()
                        t = remaining_tasks.pop(0)
                        futs.add(ex.submit(_do_tts, *t))

                    while futs:
                        done, futs = wait(futs, return_when=FIRST_COMPLETED)
                        for f in done:
                            try:
                                status, idx, info, txt = f.result()
                            except CancelledError:
                                raise
                            except Exception as _e:
                                ctx.log_ui(f"  ⚠️ TTS 线程异常: {_e}")
                                continue
                            if status == "error":
                                log_msg = info[:200]
                                ctx.log_ui(f"  ⚠️ 第{idx:4d}条 TTS 失败: {log_msg}")
                                failed_items.append((idx, info, txt))
                            else:
                                done_count += 1
                                self._progress(ctx, int(done_count * 100 / total), f"TTS ({done_count}/{total})")
                        # 每批次处理完后,如有失败项则询问用户
                        if failed_items and ctx.on_tts_error_cb:
                            # 阻断后续：取消未完成的任务,清空等待队列
                            for ff in futs:
                                ff.cancel()
                            futs.clear()
                            remaining_tasks.clear()
                            ctx.on_tts_error_cb(failed_items)
                            failed_items = []
                        if remaining_tasks and not ctx.cancelled:
                            ctx.check_cancelled()
                            t = remaining_tasks.pop(0)
                            futs.add(ex.submit(_do_tts, *t))

                # 兼容：没有 on_tts_error_cb 时的兜底
                if failed_items:
                    if ctx.on_tts_error_cb:
                        ctx.on_tts_error_cb(failed_items)
                        ctx.check_cancelled()  # abort 选择后在此中断
                    else:
                        raise RuntimeError(f"⛔TTS 合成失败: {failed_items[0][1]}")
                # 打印最终条数（即使不够 10 条)
                _final_cnt = _tts_ui_counter[0]
                if _final_cnt > 0 and _final_cnt % 10 != 0 and ctx.log_ui:
                    ctx.log_ui(f"  TTS 完成 {_final_cnt}/{total}")
            else:
                self.mark_completed(ctx)

        # ── Phase 2: 批量声纹校验（缓存结果按 TTS 文件 mtime+size 失效)──
        # 固定提示音模式跳过：已用固定提示音合成，无需再对比
        if ctx.tts_similarity_threshold > 0 and ctx.subs and not ctx.use_fixed_ref:
            _low_idx = []  # 低于阈值的 idx
            _sim_scores = {}  # idx -> similarity
            _cache_path = ctx.cache.get_path(Step.TTS, ".similarity_cache.json")
            _sim_cache = {}
            if os.path.exists(_cache_path):
                try:
                    with open(_cache_path) as _f:
                        _sim_cache = json.load(_f)
                except Exception:
                    _sim_cache = {}
            _sim_cache_dirty = False
            # 收集日志,先输出满足的、再输出不满足的
            _pass_logs = []  # (msg, file_only)  file_only=True 只写文件
            _fail_logs = []  # (msg,)  ⚠️ 相似度不足
            _error_logs = []  # (msg,)  ❌ 无法检测
            _need_sim = []  # 非缓存条目,稍后并行推理
            for sub in ctx.subs:
                ctx.check_cancelled()
                idx = sub.idx
                tts_path = ctx.cache.tts_path(sub)
                if not os.path.exists(tts_path):
                    continue
                # 检查相似度缓存（按 TTS 文件 mtime+size 生效)
                _score = None
                try:
                    if os.path.exists(_cache_path):
                        _st = os.stat(tts_path)
                        _entry = _sim_cache.get(str(idx))
                        if _entry and _entry.get("tts_mtime") == _st.st_mtime and _entry.get("tts_size") == _st.st_size:
                            _score = _entry.get("score")
                except Exception:
                    pass
                if _score is not None:
                    _sim_scores[idx] = _score
                    if _score < ctx.tts_similarity_threshold:
                        # 缓存分数低于阈值,但如果是对固定提示音的比对且 >=0.6 则跳过重试
                        if _entry.get("ref") == "fixed" and _score >= 0.6:
                            _pass_logs.append((idx, _score, "☑️"))
                            continue
                        _low_idx.append(idx)
                        _g = sub.gender
                        _fail_logs.append((f"  ⚠️ 第{idx:4d}条 相似度(缓存)不足: {_score:.3f} {_g}",))
                    else:
                        _pass_logs.append((idx, _score, "✅"))
                    continue
                # 非缓存条目：收集后并行推理
                _need_sim.append((idx, sub, tts_path))
            # 全部命中缓存 → 跳过设备加载与推理,仅输出缓存日志
            if not _need_sim:
                _hit = len(_sim_scores)
                _low = len(_low_idx)
                ctx.log_ui(f"  声纹校验缓存全命中: {_hit} 条,跳过推理" +
                          (f" (其中 {_low} 条相似度不足)" if _low else ""))
                _batch = [f"{t}#{i:02d}({s:.3f})" for i, s, t in _pass_logs]
                for i in range(0, len(_batch), 8):
                    ctx.log_file(" ".join(_batch[i:i+8]))
                if _pass_logs:
                    ctx.log_file(f"  ✅ok {len(_pass_logs)}")
                for _msg in _fail_logs:
                    ctx.log_file(_msg[0] if isinstance(_msg, tuple) else _msg)
                if _sim_cache_dirty:
                    try:
                        with open(_cache_path, 'w') as _f:
                            json.dump(_sim_cache, _f, indent=2)
                    except Exception:
                        pass
            else:
                self._progress(ctx, 99, "声纹校验...")
                ctx.log_ui(f"  声纹校验设备: {resolve_device()} (缓存命中 {len(_sim_scores)}/{len(_sim_scores)+len(_need_sim)})")
                # 并行执行声纹校验
                from concurrent.futures import ThreadPoolExecutor as _TPE
                def _run_sim(idx, sub, tts_path):
                    nonlocal _sim_cache_dirty
                    _ref = ctx.cache.ref_path(sub)
                    if not os.path.exists(_ref):
                        try:
                            split_audio_np(ctx.vocals_path, sub.eff_start_ms, sub.eff_end_ms, _ref)
                        except Exception:
                            return (idx, None, f"  ⚠️ 第{idx:4d}条 无法提取参考音频,跳过")
                    _sim = compare_similarity(tts_path, _ref)
                    if _sim["error"]:
                        return (idx, None, f"  ❌ 第{idx:4d}条 声纹检测失败: {_sim['error']}")
                    if _sim["similarity"] is not None:
                        try:
                            _st = os.stat(tts_path)
                            _sim_cache[str(idx)] = {"ref": "orig", "score": _sim["similarity"], "device": _sim["device"],
                                                    "tts_mtime": _st.st_mtime, "tts_size": _st.st_size}
                            _sim_cache_dirty = True
                        except Exception:
                            pass
                        if _sim["similarity"] < ctx.tts_similarity_threshold:
                            _g = sub.gender
                            return (idx, _sim["similarity"], f"  ⚠️ 第{idx:4d}条 相似度不足: {_sim['similarity']:.3f} {_g}")
                        else:
                            return (idx, _sim["similarity"], f"  ✅ 第{idx:4d}条 相似度满足: {_sim['similarity']:.3f}")
                    return (idx, None, None)
                with _TPE(max_workers=4) as _ex:
                    _futs = {_ex.submit(_run_sim, idx, sub, tts_path): idx for idx, sub, tts_path in _need_sim}
                    _sim_results = []
                    for _f in as_completed(_futs):
                        _sim_results.append(_f.result())
                    # 按 idx 排序后统一处理,保证日志顺序
                    _sim_results.sort(key=lambda r: r[0])
                    for idx, score, log_msg in _sim_results:
                        if log_msg is None:
                            continue
                        if score is not None:
                            _sim_scores[idx] = score
                            if score < ctx.tts_similarity_threshold:
                                _low_idx.append(idx)
                                _fail_logs.append((log_msg,))
                            else:
                                _pass_logs.append((idx, score, "✅"))
                        else:
                            _error_logs.append((log_msg,))
                # 写入 Phase 2 的相似度缓存
                if _sim_cache_dirty:
                    try:
                        with open(_cache_path, 'w') as _f:
                            json.dump(_sim_cache, _f, indent=2)
                        _sim_cache_dirty = False
                    except Exception:
                        pass
                # 集中写日志文件: 每行 8 条
                _batch = [f"{t}#{i:02d}({s:.3f})" for i, s, t in _pass_logs]
                for i in range(0, len(_batch), 8):
                    ctx.log_file(" ".join(_batch[i:i+8]))
                if _pass_logs:
                    ctx.log_ui(f"  ✅ 共 {len(_pass_logs)} 条相似度满足")
                # ❌ 错误同步写到文件（不显示到 UI)
                for _msg in _error_logs:
                    ctx.log_file(_msg[0])
                if _error_logs:
                    ctx.log_ui(f"  ❌ 共 {len(_error_logs)} 条声纹检测失败")
                for _msg in _fail_logs:
                    ctx.log_file(_msg[0])
                if _fail_logs:
                    ctx.log_ui(f"  🔄 共 {len(_fail_logs)} 条相似度不足,与固定提示音对比...")

                # ── Phase 3: 批量对比固定提示音 + 批量重试 ──
                if _low_idx:
                    # Phase 3a: 批量用固定提示音对比,收集结果
                    _to_retry = []
                    _skip_logs = []  # ✅ 跳过重试
                    _failed_logs = []  # 🚫 无法重试
                    _prep_retry_logs = []  # ⚠️ 准备重试
                    for idx in _low_idx:
                        ctx.check_cancelled()
                        sub = next((s for s in ctx.subs if s.idx == idx), None)
                        if sub is None:
                            continue
                        _retry_ref = ""
                        if sub.gender == "female":
                            _retry_ref = ctx.fixed_ref_audio_female or ""
                        elif sub.gender == "male":
                            _retry_ref = ctx.fixed_ref_audio_male or ""
                        else:
                            _failed_logs.append((f"  🚫 第{idx:4d}条 相似度不足,但性别为空,跳过",))
                            continue
                        if not _retry_ref or not os.path.exists(_retry_ref):
                            _g = sub.gender
                            _failed_logs.append((f"  🚫 第{idx:4d}条 相似度不足,但无 {_g} 提示音",))
                            continue
                        tts_path = ctx.cache.tts_path(sub)
                        if os.path.exists(tts_path):
                            _sim_ref = compare_similarity(tts_path, _retry_ref)
                            _ref_score = _sim_ref["similarity"] if _sim_ref["similarity"] is not None else 0
                            if _ref_score >= 0.6:
                                _skip_logs.append((f"  ✅ 第{idx:4d}条 跳过重试,固定提示音相似度 {_ref_score:.3f}",))
                                continue
                            _prep_retry_logs.append((f"  ⚠️ 第{idx:4d}条 固定提示音相似度 {_ref_score:.3f}<0.6,准备重试",))
                        _to_retry.append((idx, sub, _retry_ref))
                    # 逐条仅写文件,统计显示到界面
                    for _msg in _skip_logs:
                        ctx.log_file(_msg[0])
                    for _msg in _failed_logs:
                        ctx.log_file(_msg[0])
                    if _skip_logs + _failed_logs + _prep_retry_logs:
                        _part = []
                        if _skip_logs:
                            _part.append(f"✅跳过{len(_skip_logs)}")
                        if _failed_logs:
                            _part.append(f"🚫无法{len(_failed_logs)}")
                        if _prep_retry_logs:
                            _part.append(f"⚠️重试{len(_prep_retry_logs)}")
                        ctx.log_ui(f"  📊 固定提示音对比: {', '.join(_part)}")
                    for _msg in _prep_retry_logs:
                        ctx.log_file(_msg[0])

                    # Phase 3b: 批量 TTS 重试（固定提示音合成)
                    if _to_retry:
                        ctx.log_ui(f"  🔄 {len(_to_retry)} 条固定提示音不匹配,开始重试合成...")
                        for idx, sub, _retry_ref in _to_retry:
                            ctx.check_cancelled()
                            _cache_path = ctx.cache.get_path(Step.TTS, ".similarity_cache.json")
                            tts_path = ctx.cache.tts_path(sub)
                            _old_path = tts_path + ".bak"
                            try:
                                shutil.copy2(tts_path, _old_path)
                                _retry_cfg = _ctx_to_tts_config(ctx)
                                _retry_cfg.use_fixed_ref = True
                                _retry_cfg.fixed_ref_audio_male = _retry_ref if sub.gender == "male" else ""
                                _retry_cfg.fixed_ref_audio_female = _retry_ref if sub.gender == "female" else ""
                                _new_path = synthesize_tts_segment(
                                    sub,
                                    cache=ctx.cache, work_dir=ctx.work_dir,
                                    tts_cfg=_retry_cfg,
                                    log_cb=ctx.log_ui,
                                )
                                if _new_path:
                                    _sim2 = compare_similarity(_new_path, _retry_ref)
                                    # 始终使用重试结果,不比较原分数
                                    if _sim2["similarity"] is not None and _sim2["similarity"] >= 0.6:
                                        ctx.log_ui(f"  ✅ 第{idx:4d}条 重试满足: {_sim2['similarity']:.3f}")
                                    elif _sim2["similarity"] is not None:
                                        ctx.log_ui(f"  📈 第{idx:4d}条 重试 {_sim2['similarity']:.3f}<0.6")
                                    else:
                                        ctx.log_ui(f"  📈 第{idx:4d}条 重试完成(无法检测相似度)")
                                    # 接受新结果,清除对应混音缓存
                                    _mixed = ctx.cache.mixed_path(sub)
                                    if os.path.exists(_mixed):
                                        os.remove(_mixed)
                                    os.remove(_old_path)
                                    # 更新缓存
                                    try:
                                        _st = os.stat(_new_path)
                                        _score = _sim2["similarity"] if _sim2["similarity"] is not None else 0
                                        _sim_cache[str(idx)] = {"ref": "fixed", "score": _score, "device": _sim2["device"],
                                                                  "tts_mtime": _st.st_mtime, "tts_size": _st.st_size}
                                        _sim_cache_dirty = True
                                    except Exception:
                                        pass
                                else:
                                    shutil.move(_old_path, tts_path)
                                    ctx.log_ui(f"  ⚠️ 第{idx:4d}条 重试失败,保留原结果")
                                    _sim_cache.pop(str(idx), None)
                                    _sim_cache_dirty = True
                            except Exception as _re:
                                if os.path.exists(_old_path):
                                    shutil.move(_old_path, tts_path)
                                ctx.log_ui(f"  ⚠️ 第{idx:4d}条 重试异常: {_re}")
                            _sim_cache.pop(str(idx), None)
                            _sim_cache_dirty = True
                            _sim_cache.pop(str(idx), None)
                            _sim_cache_dirty = True
                    # 一次性写入重试后的缓存
                    if _sim_cache_dirty:
                        try:
                            with open(_cache_path, 'w') as _f:
                                json.dump(_sim_cache, _f, indent=2)
                        except Exception:
                            pass

        self.mark_completed(ctx)
        self._progress(ctx, 100, "TTS 完成")


def _safe_edge(ctx: PipelineContext, idx: int, start_ms: int, end_ms: int,
                sub_map: dict = None) -> int:
    """计算安全的边界扩展值,避免与相邻字幕重叠

    检查前后字幕的间距,若间距 < edge_ms*2 则缩小扩展值。
    若 ctx.edge_ms == 0 则返回 0（不扩展)。
    """
    if ctx.edge_ms == 0:
        return 0
    edge = ctx.edge_ms
    if sub_map is None:
        sub_map = {s.idx: s for s in ctx.subs}
    prev = sub_map.get(idx - 1)
    if prev and prev.end_ms > 0:
        gap = start_ms - prev.end_ms
        if gap < edge * 2:
            edge = max(0, gap // 2)
    nxt = sub_map.get(idx + 1)
    if nxt and nxt.start_ms > 0:
        gap = nxt.start_ms - end_ms
        if gap < edge * 2:
            edge = min(edge, max(0, gap // 2))
    return edge


# ── 独立可复用的 TTS 合成与音频处理函数 ──────────────────────────────────────


@dataclass
class TTSConfig:
    """TTS 合成配置"""
    use_local_tts: bool = False             # 使用本地引擎
    tts_local_mode: str = "indextts"
    api_url: str = "http://localhost:9001"
    api_key: str = ""
    mode: str = "rainfall"
    model_name: str = ""
    language: str = "zh"
    timeout: int = 120
    extra_params: dict = field(default_factory=dict)
    speaker_embedding_path_male: str = ""
    speaker_embedding_path_female: str = ""
    send_prompt_text: bool = False
    use_fixed_ref: bool = False
    fixed_ref_audio_male: str = ""
    fixed_ref_audio_female: str = ""
    vocals_path: str = ""
    prompt_text: str = ""                   # 原声字幕文本

    @classmethod
    def from_ctx(cls, ctx) -> 'TTSConfig':
        """从 PipelineContext 提取 TTS 配置"""
        return cls(
            use_local_tts=ctx.use_local_tts,
            tts_local_mode=getattr(ctx, 'tts_local_mode', 'indextts'),
            api_url=ctx.tts_api_url, api_key=ctx.tts_api_key,
            mode=ctx.tts_mode, model_name=ctx.tts_model,
            language=ctx.tts_language, timeout=ctx.tts_timeout,
            extra_params=ctx.tts_extra_params,
            send_prompt_text=ctx.send_prompt_text,
            use_fixed_ref=ctx.use_fixed_ref,
            fixed_ref_audio_male=ctx.fixed_ref_audio_male,
            fixed_ref_audio_female=ctx.fixed_ref_audio_female,
            speaker_embedding_path_male=getattr(ctx, 'speaker_embedding_path_male', ''),
            speaker_embedding_path_female=getattr(ctx, 'speaker_embedding_path_female', ''),
            vocals_path=ctx.vocals_path,
            prompt_text=getattr(ctx, 'prompt_text', ''),
        )

    @classmethod
    def from_dict(cls, d: dict) -> 'TTSConfig':
        """从 settings dict 构建 TTSConfig"""
        _defaults = cls()
        return cls(
            use_local_tts=d.get("use_local_tts", _defaults.use_local_tts),
            tts_local_mode=d.get("tts_local_mode", _defaults.tts_local_mode),
            api_url=d.get("tts_api_url", _defaults.api_url),
            api_key=d.get("tts_api_key", _defaults.api_key),
            mode=d.get("tts_mode", _defaults.mode),
            model_name=d.get("tts_model", _defaults.model_name),
            language=d.get("tts_language", _defaults.language),
            timeout=d.get("tts_timeout", _defaults.timeout),
            extra_params=d.get("tts_extra_params", _defaults.extra_params),
            send_prompt_text=d.get("send_prompt_text", _defaults.send_prompt_text),
            use_fixed_ref=d.get("use_fixed_ref", _defaults.use_fixed_ref),
            fixed_ref_audio_male=d.get("fixed_ref_audio_male", _defaults.fixed_ref_audio_male),
            fixed_ref_audio_female=d.get("fixed_ref_audio_female", _defaults.fixed_ref_audio_female),
            speaker_embedding_path_male=d.get("speaker_embedding_path_male", _defaults.speaker_embedding_path_male),
            speaker_embedding_path_female=d.get("speaker_embedding_path_female", _defaults.speaker_embedding_path_female),
            vocals_path=d.get("vocals_path", _defaults.vocals_path),
            prompt_text=d.get("prompt_text", _defaults.prompt_text),
        )


# 保留兼容别名
_ctx_to_tts_config = TTSConfig.from_ctx
_settings_to_tts_config = TTSConfig.from_dict


def synthesize_tts_segment(
    sub,
    cache, work_dir: str,
    tts_cfg: TTSConfig,
    log_cb=None, check_cancelled=None,
) -> Optional[str]:
    """合成单条 TTS 并写入缓存目录"""
    idx = sub.idx
    start_ms = sub.eff_start_ms
    end_ms = sub.eff_end_ms
    text = sub.text
    gender = sub.gender

    def _log(msg):
        """整百条显示到窗口,其他不打印（文件日志由调用者在外部处理)"""
        if log_cb and idx % 100 == 0:
            log_cb(msg)

    if check_cancelled:
        check_cancelled()

    txt = text.replace('\n', ' ').strip()
    if (txt.startswith('(') and txt.endswith(')')) or \
       (txt.startswith('（') and txt.endswith(')')):
        return None  # 背景解说跳过（pipeline 在此处不输出日志)

    _log(f"  TTS {idx} 开始: (API:{tts_cfg.api_url.split(':')[-1]})")

    # 确定参考音频和情绪音频
    ref_audio = None
    emo_ref_audio = None
    if tts_cfg.use_fixed_ref:
        # 固定提示音模式：固定提示音作为音色参考,原视频人声作为情绪参考
        # 说话人嵌入模式：用原视频人声做参考(风格由嵌入提供)
        _emb_key = "speaker_embedding_path_male" if gender == "male" else "speaker_embedding_path_female"
        _has_emb = getattr(tts_cfg, _emb_key, '') and os.path.exists(getattr(tts_cfg, _emb_key, ''))
        if _has_emb:
            # 有训练好的嵌入时,固定提示音不是必须的,用原视频人声做参考
            # 嵌入在 _local_tts 前加载(见下方固定提示音 + .pt 嵌入代码)
            pass
        elif gender == "female" and tts_cfg.fixed_ref_audio_female:
            ref_audio = tts_cfg.fixed_ref_audio_female
        elif gender == "male" and tts_cfg.fixed_ref_audio_male:
            ref_audio = tts_cfg.fixed_ref_audio_male
        else:
            ref_audio = tts_cfg.fixed_ref_audio_female or tts_cfg.fixed_ref_audio_male
        # 固定提示音模式要求文件必须存在,缺失直接报错（不退化为用人声做参考)
        if not _has_emb and (not ref_audio or not os.path.exists(ref_audio)):
            raise RuntimeError(f"TTS 合成失败 (第{idx}条): 固定提示音模式未配置有效的提示音文件")
        # 原视频人声片段作为情绪参考（复用 _ref_ 缓存文件,与非固定模式同源)
        if tts_cfg.vocals_path:
            emo_ref_audio = cache.ref_path(sub)
            if not os.path.exists(emo_ref_audio):
                try:
                    split_audio_np(tts_cfg.vocals_path, start_ms, end_ms, emo_ref_audio)
                except Exception:
                    emo_ref_audio = None
            if emo_ref_audio and not os.path.exists(emo_ref_audio):
                emo_ref_audio = None

    if not ref_audio:
        ref_audio = cache.ref_path(sub)
        if not os.path.exists(ref_audio):
            try:
                split_audio_np(tts_cfg.vocals_path, start_ms, end_ms, ref_audio)
            except Exception:
                _log(f"  ⏭️ 第{idx:4d}条: 无法从人声提取参考音频,跳过")
                return None
        if not os.path.exists(ref_audio):
            _log(f"  ⏭️ 第{idx:4d}条: 参考音频文件不存在 ({ref_audio}),跳过")
            return None

    # TTS 合成
    tts_path = cache.tts_path(sub)

    if tts_cfg.use_local_tts:
        # 本地 TTS 引擎
        from .tts_indextts2 import tts_synthesize as _local_tts
        _local_kw = {}

        _target_ms = end_ms - start_ms + 500  # 多生成一点余量,VAD 修剪后不会截断
        _log(f"  TTS {idx} 开始: (本地引擎 indextts, 目标 {_target_ms}ms)")
        try:
            _kwargs = dict(
                text=txt,
                target_duration_ms=_target_ms,
                stretch_to_target=False,
                work_dir=work_dir,
                output_path=tts_path,
                **_local_kw,
            )
            # 固定提示音 + .pt 模式不传 ref_audio_path(引擎直接加载缓存)
            if tts_cfg.use_fixed_ref:
                _emb_key = "speaker_embedding_path_male" if sub.gender == "male" else "speaker_embedding_path_female"
                _emb_path = getattr(tts_cfg, _emb_key, '') or ''
                if _emb_path and os.path.exists(_emb_path):
                    _kwargs["_emb_path_hint"] = _emb_path
                else:
                    _kwargs["ref_audio_path"] = ref_audio
            else:
                _kwargs["ref_audio_path"] = ref_audio
            # 情绪参考(固定提示音和人声模式都需要)
            if emo_ref_audio and tts_cfg.tts_local_mode == "indextts":
                _kwargs["emo_audio_path"] = emo_ref_audio
            _audio_data = _local_tts(**_kwargs)
        except Exception as e:
            raise RuntimeError(f"TTS 合成失败 (第{idx}条): {e}")
        if not _audio_data:
            raise RuntimeError(f"TTS 返回空数据 (第{idx}条)")
    else:
        # TTS API 调用
        tts = TTSClient(
            api_url=tts_cfg.api_url, api_key=tts_cfg.api_key,
            mode=tts_cfg.mode, model_name=tts_cfg.model_name,
            language=tts_cfg.language, timeout=tts_cfg.timeout,
            extra_params=tts_cfg.extra_params or {},
        )
        kw = {"text": txt, "ref_audio_path": ref_audio, "item_idx": idx}
        if tts_cfg.send_prompt_text:
            kw["ref_text"] = txt
        try:
            _audio_data = tts.synthesize(**kw)
        except Exception as e:
            raise RuntimeError(f"TTS 合成失败 (第{idx}条): {e}")
        if not _audio_data or len(_audio_data) < 100:
            raise RuntimeError(f"TTS 返回空数据 (第{idx}条)")

        with open(tts_path, 'wb') as f:
            f.write(_audio_data)

    _log(f"  TTS {idx} 完成")
    return tts_path


def mix_tts_segment(
    sub,
    tts_path: str, vocals_path: str, bg_path: str,
    cache, work_dir: str,
    vad_mode: str = "原声对齐", vad_pad_ms: int = 50,
    edge_ms: int = 100, fade: bool = True,
    log_cb=None, file_log_cb=None,
) -> tuple:
    """对单条 TTS 进行 VAD 修剪 + 前导对齐 + 增益匹配 + 背景混音"""
    idx = sub.idx
    start_ms = sub.eff_start_ms
    end_ms = sub.eff_end_ms

    _log = make_logger(log_cb)
    _file_log = make_logger(file_log_cb)

    mixed_clip = cache.mixed_path(sub)

    speech_start = 0
    tts_path_faded = tts_path

    # 淡入淡出（写入 .faded 副本,不覆盖缓存)
    if fade:
        try:
            tts_dur = get_audio_info(tts_path).duration_ms
            tts_vad = vad_detect_speech(tts_path)
            speech_start = tts_vad.get('start_ms', 0) if isinstance(tts_vad, dict) else 0
            if speech_start > 10 and tts_dur > 50:
                y_data, sr_f = _sf.read(tts_path)
                fl_in = min(int(0.040 * sr_f), len(y_data) - int(speech_start * sr_f / 1000))
                if fl_in > 1:
                    fi = _np.linspace(0.0, 1.0, fl_in)
                    if y_data.ndim > 1:
                        fi = fi.reshape(-1, 1)
                    y_data[int(speech_start * sr_f / 1000):int(speech_start * sr_f / 1000) + fl_in] *= fi
                fl_out = min(int(0.100 * sr_f), len(y_data) // 2)
                if fl_out > 1:
                    fo = _np.exp(-_np.linspace(0, 5, fl_out))
                    if y_data.ndim > 1:
                        fo = fo.reshape(-1, 1)
                    y_data[-fl_out:] *= fo
                tts_path_faded = tts_path + ".faded.wav"
                _sf.write(tts_path_faded, y_data, sr_f)
            _file_log(f"  TTS {idx}: 淡入开始={speech_start}ms 淡出开始={tts_dur - 100}ms")
        except Exception:
            pass

    # 裁剪原声参考片段（供 VAD 修剪和前导对齐使用)
    vocals_ref = os.path.join(work_dir, f"_vocals_{idx:04d}.wav")
    try:
        split_audio_np(vocals_path, start_ms, end_ms, vocals_ref)
    except Exception:
        vocals_ref = ""

    # VAD 修剪
    aligned_path = os.path.join(work_dir, f"aligned_{idx:04d}.wav")
    try:
        _, trim_info = vad_trim_silence(
            tts_path_faded, aligned_path,
            ref_audio_path=vocals_ref or vocals_path,
            pre_speech_start_ms=speech_start,
            ref_speech_start_ms=getattr(sub, 'calib_vad_ms', -1))
    except Exception:
        trim_info = {}
        shutil.copy2(tts_path_faded, aligned_path)

    # 前导静音对齐
    tmp_sil = aligned_path + ".sil_tmp"
    try:
        if vocals_ref:
            _, _ = add_leading_silence(
                aligned_path, tmp_sil,
                ref_audio_path=vocals_ref,
                mode=vad_mode, margin_ms=vad_pad_ms)
            os.replace(tmp_sil, aligned_path)
    except Exception:
        pass

    # RMS 增益匹配
    gain_path = os.path.join(work_dir, f"gain_{idx:04d}.wav")
    try:
        if vocals_ref:
            match_rms_gain(aligned_path, vocals_ref, gain_path)
        else:
            shutil.copy2(aligned_path, gain_path)
    except Exception:
        shutil.copy2(aligned_path, gain_path)

    # 边界扩展 + 混音
    _edge_ms = edge_ms
    _gain_dur_ms = get_audio_info(gain_path).duration_ms
    if _edge_ms > 0:
        _padded_tts = os.path.join(work_dir, f"_padded_{idx:04d}.wav")
        pad_audio_np(gain_path, _padded_tts, front_ms=_edge_ms)
        _padded_dur_ms = get_audio_info(_padded_tts).duration_ms
        _exp_start = max(0, start_ms - _edge_ms)
        _exp_end = max(start_ms + _padded_dur_ms, end_ms + _edge_ms)
        _mix_tts = _padded_tts
    else:
        _exp_start = start_ms
        _exp_end = max(start_ms + _gain_dur_ms, end_ms)
        _mix_tts = gain_path
        _padded_tts = gain_path

    bg_ref = os.path.join(work_dir, f"_bg_{idx:04d}.wav")
    try:
        split_audio_np(bg_path, _exp_start, _exp_end, bg_ref)
        mix_segment_clip(_mix_tts, bg_ref, mixed_clip, edge_ms=_edge_ms)
    except Exception:
        shutil.copy2(_mix_tts, mixed_clip)

    # 清理临时文件
    try:
        os.remove(bg_ref)
    except Exception:
        pass
    if tts_path_faded != tts_path:
        try:
            os.remove(tts_path_faded)
        except Exception:
            pass

    return mixed_clip, trim_info, vocals_ref, aligned_path, gain_path, _padded_tts


# ── Step 5: 全长重建（混音+拼接)──────────────────────────

class AudioMixAndMergeStep(BaseStep):
    """对每条 TTS 混音 → 拼接为全长音频"""
    name = "全长重建"
    step_index = 4
    cache_key = Step.MIX
    dependencies = ["extract", "demucs", "subs", "tts"]

    def get_target_files(self, ctx: PipelineContext) -> list[str]:
        return [
            ctx.cache.mixed_path(sub)
            for sub in ctx.subs
        ]

    def run(self, ctx: PipelineContext, on_mix_item=None):
        status = self.check_cache(ctx)
        if status == CacheStatus.FULL:
            _build_tts_segments(ctx)
            self._progress(ctx, 100)
            return

        self._progress(ctx, 0, "处理音频...")

        import time as _mix_timer
        _mix_t0 = _mix_timer.time()
        ctx.log_ui("===== 分段混音阶段开始 =====")
        ctx.log_ui(f"  参数: vad_mode={ctx.vad_mode}  edge_ms={ctx.edge_ms}ms  "
                 f"crossfade_ms=40ms  vad_pad_ms={ctx.vad_pad_ms}ms(仅\"字幕对齐\"模式有效)")

        total = len(ctx.subs)
        if total > 0:
            self._progress(ctx, 0, f"混合 (0/{total})")
        _batch_ids = []
        # 构建 sub_map 一次,避免 _safe_edge 每次重建
        _sub_map = {s.idx: s for s in ctx.subs}

        # TTS + mix 日志缓冲区（并行线程导致输出乱序,统一收集后排序)
        _mix_log_buffer = []

        def _process_one(sub):
            """处理单条字幕（供线程池调用)"""
            idx = sub.idx
            start_ms = sub.eff_start_ms
            end_ms = sub.eff_end_ms

            # 检查单条混合缓存
            mixed_clip = ctx.cache.mixed_path(sub)
            if os.path.exists(mixed_clip):
                return None

            # 查找原始 TTS 文件
            tts_path = ctx.cache.tts_path(sub)
            if not os.path.exists(tts_path):
                if ctx.log_ui:
                    ctx.log_ui(f"  ⏭️ mix {idx}: TTS 文件不存在,跳过混音")
                return None

            # TTS 原始时长（淡出位置基于此计算)
            _tts_dur = get_audio_info(tts_path).duration_ms

            _edge_ms = _safe_edge(ctx, idx, start_ms, end_ms, sub_map=_sub_map)
            # file_log_cb=None 抑制 mix_tts_segment 内即时 TTS 日志（由主线程统一输出)
            mixed_clip, trim_info, vocals_ref, aligned_path, gain_path, _padded_tts = mix_tts_segment(
                sub=sub,
                tts_path=tts_path, vocals_path=ctx.vocals_path, bg_path=ctx.bg_path,
                cache=ctx.cache, work_dir=ctx.work_dir,
                vad_mode=ctx.vad_mode, vad_pad_ms=ctx.vad_pad_ms,
                edge_ms=_edge_ms, fade=True,
                log_cb=ctx.log_ui,
                file_log_cb=None,
            )
            # 注：展开区间由 _build_tts_segments 重建,mix 行在拼接后统一输出
            return idx, start_ms, end_ms, _tts_dur, _edge_ms, trim_info, vocals_ref, aligned_path, gain_path, _padded_tts

        # 多线程并行混音
        _mix_threads = get_threads(ctx, 'mix_threads')
        with ThreadPoolExecutor(max_workers=_mix_threads) as ex:
            futs = {ex.submit(_process_one, sub): sub for sub in ctx.subs}
            for i, f in enumerate(as_completed(futs)):
                ctx.check_cancelled()
                result = f.result()
                if result is not None:
                    idx, start_ms, end_ms, _tts_dur, _edge_ms, trim_info, vocals_ref, aligned_path, gain_path, _padded_tts = result
                    # ── raw/tts 日志（mix 行在拼接后统一收集,按 idx 排序输出)──
                    try:
                        _t = fmt_time

                        # raw: 字幕区间（id 在前,时长=字幕时长)
                        _raw_tail = f"时长{(end_ms - start_ms)/1000:.3f}s"
                        _raw_part1 = f"{idx:>3d} raw: 字幕开始= {_t(start_ms)}"
                        _raw_part2 = f"字幕结束= {_t(end_ms)}"
                        _mix_log_buffer.append((
                            idx, 1,
                            f"{_raw_part1:<42s}{_raw_part2:<28s}  {_raw_tail}"
                        ))

                        # tts: 淡入/淡出位置（无 id,时长=TTS 总时长)
                        _tts_raw = int(trim_info.get('speech_start_ms', 0)) if trim_info else 0
                        _tts_lead = int(trim_info.get('leading_trimmed_ms', 0)) if trim_info else 0
                        _tts_speech = max(0, _tts_raw - _tts_lead)  # VAD 修剪后的语音偏移
                        _fade_in_abs = start_ms + _tts_speech
                        _fade_out_abs = start_ms + _tts_dur - _tts_lead  # TTS 实际结束位置（VAD 修剪后)
                        _fade_out_offset = _fade_out_abs - end_ms  # 基于 raw 结束
                        _tts_tail = f"时长{(_fade_out_abs - _fade_in_abs)/1000:.3f}s"
                        _tts_part1 = f"    tts: 淡入开始= {_t(_fade_in_abs)}({_tts_speech:+5d}ms)"
                        _tts_part2 = f"淡出结束= {_t(_fade_out_abs)}({_fade_out_offset:+5d}ms)"
                        _mix_log_buffer.append((
                            idx, 2,
                            f"{_tts_part1:<42s}{_tts_part2:<28s}  {_tts_tail}"
                        ))

                        # 展开区间由 _build_tts_segments 重建,mix 行在拼接后统一输出
                        # 实际计算（被 seg_len//4 与相邻片段间距夹断),此处无法预知。
                    except Exception:
                        pass
                    # 清理临时文件
                    for tmp in [aligned_path, gain_path, _padded_tts, vocals_ref]:
                        if os.path.exists(tmp):
                            try:
                                os.remove(tmp)
                            except Exception:
                                pass
                    _batch_ids.append(idx)
                    if on_mix_item and len(_batch_ids) >= 5:
                        on_mix_item(_batch_ids)
                        _batch_ids = []
                self._progress(ctx, int((i + 1) * 100 / total), f"混合 ({i+1}/{total})")

        if on_mix_item and _batch_ids:
            on_mix_item(_batch_ids)

        ctx.log_ui(f"===== 分段混音阶段完成 ({_mix_timer.time()-_mix_t0:.1f}s) =====")
        _build_tts_segments(ctx)
        _sp_t0 = _mix_timer.time()

        # Phase 2: 拼接为全长音频
        self._progress(ctx, 50, "拼接音频...")
        final_audio = ctx.cache.get_path(Step.MIX, "final_audio.wav")
        _xfade_info = []
        splice_segments_into_base(
            ctx.audio_path, ctx.tts_segments, final_audio, crossfade_ms=40,
            crossfade_info=_xfade_info,
        )
        ctx.final_audio_path = final_audio
        if ctx.on_mix_done:
            ctx.on_mix_done(final_audio)
        ctx.log_ui(f"  splice 拼接: {_mix_timer.time()-_sp_t0:.1f}s")
        _n_seg = len(ctx.tts_segments)
        _dur = get_audio_info(final_audio).duration_ms / 1000
        ctx.log_ui(f"  拼接完成: {_n_seg} 段替换, 总长 {_dur:.1f}s")

        # 收集交叉淡变日志（按 idx 排序后与 raw/tts 统一输出)
        if _xfade_info:
            _sub_map = {s.idx: s for s in ctx.subs}
            _seg2idx = {}
            for sub in ctx.subs:
                _edge = _safe_edge(ctx, sub.idx, sub.eff_start_ms, sub.eff_end_ms, sub_map=_sub_map)
                _exp_start = max(0, sub.eff_start_ms - _edge)
                mixed = ctx.cache.mixed_path(sub)
                if os.path.exists(mixed):
                    _seg2idx[_exp_start] = sub.idx
            for _es, _ee, _lin, _lout in _xfade_info:
                _idx = _seg2idx.get(_es, 0)
                _sub = _sub_map.get(_idx)
                if _sub:
                    _mix_part1 = f"    mix: 交叉开始= {fmt_time(_es)}({_es - _sub.start_ms:+5d}ms)"
                    _mix_part2 = f"交叉结束= {fmt_time(_ee)}({_ee - _sub.end_ms:+5d}ms)"
                else:
                    _mix_part1 = f"    mix: 交叉开始= {fmt_time(_es)}"
                    _mix_part2 = f"交叉结束= {fmt_time(_ee)}"
                _mix_tail = f"时长{(_ee-_es)/1000:.3f}s"
                _mix_log_buffer.append((
                    _idx, 3,
                    f"{_mix_part1:<42s}{_mix_part2:<28s}  {_mix_tail}"
                ))

        # 统一输出所有日志（按 idx 升序,同 idx 内 raw→tts→mix)
        _mix_log_buffer.sort(key=lambda x: (x[0], x[1]))
        for _, _, _line in _mix_log_buffer:
            ctx.log_file(_line)

        ctx.log_ui(f"===== 分段混音阶段完成 ({_mix_timer.time()-_mix_t0:.1f}s) =====")
        self.mark_completed(ctx)
        self._progress(ctx, 100, "全长重建完成")



def _build_tts_segments(ctx: PipelineContext):
    """构建 tts_segments 供 splice 步骤使用（模块级,供多个 step 调用)"""
    ctx.tts_segments = []
    _sub_map = {s.idx: s for s in ctx.subs}
    for sub in ctx.subs:
        idx = sub.idx
        mixed = ctx.cache.mixed_path(sub)
        if os.path.exists(mixed):
            dur = get_audio_info(mixed).duration_ms
            exp_start = max(0, sub.eff_start_ms - _safe_edge(ctx, sub.idx, sub.eff_start_ms, sub.eff_end_ms, sub_map=_sub_map))
            seg_end = exp_start + dur if dur > 0 else sub.eff_end_ms
            ctx.tts_segments.append((exp_start, seg_end, mixed))


# ── Step 6: 合并回视频 ────────────────────────────────

class VideoMergeStep(BaseStep):
    """将最终音频合并回视频文件"""
    name = "合并视频"
    step_index = 5
    cache_key = ""

    def get_target_files(self, ctx: PipelineContext) -> list[str]:
        return []

    def check_cache(self, ctx: PipelineContext) -> 'CacheStatus':
        return CacheStatus.NONE  # 每次重新合并

    def run(self, ctx: PipelineContext):
        self._progress(ctx, 0, "合并视频...")
        _t0 = time.time()
        final_audio = ctx.cache.get_path(Step.MIX, "final_audio.wav")
        if not os.path.exists(final_audio):
            ctx.log_ui("  ⚠️ 最终音频不存在,跳过合并")
            return
        video_name = Path(ctx.video_path).stem
        output_video = os.path.join(ctx.output_dir, f"{video_name}.ts.mp4")
        # 存在则自动递增序号
        if os.path.exists(output_video):
            _n = 1
            while os.path.exists(os.path.join(ctx.output_dir, f"{video_name}.ts{_n}.mp4")):
                _n += 1
            output_video = os.path.join(ctx.output_dir, f"{video_name}.ts{_n}.mp4")
        ctx.log_ui(f"  ffmpeg 合并视频开始: {os.path.basename(output_video)}")
        cmd = [
            'ffmpeg', '-y', '-hide_banner',
            '-i', ctx.video_path,
            '-i', final_audio,
            '-c:v', 'copy',
            '-c:a', 'libmp3lame', '-b:a', '192k',
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest',
            output_video
        ]
        try:
            sp.run(cmd, check=True, capture_output=True, timeout=1200)
        except sp.CalledProcessError as e:
            raise RuntimeError(f"⛔视频合并失败: {e.stderr.decode('utf-8', errors='replace')[:200]}")
        ctx.log_ui(f"  ffmpeg mux: {time.time()-_t0:.1f}s")
        self._progress(ctx, 100, "视频合并完成")


# ── 流水线编排器 ─────────────────────────────────────────

class PipelineOrchestrator:
    """按顺序执行各步骤,处理缓存和依赖关系"""

    STEPS = [
        ExtractAudioStep(),
        SeparateVocalsStep(),
        SubStep(),
        TTSSynthesisStep(),
        AudioMixAndMergeStep(),
        VideoMergeStep(),
    ]

    def __init__(self, ctx: PipelineContext, tts_item_cb=None, mix_item_cb=None):
        self.ctx = ctx
        self.tts_item_cb = tts_item_cb
        self.mix_item_cb = mix_item_cb

    def run(self):
        """执行完整流水线"""
        if not self.ctx.work_dir:
            self.ctx.work_dir = os.path.join(
                tempfile.gettempdir(), f"dub_{self.ctx.cache.video_hash}"
            )
            os.makedirs(self.ctx.work_dir, exist_ok=True)

        # 预解析 Demucs 设备（避免每次循环重复 import torch)
        _demucs_dev = resolve_device(self.ctx.demucs_device)

        try:
            for step in self.STEPS:
                self.ctx.check_cancelled()
                self._check_deps(step)

                # 对于依赖 subs 的步骤,确保 context 已恢复
                if step.step_index >= 2 and not self.ctx.subs:
                    self._restore_calib_subs()

                _t = time.time()
                if self.ctx.log_ui:
                    self.ctx.log_ui("")  # 空行分隔（无时间戳)
                    _header = f"=== {step.step_index+1}.{step.name}"
                    # 大模型步骤标注设备
                    if step.step_index == 1:  # 分离人声 (Demucs)
                        _header += f" ({_demucs_dev})"
                    elif step.step_index == 2:  # 字幕处理 (Qwen/WavLM)
                        _header += f" ({resolve_device()} bf16)"
                    elif step.step_index == 3:  # TTS 合成
                        _header += f" ({resolve_device()} bf16)"
                    self.ctx.log_ui(f"{_header} ===")
                if step.step_index == 3:
                    step.run(self.ctx, on_item=self.tts_item_cb)
                elif step.step_index == 4:
                    step.run(self.ctx, on_mix_item=self.mix_item_cb)
                else:
                    step.run(self.ctx)
                if self.ctx.log_ui:
                    self.ctx.log_ui(f"  {step.step_index+1} finish, time use {time.time()-_t:.1f}s")
        except CancelledError:
            raise
        except Exception as e:
            # traceback.print_exc()
            raise RuntimeError(f"⛔[{step.step_index+1}.{step.name}] 失败: {e}")
        finally:
            # 清理临时工作目录
            # work_dir 在 cache_root/temp/ 下固定，由用户管理缓存时统一清理
            if self.ctx.work_dir and os.path.isdir(self.ctx.work_dir):
                if not self.ctx.keep_temp:
                    try:
                        for _f in os.listdir(self.ctx.work_dir):
                            _fp = os.path.join(self.ctx.work_dir, _f)
                            if os.path.isfile(_fp):
                                os.remove(_fp)
                            elif os.path.isdir(_fp):
                                shutil.rmtree(_fp, ignore_errors=True)
                    except Exception:
                        pass

    def run_single_step(self, step_index: int):
        """执行单个步骤（用于重试/断点续传)"""
        if step_index < 0 or step_index >= len(self.STEPS):
            raise ValueError(f"无效步骤索引: {step_index}")
        step = self.STEPS[step_index]
        self._check_deps(step)

        if not self.ctx.work_dir:
            self.ctx.work_dir = os.path.join(
                tempfile.gettempdir(), f"dub_{self.ctx.cache.video_hash}"
            )
            os.makedirs(self.ctx.work_dir, exist_ok=True)
        # 恢复音频路径（支持单步执行,step>=2 需要 vocals_path)
        if step.step_index >= 2:
            if not self.ctx.audio_path:
                self.ctx.audio_path = self.ctx.cache.get_path(Step.EXTRACT, "mix_orig.wav")
            if not self.ctx.vocals_path:
                self.ctx.vocals_path = self.ctx.cache.get_path(Step.DEMUCS, "vocals_orig.wav")
            if not self.ctx.bg_path:
                self.ctx.bg_path = self.ctx.cache.get_path(Step.DEMUCS, "background.wav")
        if not self.ctx.subs:
            self._restore_calib_subs()
        if step.step_index == 3:
            step.run(self.ctx, on_item=self.tts_item_cb)
        elif step.step_index == 4:
            step.run(self.ctx, on_mix_item=self.mix_item_cb)
        else:
            step.run(self.ctx)

    def _restore_calib_subs(self):
        """从缓存恢复 subs（支持单步执行)"""
        if not self.ctx.subs:
            _subs, _raw_subs, _has_calib = self.ctx.cache.restore_calib_subs(
                self.ctx.dst_srt_path, parse_srt, self.ctx.raw_src_path)
            self.ctx.subs = _subs or []
            self.ctx.raw_subs = _raw_subs or []

    def _check_deps(self, step: BaseStep):
        """检查前置步骤是否已完成（按目标文件或 cache_key)"""
        for dep_key in step.dependencies:
            # 找到依赖的步骤实例
            dep_step = None
            for s in self.STEPS:
                if s.cache_key == dep_key:
                    dep_step = s
                    break
            if dep_step:
                dep_status = dep_step.check_cache(self.ctx)
                if dep_status == CacheStatus.NONE:
                    raise RuntimeError(
                        f"前置步骤 [{dep_step.name}] 未完成,无法执行 [{step.name}]"
                    )
            else:
                # 无对应步骤的 key → 回退到 is_step_completed
                if not self.ctx.cache.is_step_completed(dep_key):
                    raise RuntimeError(
                        f"前置步骤 [{dep_key}] 未完成,无法执行 [{step.name}]"
                    )

    def get_first_incomplete(self) -> int:
        """返回第一个未完成的步骤索引"""
        for i, step in enumerate(self.STEPS):
            status = step.check_cache(self.ctx)
            if status != CacheStatus.FULL:
                return i
        return len(self.STEPS)

    @staticmethod
    def step_count() -> int:
        return len(PipelineOrchestrator.STEPS)


# ── 辅助操作：单条重生成 / 重新混音 ────────────────

def regen_single_tts(
    sub,
    settings: dict,
    cache, work_dir: str,
    log_cb=None, done_cb=None, edge_ms=100,
):
    """单条 TTS 重新生成（线程安全)"""
    idx = sub.idx
    text = sub.text
    gender = sub.gender
    start_ms = sub.eff_start_ms
    end_ms = sub.eff_end_ms
    _log = make_logger(log_cb)

    try:
        os.makedirs(work_dir, exist_ok=True)

        # ── 1. TTS 合成（复用 synthesize_tts_segment)──
        tts_cfg = _settings_to_tts_config(settings)
        # CPU 时本地引擎不可用,强制切换为 API 模式（与 TtsStep.run 一致)
        if tts_cfg.use_local_tts:
            from core.utils import resolve_device
            if resolve_device("auto") == "cpu":
                _log(f"  ⚠️ CPU 模式下本地引擎不可用,第 {idx} 条自动切换为 API 模式")
                tts_cfg.use_local_tts = False
        if not tts_cfg.vocals_path:
            tts_cfg.vocals_path = cache.get_path(Step.DEMUCS, "vocals_orig.wav")
        tts_path = synthesize_tts_segment(
            sub,
            cache=cache, work_dir=work_dir,
            tts_cfg=tts_cfg,
            log_cb=log_cb,
        )
        if not tts_path:
            if done_cb:
                done_cb(False, "")
            return
        # 声纹校验 + 重试
        _threshold = settings.get("tts_similarity_threshold", 0.0)
        _score = None
        if _threshold > 0:
            try:
                _ref = cache.ref_path(sub)
                _sim = compare_similarity(tts_path, _ref)
                _cache_path = cache.get_path(Step.TTS, ".similarity_cache.json")
                # 一次性加载缓存
                _sim_cache = {}
                if os.path.exists(_cache_path):
                    try:
                        with open(_cache_path) as _f:
                            _sim_cache = json.load(_f)
                    except Exception:
                        _sim_cache = {}
                _sim_cache_dirty = False
                if _sim["similarity"] is not None:
                    _score = _sim["similarity"]
                    try:
                        _st = os.stat(tts_path)
                        _sim_cache[str(idx)] = {"ref": "orig", "score": _score, "device": _sim["device"],
                                                "tts_mtime": _st.st_mtime, "tts_size": _st.st_size}
                        _sim_cache_dirty = True
                    except Exception:
                        pass
                if _sim["error"]:
                    _log(f"  ❌ 第{idx:4d}条 声纹检测失败: {_sim['error']}")
                elif _score is not None and _score < _threshold:
                    _retry_ref = (settings.get("fixed_ref_audio_female", "")
                                  if gender == "female" else settings.get("fixed_ref_audio_male", "")) or ""
                    if _retry_ref and os.path.exists(_retry_ref):
                        _log(f"  🔄 第{idx:4d}条 初次相似度 {_sim['similarity']:.3f} < {_threshold},重试中...")
                        _retry_cfg = _settings_to_tts_config(settings)
                        _retry_cfg.use_fixed_ref = True
                        _retry_cfg.use_local_tts = tts_cfg.use_local_tts  # 继承 CPU 回退结果
                        _retry_cfg.fixed_ref_audio_male = _retry_ref if gender == "male" else ""
                        _retry_cfg.fixed_ref_audio_female = _retry_ref if gender == "female" else ""
                        if not _retry_cfg.vocals_path:
                            _retry_cfg.vocals_path = cache.get_path(Step.DEMUCS, "vocals_orig.wav")
                        _retry_path = synthesize_tts_segment(
                            sub,
                            cache=cache, work_dir=work_dir,
                            tts_cfg=_retry_cfg,
                            log_cb=log_cb,
                        )
                        if _retry_path:
                            _sim2 = compare_similarity(_retry_path, _retry_ref)
                            _sim2_score = _sim2["similarity"] if _sim2 and _sim2.get("similarity") is not None else 0
                            _sim1_score = _sim.get("similarity") or 0
                            if _sim2_score > _sim1_score:
                                tts_path = _retry_path
                                _score = _sim2_score
                                _log(f"  ✅ 第{idx:4d}条 重试后相似度: {_score:.3f} (原 {_sim1_score:.3f})")
                                try:
                                    _st = os.stat(tts_path)
                                    _sim_cache[str(idx)] = {"ref": "fixed", "score": _score, "device": _sim2.get("device", "") if _sim2 else "",
                                                            "tts_mtime": _st.st_mtime, "tts_size": _st.st_size}
                                    _sim_cache_dirty = True
                                except Exception:
                                    pass
                            else:
                                _log(f"  ⚠️ 第{idx:4d}条 重试相似度不足: {_sim2_score:.3f} ≤ {_sim1_score:.3f},保留原结果")
                        else:
                            _log(f"  ⚠️ 第{idx:4d}条 重试失败,保留原结果")
                    else:
                        _log(f"  🚫 第{idx:4d}条 相似度不足,但无 {gender} 提示音")
                # 一次性写入缓存
                if _sim_cache_dirty:
                    try:
                        with open(_cache_path, 'w') as _f:
                            json.dump(_sim_cache, _f, indent=2)
                    except Exception:
                        pass
            except Exception:
                pass

        # ── 2. VAD修剪 + 对齐 + 增益 + 混音（复用 mix_tts_segment)──
        mixed_clip, trim_info, vocals_ref, aligned_path, gain_path, padded_path = mix_tts_segment(
            sub=sub,
            tts_path=tts_path,
            vocals_path=cache.get_path(Step.DEMUCS, "vocals_orig.wav"),
            bg_path=cache.get_path(Step.DEMUCS, "background.wav"),
            cache=cache, work_dir=work_dir,
            vad_mode=settings.get("vad_mode", "原声对齐"),
            vad_pad_ms=settings.get("vad_pad_ms", 50),
            edge_ms=edge_ms,
            fade=True,  # 与流水线保持一致
            log_cb=log_cb,
            file_log_cb=None,
        )
        # 注：交叉淡变宽度在重新拼接(remix_from_cache)时由 splice 实际计算并打印,
        # 此处单条片段尚未拼接、无法预知真实淡变,故不再输出易误解的 mix 偏移行。

        # 输出混音日志 (与 AudioMixAndMergeStep 一致)
        if trim_info:
            from .utils import fmt_time as _t
            _tts_raw = int(trim_info.get('speech_start_ms', 0))
            _tts_lead = int(trim_info.get('leading_trimmed_ms', 0))
            _tts_speech = max(0, _tts_raw - _tts_lead)
            _fade_in_abs = start_ms + _tts_speech
            _tts_dur = int(trim_info.get('duration_ms', end_ms - start_ms))
            _fade_out_abs = start_ms + _tts_dur - _tts_lead
            _fade_out_offset = _fade_out_abs - end_ms
            _raw_tail = f"时长{(end_ms - start_ms)/1000:.3f}s"
            _raw_part1 = f"{idx:>3d} raw: 字幕开始= {_t(start_ms)}"
            _raw_part2 = f"字幕结束= {_t(end_ms)}"
            _log(f"{_raw_part1:<42s}{_raw_part2:<28s}  {_raw_tail}")
            _tts_tail = f"时长{(_fade_out_abs - _fade_in_abs)/1000:.3f}s"
            _tts_part1 = f"    tts: 淡入开始= {_t(_fade_in_abs)}({_tts_speech:+5d}ms)"
            _tts_part2 = f"淡出结束= {_t(_fade_out_abs)}({_fade_out_offset:+5d}ms)"
            _log(f"{_tts_part1:<42s}{_tts_part2:<28s}  {_tts_tail}")

        # 清理临时文件（不含 _ref_ 参考音频,保留供下次复用)
        for tmp in [vocals_ref, aligned_path, gain_path, padded_path]:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

        if done_cb:
            done_cb(True, mixed_clip)

    except Exception as e:
        _log(f"【错误】重生成第 {idx} 条失败: {e}")
        if done_cb:
            done_cb(False, "")

