"""执行控制 Mixin：流水线/单步执行启动、取消、上下文构建、重新混音/重新合成

所需 self 属性:
    self._pipeline_worker, self._single_step_worker
    self._regen_thread, self._regen_queue, self._regen_result
    self.btn_start, self.btn_cancel, self.btn_play_mix
    self.prompt_mode_original, self.prompt_mode_fixed
    self.waveform_preview
    self.video_path_edit, self.srt_path_edit, self.output_dir_edit
    self.src_srt_path_edit, self.keep_temp_cb
    self._pipeline_running
    self.current_subtitles, self._subtitle_model
    self._row_to_idx, self.subtitle_row_map
    self._log_signal

方法:
    self.log(), self.log_file()
    self._get_cache(), self._set_status()
    self._delete_tts_segment(), self._update_cache_status()
    self._on_tts_step_completed(), self._on_mixed_audio_ready()
    self.reset_preview_buttons()
"""
import os
import tempfile
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from ui.pipeline_worker import PipelineWorker, TaskThread



class ExecutionMixin:
    """流水线/单步执行启动、取消、上下文构建、重新混音/重新合成"""

    def _validate_paths(self, video_path: str, srt_path: str, output_dir: str = "") -> tuple:
        """校验视频/字幕路径,返回 (ok: bool, output_dir: str)"""
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "提示", "请选择有效的视频文件")
            return False, ""
        if not srt_path or not os.path.exists(srt_path):
            QMessageBox.warning(self, "提示", "请选择有效的字幕文件")
            return False, ""
        if not output_dir:
            output_dir = str(Path(video_path).parent)
            self.output_dir_edit.setText(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        return True, output_dir

    def _set_pipeline_running(self, running: bool):
        """统一设置流水线运行时的 UI 状态"""
        self._pipeline_running = running
        self._gender_proxy.set_pipeline_running(running)
        self.btn_start.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        if running:
            self.prompt_mode_original.setEnabled(False)
            self.prompt_mode_fixed.setEnabled(False)
        else:
            self.prompt_mode_original.setEnabled(True)
            self.prompt_mode_fixed.setEnabled(True)

    def _build_context(self, video_path: str, srt_path: str, output_dir: str):
        """统一构建 PipelineContext"""
        from core.pipeline import PipelineContext
        cache_root = os.path.join(os.getcwd(), ".cache")
        ctx = PipelineContext(video_path, srt_path, output_dir, cache_root)

        raw_src_srt = self.src_srt_path_edit.text().strip()
        if raw_src_srt and os.path.exists(raw_src_srt):
            ctx.raw_src_path = raw_src_srt

        # 全局唯一配置参数来自 self.cfg
        for k, v in self.cfg.__dict__.items():
            setattr(ctx, k, v)
        ctx.keep_temp = self.keep_temp_cb.isChecked()
        # 说话人嵌入路径来自 UI 当前选择
        _get_emb = lambda combo: (
            os.path.join(os.getcwd(), "role", combo.currentText())
            if combo.currentIndex() > 0 else "")
        ctx.speaker_embedding_path_male = _get_emb(
            getattr(self, 'speaker_emb_male', None) or type('o',(),{'currentIndex':lambda:0})())
        ctx.speaker_embedding_path_female = _get_emb(
            getattr(self, 'speaker_emb_female', None) or type('o',(),{'currentIndex':lambda:0})())
        return ctx

    def _start_dub(self):
        video_path = self.video_path_edit.text().strip()
        srt_path = self.srt_path_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()

        ok, output_dir = self._validate_paths(video_path, srt_path, output_dir)
        if not ok:
            return

        self.tts_paths.clear()
        self.raw_tts_paths.clear()
        self.reset_preview_buttons()
        self.waveform_preview.set_waveform([], 1)
        self.waveform_preview.set_position(0)
        self.waveform_preview.setEnabled(False)
        _mode = "固定提示音" if self.cfg.use_fixed_ref else "原视频"
        _pt_male = getattr(self, 'speaker_emb_male', None)
        _pt_female = getattr(self, 'speaker_emb_female', None)
        _m = _pt_male.currentText() if _pt_male and _pt_male.currentIndex() > 0 else "无"
        _f = _pt_female.currentText() if _pt_female and _pt_female.currentIndex() > 0 else "无"
        self.log(f"提示音模式: {_mode}, 男声pt: {_m}, 女声pt: {_f}")

        ctx = self._build_context(video_path, srt_path, output_dir)
        try:
            ctx.log_file = self.log_file
        except Exception:
            pass

        self._pipeline_worker = PipelineWorker(ctx)
        self._pipeline_worker.log_msg.connect(self.log)
        self._pipeline_worker.step_progress.connect(self._on_step_progress)
        self._pipeline_worker.finished.connect(self._on_finished)
        self._pipeline_worker.subs_calib_gender_ready.connect(self._on_subs_ready)
        self._pipeline_worker.calib_src_ready.connect(self._on_calib_src_ready)
        self._pipeline_worker.tts_item_ready.connect(self._on_tts_item_ready)
        self._pipeline_worker.tts_error.connect(self._on_subtitle_error)
        self._pipeline_worker.tts_cache_hit.connect(self._on_tts_cache_hit)
        self._pipeline_worker.mix_item_ready.connect(self._on_mix_item_ready)
        self._pipeline_worker.mixed_audio_ready.connect(self._on_mixed_audio_ready)
        self._pipeline_worker.refresh_ctx_config.connect(self._refresh_worker_ctx_config)

        self._set_pipeline_running(True)
        self._fix_srt_generated = False  # 重置,允许新流程生成校正字幕
        self._pipeline_worker.start()

        self.log("===== 开始配音任务 =====")
        self.log(f"视频: {Path(video_path).name}")
        self.log(f"字幕: {Path(srt_path).name}")
        raw_src_srt = self.src_srt_path_edit.text().strip()
        if raw_src_srt and os.path.exists(raw_src_srt):
            self.log(f"原声字幕: {Path(raw_src_srt).name}")
        _tts_mode = "本地引擎(IndexTTS2)" if self.cfg.use_local_tts else \
                    f"API: {self.cfg.tts_api_url} ({self.cfg.tts_mode})"
        self.log(f"TTS: {_tts_mode}")

    def _cancel_dub(self):
        if getattr(self, '_pipeline_worker', None) and self._pipeline_worker.isRunning():
            self._pipeline_worker.cancel()
            self.log("正在取消...")

    def _on_step_btn_clicked(self, step_idx: int):
        """点击分段进度中的 ▶ 按钮 → 单独执行某一步"""
        if self._pipeline_running:
            self.log(f"⚠️ 流水线正在运行,不能单独执行 step {step_idx+1}")
            return


        self._run_single_step(step_idx)

    def _run_single_step(self, step_idx: int):
        """通用：构建上下文并执行单步"""
        video_path = self.video_path_edit.text().strip()
        srt_path = self.srt_path_edit.text().strip()
        ok, _ = self._validate_paths(video_path, srt_path)
        if not ok:
            return

        ctx = self._build_context(video_path, srt_path, str(Path(video_path).parent))
        try:
            ctx.log_file = self.log_file
        except Exception:
            pass

        w = PipelineWorker(ctx, step_idx)
        w.step_progress.connect(self._on_step_progress)
        w.subs_calib_gender_ready.connect(self._on_subs_ready)
        w.calib_src_ready.connect(self._on_calib_src_ready)
        w.mixed_audio_ready.connect(self._on_mixed_audio_ready)
        w.tts_item_ready.connect(self._on_tts_item_ready)
        w.tts_cache_hit.connect(self._on_tts_cache_hit)
        w.mix_item_ready.connect(self._on_mix_item_ready)
        w.log_msg.connect(self.log)
        w.tts_error.connect(self._on_subtitle_error)
        w.refresh_ctx_config.connect(self._refresh_worker_ctx_config)
        w.finished.connect(self._on_step_finished)
        self._single_step_worker = w

        self._set_pipeline_running(True)
        self.log(f"===== 单独执行 Step {step_idx+1} =====")
        w.start()


    def _regen_segment(self, row: int):
        idx = self._row_to_idx.get(row)
        if idx is None or idx > len(self.current_subtitles):
            return
        sub = self.current_subtitles[idx - 1]
        if any(q[0] == idx for q in self._regen_queue):
            return
        self._regen_queue.append((idx, sub, row))
        self._subtitle_model.set_status(row, "tts_synthesizing")
        self.log(f"第 {idx} 条加入重新生成队列 ({len(self._regen_queue)})")
        if len(self._regen_queue) == 1:
            self._process_regen_queue()

    def _process_regen_queue(self):
        if not self._regen_queue:
            return
        from core.pipeline import regen_single_tts
        idx, sub, row = self._regen_queue[0]
        self._subtitle_model.set_status(row, "tts_synthesizing")
        self._delete_tts_segment(row)
        # 打印当前提示音模式
        _mode = "固定提示音" if self.cfg.use_fixed_ref else "原视频"
        _pt_male = getattr(self, 'speaker_emb_male', None)
        _pt_female = getattr(self, 'speaker_emb_female', None)
        _m = _pt_male.currentText() if _pt_male and _pt_male.currentIndex() > 0 else "无"
        _f = _pt_female.currentText() if _pt_female and _pt_female.currentIndex() > 0 else "无"
        self.log(f"提示音模式: {_mode}, 男声pt: {_m}, 女声pt: {_f}")
        cache = self._get_cache()
        if not cache:
            self.log("【警告】请先选择视频文件")
            self._regen_queue.pop(0)
            self._subtitle_model.set_status(row, "pending")
            return
        if not self.cfg.use_fixed_ref:
            vp = cache.vocals_path
            if not os.path.exists(vp):
                self.log("【警告】原始人声缓存不可用,请先运行完整配音")
                self._regen_queue.pop(0)
                self._update_cache_status()
                self._process_regen_queue()
                return
        # 取原声字幕文本（dots prompt_text 用)
        _prompt_text = ""
        if hasattr(self, '_src_subs') and self._src_subs:
            for _s in self._src_subs:
                if _s.idx == idx:
                    _prompt_text = _s.text or ""
                    break
        # 固定提示音模式：文本来自设置
        if self.cfg.use_fixed_ref:
            _prompt_text = self.cfg.fixed_ref_text_female if sub.gender == "female" else self.cfg.fixed_ref_text_male

        work_dir = tempfile.mkdtemp(prefix=f"dub_{Path(cache.video_path).stem}_")

        # 把 prompt_text 注入 settings
        _settings = dict(self.cfg.__dict__)
        _settings["prompt_text"] = _prompt_text
        # 说话人嵌入路径由 UI 当前选择决定
        for _g in ('male', 'female'):
            _combo = getattr(self, f'speaker_emb_{_g}', None)
            if _combo and _combo.currentIndex() > 0:
                _settings[f'speaker_embedding_path_{_g}'] = os.path.join(
                    os.getcwd(), "role", _combo.currentText())
            else:
                _settings[f'speaker_embedding_path_{_g}'] = ""

        t = TaskThread(
            target=regen_single_tts,
            args=[sub, _settings, cache, work_dir],
            kwargs={'edge_ms': self.cfg.edge_ms},
            log_cb=self._log_signal.emit,
        )
        t.done.connect(lambda result: self._on_regen_done_wrapper(result, idx))
        t.finished.connect(self._on_regen_finished)
        self._regen_thread = t
        self._regen_result = None
        t.start()

    def _on_regen_done_wrapper(self, result, idx):
        status, *args = result
        if status == 'ok' and args[0]:
            self._regen_result = (idx, args[1], "mixed_done")
        else:
            self._regen_result = None

    def _on_regen_finished(self):
        result = getattr(self, '_regen_result', None)
        try:
            if result:
                idx, mixed_clip, status = result
                self._on_tts_step_completed(idx, mixed_clip, status)
        except Exception as e:
            self.log(f"【错误】重生成结果处理失败: {e}")
        finally:
            self._regen_result = None
            self._regen_thread = None
            if self._regen_queue:
                self._regen_queue.pop(0)
            self._update_cache_status()
            if self._regen_queue:
                self._process_regen_queue()
