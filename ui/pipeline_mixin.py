"""流水线回调 + UI 状态更新 Mixin

所需 self 属性:
    self._pipeline_worker, self._single_step_worker
    self._tts_error_dlg
    self.btn_start, self.btn_cancel, self.btn_play_mix, self.btn_open_output
    self.btn_stop_mix
    self.prompt_mode_original, self.prompt_mode_fixed
    self.waveform_preview
    self._step_cells, self._step_progress_bar, self._step_progress_label, self._step_progress_pct
    self.step_names
    self._subtitle_model, self._src_model, self._gender_proxy
    self.subtitle_row_map, self._row_to_idx
    self._tts_cache_hits, self.current_subtitles, self._segments_data
    self.srt_path_edit, self.video_path_edit, self.output_dir_edit
    self._src_title_label, self._btn_switch_src, self.src_count_label, self.src_table
    self.settings, self._pipeline_running
    self._last_output_path, self._last_srt_path

方法:
    self.log(), self.log_file()
    self._get_cache(), self._set_status(), self._fmt_time()
    self._load_waveform_preview(), self._update_cache_status()
"""
import os
import glob
import re
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from ui.table_models import SRC_COLUMNS

_STEP_COUNT_RE = re.compile(r"(\(\d+/\d+\))")


class PipelineMixin:
    """流水线信号回调、步骤进度 UI、字幕状态更新"""

    def _reset_pipeline_ui_state(self):
        """重置流水线运行时的 UI 状态（全流程/单步共用)"""
        self._pipeline_running = False
        self._gender_proxy.set_pipeline_running(False)
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.prompt_mode_original.setEnabled(True)
        self.prompt_mode_fixed.setEnabled(True)

    def _on_tts_step_completed(self, idx: int, tts_path: str, status: str):
        """TTS 片段完成：合并更新按钮、状态、试听可用性"""
        row = self.subtitle_row_map.get(idx)
        if row is None:
            return
        self.tts_paths[row] = tts_path
        cm = self._get_cache()
        if cm:
            raw_path = cm.find_raw_tts(idx)
            if raw_path:
                self.raw_tts_paths[row] = raw_path
        key = "mixed" if status == "mixed_done" else status
        self._set_status(row, key)

    def _on_subs_ready(self, subs: list):
        """字幕处理完成 — 直接替换模型数据"""
        self.current_subtitles = subs
        self._subtitle_model.set_data(subs)
        self._segments_data = [(s.eff_start_ms, s.eff_end_ms) for s in subs]
        self.waveform_preview.set_segments(self._segments_data)
        n = self._subtitle_model.rowCount()
        if n > 0:
            self._subtitle_model.dataChanged.emit(
                self._subtitle_model.index(0, 0),
                self._subtitle_model.index(n - 1, self._subtitle_model.columnCount() - 1)
            )
        if hasattr(self, '_gender_proxy'):
            self._gender_proxy.invalidateFilter()
        if any(s.is_calibrated and (s.calib_start_ms != s.start_ms or s.calib_end_ms != s.end_ms) for s in subs):
            if not getattr(self, '_fix_srt_generated', False):
                self._fix_srt_generated = True
                self._generate_fixed_srt()

    def _on_calib_src_ready(self, segments: list):
        """校准后的原声字幕就绪 — 更新 SRC 表时间,标记有变动的条目

        Args:
            segments: List[SubtitleItem] — 原声字幕列表（calib 已写入字段)
        """
        if not segments:
            return
        self._get_cache()
        if hasattr(self, '_src_title_label'):
            _mode_map = {0: "<b>原声字幕</b>", 1: "<b>原声字幕（Qwen校准)</b>"}
            self._src_title_label.setText(_mode_map.get(getattr(self, '_src_view_mode', 0), "<b>原声字幕</b>"))
        self._src_subs = segments
        self._btn_switch_src.setVisible(True)
        self._btn_switch_src.setEnabled(True)
        if self._src_view_mode == 1:
            self._btn_switch_src.setText("切换到原始")
        else:
            self._btn_switch_src.setText("切换到Qwen")
        self._src_model.set_data(segments, is_calibrated='Qwen')
        self.src_table.viewport().update()
        self.src_count_label.setText(f"原声字幕（校准后): {len(segments)} 条")
        for ci, (_, w) in enumerate(SRC_COLUMNS):
            if w > 0:
                self.src_table.setColumnWidth(ci, w)

    def _on_tts_item_ready(self, idx: int, tts_path: str):
        """单条 TTS 合成完成 → 更新按钮、路径、状态"""
        row = self.subtitle_row_map.get(idx)
        if row is None:
            self.log(f"⚠️ TTS完成但 row 映射不存在: idx={idx}")
            return
        if not tts_path:
            self._set_status(row, "skipped")
            return
        if os.path.exists(tts_path):
            self.raw_tts_paths[row] = tts_path
        self._on_tts_step_completed(idx, tts_path, "tts_done")

    def _on_tts_cache_hit(self, idx: int):
        """TTS 缓存命中 → 状态设为「缓存命中」"""
        row = self.subtitle_row_map.get(idx)
        if row is None:
            return
        self._set_status(row, "tts_done")

    def _on_mix_item_ready(self, ids: list):
        """每 5 条混音完成 → 更新播放路径为混音片段"""
        cm = self._get_cache()
        for idx in ids:
            row = self.subtitle_row_map.get(idx)
            if row is None:
                continue
            if cm and 1 <= idx <= len(self.current_subtitles):
                sub = self.current_subtitles[idx - 1]
                if sub:
                    mixed_path = cm.mixed_path(sub)
                    if os.path.exists(mixed_path):
                        self.tts_paths[row] = mixed_path
            self._set_status(row, "mixed")
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.processEvents()

    def _on_mixed_audio_ready(self, mixed_path: str):
        """混音完成,激活预览"""
        self.waveform_preview.setEnabled(True)
        self._load_waveform_preview(mixed_path)
        self.log(f"混音预览就绪: {os.path.basename(mixed_path)}")

    def _on_finished(self, success: bool, message: str):
        self._reset_pipeline_ui_state()
        if success:
            video_path = self.video_path_edit.text().strip()
            if video_path:
                out_name = f"{Path(video_path).stem}.ts.mp4"
                self._last_output_path = os.path.join(str(Path(video_path).parent), out_name)
            self.btn_open_output.setEnabled(True)
            self._update_cache_status()
        else:
            if message != "已取消":
                QMessageBox.critical(self, "错误", f"配音失败:\n{message}")

        if getattr(self, '_pipeline_worker', None):
            self._pipeline_worker = None
        self._update_cache_status()
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.processEvents()
        self._rescan_mixed_cache()

    def _update_step_ui(self, active_idx: int):
        """更新分段进度图标和颜色"""
        GREEN = "#22A65E"
        BLUE = "#2196F3"
        for idx, (icon, text, _) in enumerate(self._step_cells):
            if idx < active_idx:
                icon.setText("✅")
                text.setStyleSheet(f"color: {GREEN}; font-weight: bold;")
            elif idx == active_idx:
                icon.setText("▶")
                text.setStyleSheet(f"color: {BLUE}; font-weight: bold;")
            else:
                icon.setText("⏳")
                text.setStyleSheet("color: #000000;")

    def _on_step_progress(self, step: int, pct: int, name: str):
        """更新分段进度 — 共用进度条模式"""
        if step < 0 or step >= len(self._step_cells):
            return
        _m = _STEP_COUNT_RE.search(name)
        _count_text = _m.group(1) if _m else ""
        self._step_progress_bar.setValue(pct)
        if _count_text:
            self._step_progress_pct.setText(f"{_count_text}  {pct}%")
        else:
            self._step_progress_pct.setText(f"{pct}%")
        # 已完成步骤不接受活跃态更新,防止 Phase2/Phase3 将 ✅ 改回 ▶
        icon, text, _ = self._step_cells[step]
        if icon.text() == "✅" and pct < 100:
            return
        if pct == 100:
            self._step_progress_label.setText("✅ 完成")
            self._step_progress_label.setStyleSheet("color: #22A65E; font-weight: bold;")
            self._update_step_ui(step + 1)
        else:
            self._step_progress_label.setText(self.step_names[step])
            self._step_progress_label.setStyleSheet("color: #000000;")
            self._update_step_ui(step)
            icon, text, _ = self._step_cells[step]
            if icon.text() != "▶":
                icon.setText("▶")
            text.setStyleSheet("color: #2196F3; font-weight: bold;")

    def _on_step_finished(self, success: bool, msg: str):
        """单步执行完成后的回调"""
        self._reset_pipeline_ui_state()
        if success:
            self.log(f"✅ {msg}")
        else:
            self.log(f"⛔⛔ {msg}")
        # 清除已完成 worker 引用,避免陈旧引用干扰后续错误对话框/配置刷新查找
        if getattr(self, '_single_step_worker', None):
            self._single_step_worker = None
        self._update_cache_status()

    def _on_subtitle_error(self, idx: int, error_msg: str, text: str):
        """TTS 错误时弹出非模态对话框"""
        if self._tts_error_dlg is not None:
            try:
                _old = self._tts_error_dlg.text()
                self._tts_error_dlg.setText(_old + f"\n\n第 {idx} 条追加失败:\n  {error_msg[:200]}")
                return
            except Exception:
                pass

        short_text = text[:60].replace('\n', ' ')
        if len(text) > 60:
            short_text += '...'
        msg = (
            f"第 {idx} 条字幕 TTS 合成失败:\n"
            f"  {error_msg[:200]}\n\n"
            f"字幕: 「{short_text}」\n\n"
            f"请选择操作（可先修改「⚙设置」中的 TTS 配置再重试):"
        )
        dlg = QMessageBox(self)
        dlg.setWindowTitle("TTS 错误")
        dlg.setText(msg)
        dlg.setIcon(QMessageBox.Icon.Warning)
        retry_btn = dlg.addButton("🔄 重试", QMessageBox.ButtonRole.AcceptRole)
        dlg.addButton("⏭️ 跳过", QMessageBox.ButtonRole.RejectRole)
        dlg.addButton("⛔ 终止", QMessageBox.ButtonRole.DestructiveRole)
        dlg.setDefaultButton(retry_btn)
        dlg.setModal(False)
        _choice_made = False  # 标记是否已通过按钮做出选择

        def _on_choice(choice: str):
            nonlocal _choice_made
            _choice_made = True
            self._tts_error_dlg = None
            _w = getattr(self, '_pipeline_worker', None) or getattr(self, '_single_step_worker', None)
            if _w is None:
                return
            # 先设置响应并唤醒,再关闭对话框(close 会同步触发 finished 信号)
            _w._error_mutex.lock()
            _w._error_response = choice
            _w._error_cond.wakeAll()
            _w._error_mutex.unlock()
            try:
                dlg.close()
            except Exception:
                pass

        def _on_btn_clicked(btn):
            role = dlg.buttonRole(btn)
            if role == QMessageBox.ButtonRole.AcceptRole:
                _on_choice("retry")
            elif role == QMessageBox.ButtonRole.RejectRole:
                _on_choice("skip")
            elif role == QMessageBox.ButtonRole.DestructiveRole:
                _on_choice("abort")

        dlg.buttonClicked.connect(_on_btn_clicked)
        def _on_finished(_):
            # 仅在未通过按钮做出选择时(如 Esc/点 X 关闭)兜底为 skip
            if _choice_made:
                return
            if self._tts_error_dlg is dlg:
                self._tts_error_dlg = None
                _w = getattr(self, '_pipeline_worker', None) or getattr(self, '_single_step_worker', None)
                if _w is not None:
                    _w._error_mutex.lock()
                    if _w._error_response is None:
                        _w._error_response = "skip"
                        _w._error_cond.wakeAll()
                    _w._error_mutex.unlock()
        dlg.finished.connect(_on_finished)

        self._tts_error_dlg = dlg
        dlg.show()

    def _refresh_worker_ctx_config(self):
        """worker 重试前用最新配置覆盖 ctx"""
        from config import cfg
        worker = getattr(self, '_pipeline_worker', None) or getattr(self, '_single_step_worker', None)
        if worker is None or not hasattr(worker, 'ctx'):
            return
        for k, v in cfg.__dict__.items():
            setattr(worker.ctx, k, v)
        # 说话人嵌入路径由 UI 当前选择决定
        def _get_emb(combo):
            return (
                    os.path.join(os.getcwd(), "role", combo.currentText() + ".index.pt")
                    if combo.currentIndex() > 0 else "")
        worker.ctx.speaker_embedding_path_male = _get_emb(
            getattr(self, 'speaker_emb_male', None))
        worker.ctx.speaker_embedding_path_female = _get_emb(
            getattr(self, 'speaker_emb_female', None))
        worker._config_mutex.lock()
        worker._config_refreshed = True
        worker._config_cond.wakeAll()
        worker._config_mutex.unlock()
        self.log("🔄 TTS 配置已刷新,使用最新设置重试")

    def _generate_fixed_srt(self):
        """生成校正后的字幕文件"""
        srt_path = self.srt_path_edit.text().strip()
        if not srt_path or not os.path.exists(srt_path):
            return
        if not self.current_subtitles or self._subtitle_model.rowCount() == 0:
            return

        from core.utils import fmt_time

        fixed_lines = []
        for j, sub in enumerate(self.current_subtitles, 1):
            row = self.subtitle_row_map.get(j)
            if row is not None:
                start_ms, end_ms = self._subtitle_model.get_times(row)
            else:
                start_ms, end_ms = sub.start_ms, sub.end_ms
            fixed_lines.append({
                'index': sub.idx,
                'start_ms': start_ms,
                'end_ms': end_ms,
                'text': sub.text
            })

        srt_path_obj = Path(srt_path)
        fix_path = srt_path_obj.with_name(f"{srt_path_obj.stem}_fix{srt_path_obj.suffix}")
        try:
            with open(fix_path, 'w', encoding='utf-8') as f:
                for entry in fixed_lines:
                    f.write(f"{entry['index']}\n")
                    f.write(f"{fmt_time(entry['start_ms'], ms_sep=',')} --> {fmt_time(entry['end_ms'], ms_sep=',')}\n")
                    f.write(f"{entry['text']}\n")
                    f.write("\n")
            self.log(f"  已生成校正字幕: {os.path.basename(fix_path)}")
        except Exception as ex:
            self.log(f"【警告】生成校正字幕失败: {ex}")

    def reset_preview_buttons(self):
        """重置所有试听按钮和状态为初始值"""
        cm = self._get_cache()
        row_to_idx = self._row_to_idx if hasattr(self, '_row_to_idx') else {v: k for k, v in self.subtitle_row_map.items()}
        _mixed_set = set()
        _tts_set = set()
        if cm:
            for _f in glob.glob(os.path.join(cm.cache_dir, "tts", "*.wav")):
                _bn = os.path.basename(_f)
                if _bn.startswith("mixed_"):
                    _mixed_set.add(_bn.split("_")[1])
                elif _bn.startswith("tts_"):
                    _tts_set.add(_bn.split("_")[1])
        for row in range(self._subtitle_model.rowCount()):
            idx = row_to_idx.get(row)
            if idx is not None and idx in self._tts_cache_hits:
                continue
            if cm and idx is not None:
                _idx_str = f"{idx:04d}"
                if _idx_str in _mixed_set:
                    self._set_status(row, "mixed")
                    continue
                elif _idx_str in _tts_set:
                    self._set_status(row, "tts_done")
                    continue
            self._set_status(row, "pending")
        self.btn_stop_mix.setEnabled(False)
        self.lbl_play_time.setText("00:00 / 00:00")
        for icon, text, _ in self._step_cells:
            icon.setText("⏳")
            text.setStyleSheet("color: #000000;")
        self._step_progress_bar.setValue(0)
        self._step_progress_label.setText(self.step_names[0])
        self._step_progress_pct.setText("0%")
