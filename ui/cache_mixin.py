"""缓存管理 Mixin：缓存状态检查、清理、校准加载、TTS 片段删除

所需 self 属性（由 MainWindow.__init__ 初始化):
    self._cache, self._cache_video_path, self._cached_cache_size
    self.video_path_edit, self.srt_path_edit, self.output_dir_edit
    self.lbl_cache_status, self.btn_clear_cache, self.btn_open_output
    self.player (QMediaPlayer, 清空缓存时需停止)
    self._subtitle_model, self._src_model
    self.subtitle_row_map, self._row_to_idx
    self._tts_cache_hits, self._last_cache_detail, self._last_srt_path
    self._last_output_path, self.current_subtitles
    self.tts_paths, self.raw_tts_paths
    self._pipeline_running
    self._src_title_label, self._btn_switch_src, self.src_count_label
    self.src_table, self.settings

方法:
    self.log(), self._load_subtitles()
"""
import os
import glob
from pathlib import Path

from PySide6.QtCore import QUrl

from ui.table_models import SRC_COLUMNS
from core.cache_manager import Step


class CacheMixin:
    """缓存状态检查、清理、加载、TTS片段删除"""

    def _get_cache(self):
        """获取当前视频的 CacheManager 实例（带缓存,路径变化时自动重建)"""
        from core.cache_manager import CacheManager
        video_path = self.video_path_edit.text().strip()
        if not video_path or not os.path.exists(video_path):
            return None
        if self._cache is not None and self._cache_video_path == video_path:
            return self._cache
        self._cache = CacheManager(video_path, os.path.join(os.getcwd(), ".cache"))
        self._cache_video_path = video_path
        return self._cache

    def _set_status(self, row: int, status_key: str):
        """统一设置字幕状态列（column 10)"""
        self._subtitle_model.set_status(row, status_key)

    def _update_cache_status(self, force_all=False):
        if self._pipeline_running and not force_all:
            return

        step_labels_num = {
            "extract":    "1",
            "demucs":     "2",
            "subs":       "3",
            "tts":        "4",
            "mix":        "5",
        }
        detail_parts = []
        step_status = {}
        cm = self._get_cache()
        if cm:
            try:
                step_status = cm.get_all_step_status()
                for k in step_labels_num:
                    has = step_status.get(k, False)
                    detail_parts.append(f"step{step_labels_num[k]}:{'✅' if has else '⏳'}")
            except Exception:
                pass
        if detail_parts and (force_all or detail_parts != self._last_cache_detail):
            self._last_cache_detail = detail_parts
            for line in detail_parts:
                self.log(f"缓存检查 | {line}")

        if cm and force_all:
            total_size, count = cm.calculate_total_cache_size()
            self._cached_cache_size = (total_size, count)
        elif hasattr(self, '_cached_cache_size'):
            total_size, count = self._cached_cache_size
        else:
            total_size, count = 0, 0
        if total_size > 0:
            size_str = f"{total_size/1024/1024:.1f}MB" if total_size > 1024*1024 else f"{total_size/1024:.0f}KB"
            self.lbl_cache_status.setText(f"缓存: {count} 个文件 ({size_str})")
            self.btn_clear_cache.setEnabled(True)
        elif cm:
            _count = sum(1 for _ in os.scandir(cm.cache_dir)) if os.path.exists(cm.cache_dir) else 0
            self.lbl_cache_status.setText(f"缓存: {_count} 个子目录")
            self.btn_clear_cache.setEnabled(True)
        else:
            self.lbl_cache_status.setText("缓存: 无")
            self.btn_clear_cache.setEnabled(False)

        video_path = self.video_path_edit.text().strip()
        if video_path:
            out_name = f"{Path(video_path).stem}.ts.mp4"
            out_dir = str(Path(video_path).parent)
            output_file = os.path.join(out_dir, out_name)
            if os.path.exists(output_file):
                self._last_output_path = output_file
                self.btn_open_output.setEnabled(True)

    def _clear_cache(self):
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QDialogButtonBox, QLabel, QPushButton

        self.player.stop()
        self.player.setSource(QUrl())
        self._mixed_audio_path = ""

        cm = self._get_cache()
        if not cm:
            self.log("【警告】请先选择视频文件")
            return
        cache_dir = cm.cache_dir
        if not os.path.exists(cache_dir) or not any(os.listdir(cache_dir)):
            self.log("当前视频无缓存")
            for e in os.listdir(cm.cache_root):
                p = os.path.join(cm.cache_root, e)
                if os.path.isdir(p) and os.listdir(p):
                    self.log("其他视频存在缓存,请手动删除 .cache 目录")
                    break
            self._update_cache_status()
            return
        step_labels = {
            "extract": "1.提取音频",
            "demucs": "2.Demucs人声分离",
            "subs": "3.字幕处理+性别标记",
            "tts": "4.TTS合成",
            "tts_similarity": "4.1 声纹校验缓存",
            "mix": "5.音频合成",
        }
        _all_labels = {**step_labels, "subs_gender": "3.2 性别检测", "subs_align": "3.1 字幕对齐"}
        dlg = QDialog(self)
        dlg.setWindowTitle(f"分步清空缓存 — {Path(cm.video_path).stem}")
        dlg.setMinimumWidth(420)
        lo = QVBoxLayout(dlg)
        total_step_size, total_step_count = cm.calculate_total_cache_size()
        si = f"{total_step_size/1024/1024:.1f}MB" if total_step_size > 1024*1024 else f"{total_step_size/1024:.0f}KB"
        lo.addWidget(QLabel(f"当前视频缓存: {total_step_count} 个文件 ({si})\n选择要清空的step:"))
        cbs = {}
        for k, lbl in step_labels.items():
            has = False
            file_count = 0
            if k == "tts_similarity":
                _files = glob.glob(os.path.join(cm.cache_dir, "tts", ".similarity_cache.json"))
                has = len(_files) > 0
                file_count = len(_files)
            else:
                step_dir = os.path.join(cache_dir, k)
                if os.path.exists(step_dir):
                    cfs = cm.STEP_CHECK_FILES.get(k, [])
                    if cfs:
                        for cf in cfs:
                            if '*' in cf:
                                matches = glob.glob(os.path.join(step_dir, cf))
                                file_count = len(matches)
                                if matches:
                                    has = True
                                    break
                            else:
                                fp = os.path.join(step_dir, cf)
                                if os.path.exists(fp):
                                    all_f = [f for f in os.listdir(step_dir) if os.path.isfile(os.path.join(step_dir, f))]
                                    file_count = len(all_f)
                                    has = True
                                    break
                    else:
                        all_f = [f for f in os.listdir(step_dir) if os.path.isfile(os.path.join(step_dir, f))]
                        file_count = len(all_f)
                        has = file_count > 0
            if not has:
                file_count = 0
            if has and file_count > 0:
                cb_text = f"{lbl}  ({file_count} 个文件)"
            elif has:
                cb_text = f"{lbl}  (有缓存)"
            else:
                cb_text = f"{lbl}  (无)"
            if k == "subs":
                from PySide6.QtWidgets import QLabel as _QL
                _lb = _QL(f"  {lbl}")
                _lb.setStyleSheet("font-weight: bold; color: #666; padding-left: 4px;")
                lo.addWidget(_lb)
                for _sk, _sl, _sp in [
                    ("subs_align", "3.1 字幕对齐", "calib.json"),
                    ("subs_gender", "3.2 性别检测", "genders_cache.json"),
                ]:
                    _sub_has, _sub_path, _sub_rel = cm.file_info(Step.SUBS, _sp)
                    _sub_cb = QCheckBox(f"{_sl}  ({'有缓存' if _sub_has else '无'})")
                    _sub_cb.setEnabled(_sub_has)
                    _sub_cb.setStyleSheet("padding-left: 20px;")
                    lo.addWidget(_sub_cb)
                    cbs[_sk] = _sub_cb
            else:
                cb = QCheckBox(cb_text)
                auto_check = has and k not in ("extract", "demucs", "gender", "tts")
                cb.setChecked(auto_check)
                cb.setEnabled(has)
                lo.addWidget(cb)
                cbs[k] = cb
        hr = QHBoxLayout()
        ba = QPushButton("全选")
        bn = QPushButton("取消全选")
        hr.addWidget(ba)
        hr.addWidget(bn)
        hr.addStretch()
        lo.addLayout(hr)
        def tog(v): [c.setChecked(v) for c in cbs.values() if c.isEnabled()]
        ba.clicked.connect(lambda: tog(True))
        bn.clicked.connect(lambda: tog(False))
        bt = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bt.button(QDialogButtonBox.StandardButton.Ok).setText("清空选中")
        bt.accepted.connect(dlg.accept)
        bt.rejected.connect(dlg.reject)
        lo.addWidget(bt)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        any_ = False
        for k, cb in cbs.items():
            if cb.isChecked():
                if k != "tts_similarity" and k != "subs_gender" and k != "subs_align":
                    cm.clear_step(k)
                if k == "subs_gender":
                    _gf = cm.get_path(Step.SUBS, "genders_cache.json")
                    if os.path.exists(_gf): 
                        try:
                            os.remove(_gf)
                        except Exception:
                            pass
                    for row in range(self._subtitle_model.rowCount()):
                        self._subtitle_model.update_gender(row + 1, "")
                elif k == "subs_align":
                    self.log(f"  清理字幕校准缓存: .cache{os.sep}{os.path.basename(cm.cache_dir)}{os.sep}{cm.step_dir_rel(Step.SUBS)}")
                    for _af in ["calib.json"]:
                        _p = cm.get_path(Step.SUBS, _af)
                        if os.path.exists(_p):
                            try:
                                os.remove(_p)
                                self.log(f"    已删除: {_af}")
                            except Exception as _ex:
                                self.log(f"    删除失败: {_af} ({_ex})")
                        else:
                            self.log(f"    不存在: {_af}")

                    cm.mark_incomplete(Step.SUBS)
                    self.log("  字幕校准缓存已清理,标记为未完成")
                    self._subtitle_model.reset_all_calib()
                    self._src_view_mode = 0
                    self._btn_switch_src.setVisible(False)
                    self._btn_switch_src.setEnabled(False)
                    if hasattr(self, '_src_subs') and self._src_subs:
                        # 清除所有校准字段
                        for s in self._src_subs:
                            s.calib_start_ms = 0
                            s.calib_end_ms = 0
                        self._src_model.set_data(self._src_subs, is_calibrated=False)
                        for ci, (_, w) in enumerate(SRC_COLUMNS):
                            if w > 0:
                                self.src_table.setColumnWidth(ci, w)
                    if hasattr(self, '_src_title_label'):
                        self._src_title_label.setText("<b>原声字幕</b>")
                    self.src_count_label.setText(f"原声字幕: {self._src_model.rowCount()} 条")
                if k == "tts":
                    self._tts_cache_hits.clear()
                    row_to_idx = self._row_to_idx if hasattr(self, '_row_to_idx') else {v: k for k, v in self.subtitle_row_map.items()}
                    for row in range(self._subtitle_model.rowCount()):
                        if row in row_to_idx:
                            self._set_status(row, "pending")
                if k == "mix":
                    for _f in glob.glob(os.path.join(cm.cache_dir, "tts", "mixed_*.wav")):
                        try:
                            os.remove(_f)
                        except Exception:
                            pass
                if k == "subs":
                    if hasattr(self, '_src_subs'):
                        for s in self._src_subs:
                            s.calib_start_ms = 0
                            s.calib_end_ms = 0
                    self._src_view_mode = 0
                    self._btn_switch_src.setVisible(False)
                    self._btn_switch_src.setEnabled(False)
                    self._subtitle_model._changed_indices.clear()
                    src_path = self.src_srt_path_edit.text().strip()
                    if src_path and os.path.exists(src_path):
                        from core.srt_parser import parse_srt
                        try:
                            subs = parse_srt(src_path)
                            self._src_subs = subs
                            self._src_model.set_data(subs, is_calibrated=False)
                            for ci, (_, w) in enumerate(SRC_COLUMNS):
                                if w > 0:
                                    self.src_table.setColumnWidth(ci, w)
                        except Exception:
                            pass
                    if hasattr(self, '_src_title_label'):
                        self._src_title_label.setText("<b>原声字幕</b>")
                    self.src_count_label.setText(f"原声字幕: {self._src_model.rowCount()} 条")
                    _n = self._subtitle_model.rowCount()
                    if _n > 0:
                        self._subtitle_model.dataChanged.emit(
                            self._subtitle_model.index(0, 0),
                            self._subtitle_model.index(_n - 1, 10)
                        )
                if k == "mix":
                    for p in glob.glob(os.path.join(cm.cache_dir, "tts", "mixed_*.wav")):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                    self._tts_cache_hits.clear()
                    for row in range(self._subtitle_model.rowCount()):
                        idx = self._row_to_idx.get(row)
                        if idx is None:
                            continue
                        if glob.glob(os.path.join(cm.cache_dir, "tts", f"tts_{idx:04d}_*.wav")):
                            self._on_tts_cache_hit(idx)
                if k == "tts_similarity":
                    for p in glob.glob(os.path.join(cm.cache_dir, "tts", ".similarity_cache.json")):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                if k == "subs":
                    self._last_srt_path = ""
                    self._load_subtitles()
                any_ = True
                self.log(f"🚮 已清空: {_all_labels.get(k, k)}")
            elif cb.isEnabled():
                self.log(f"跳过清空: {_all_labels.get(k, k)}")
        self._update_cache_status(force_all=True) if any_ else self.log("未选择任何step")

    def _delete_tts_segment(self, row: int):
        """删除单条 TTS 缓存"""
        self.player.stop()
        import time as _time
        idx = self._row_to_idx.get(row)
        if idx is None:
            self.log(f"【警告】无对应字幕索引: row={row}")
            return
        cm = self._get_cache()
        if not cm:
            self.log("【警告】请先选择视频文件")
            return
        cache_dir = cm.cache_dir
        if not os.path.isdir(cache_dir):
            self.log(f"【警告】缓存目录不存在: .cache{os.sep}{os.path.basename(cache_dir)}")
            return
        tts_dir = os.path.join(cache_dir, "tts")
        deleted = False
        for pattern in [f"tts_{idx:04d}_*.wav", f"mixed_{idx:04d}_*.wav"]:
            for p in glob.glob(os.path.join(tts_dir, pattern)):
                for _ in range(5):
                    try:
                        os.remove(p)
                        deleted = True
                        break
                    except PermissionError:
                        _time.sleep(0.2)
        self._tts_cache_hits.pop(idx, None)
        if cm.is_step_completed(Step.TTS):
            cm.mark_incomplete(Step.TTS)
        if deleted:
            self.tts_paths.pop(row, None)
            self.raw_tts_paths.pop(row, None)
            self._subtitle_model.set_status(row, "pending")
            self.log(f"已删除第 {idx} 条 TTS 缓存")
            self._update_cache_status()
        else:
            self.log(f"【警告】第 {idx} 条没有缓存可删除")

    def _rescan_mixed_cache(self):
        """重新扫描 mixed_{idx}_*.wav 缓存,更新试听按钮状态"""
        cm = self._get_cache()
        if not cm:
            return
        _mixed_map = {}
        for _f in glob.glob(os.path.join(cm.cache_dir, "tts", "mixed_*.wav")):
            _bn = os.path.basename(_f)
            _idx_str = _bn.split("_")[1]
            if _idx_str not in _mixed_map:
                _mixed_map[_idx_str] = _f
        _mixed_exists, _, _ = cm.file_info(Step.MIX, "final_audio.wav")
        for row in range(self._subtitle_model.rowCount()):
            idx = self._row_to_idx.get(row)
            if idx is None:
                continue
            mixed_path = _mixed_map.get(f"{idx:04d}")
            if mixed_path:
                self.tts_paths[row] = mixed_path
                status_key = "mixed" if _mixed_exists else "tts_done"
                self._set_status(row, status_key)

