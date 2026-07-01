"""播放控制 Mixin：音频播放、波形加载、播放器状态管理"""
import os

from PySide6.QtCore import QUrl, QTimer
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaDevices

from core.audio_tools import get_audio_info


class PlaybackMixin:
    """音频播放控制、波形加载、播放器状态管理"""

    def _setup_player_connections(self):
        """连接媒体播放器信号"""
        self.player.positionChanged.connect(self._on_player_position)
        self.player.durationChanged.connect(self._on_player_duration)
        self.player.mediaStatusChanged.connect(self._on_player_status)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        self.waveform_preview.seeked.connect(self._on_waveform_seek)
        self.waveform_preview.released.connect(self._on_waveform_released)

    @staticmethod
    def _check_file_duration(file_path: str, start_ms: int, end_ms: int) -> tuple:
        """检查 seek 时间是否在文件时长内,返回 (ok: bool, capped_end_ms: int, file_dur_ms: int)"""
        try:
            _dur = get_audio_info(file_path).duration_ms
            if start_ms >= _dur:
                return (False, end_ms, _dur)
            if end_ms > _dur:
                return (True, _dur, _dur)
            return (True, end_ms, _dur)
        except Exception:
            return (True, end_ms, 0)

    def _check_and_rebuild_audio_output(self):
        """播放前检查系统默认输出设备,若已变更则重建 QAudioOutput"""
        current_default = QMediaDevices.defaultAudioOutput()
        if self.audio_output.device() != current_default:
            old_vol = self.audio_output.volume()
            self.player.stop()
            self.audio_output = QAudioOutput()
            self.audio_output.setVolume(old_vol)
            self.player.setAudioOutput(self.audio_output)
            self.log("系统默认输出设备已变更,已重建 QAudioOutput")

    def _play_audio(self, file_path: str, start_ms: int = 0, end_ms: int = 0,
                    label: str = "", row: int = None):
        """统一音频播放入口

        - start_ms=0 从头播放,end_ms=0 播放到文件结束
        - 区间停止由 positionChanged 回调检查 _play_end_ms 实现
        - 切源时 setSource 后立即 play(),避免依赖 LoadedMedia 信号
          （Windows Media Foundation 后端从 EndOfMedia/LoadedMedia 切源时
           常丢失 LoadedMedia 信号,导致首次点击无声)
        - seek 必须在媒体加载后执行,用 _pending_seek + LoadedMedia 回调
        """
        if not file_path or not os.path.exists(file_path):
            if label and row is not None:
                self.log(f"【警告】第 {row+1} 行 {label}音频不可用")
            return

        # 播放前检查系统默认输出设备是否变更
        self._check_and_rebuild_audio_output()

        if start_ms > 0:
            _ok, end_ms, _dur = self._check_file_duration(file_path, start_ms, end_ms)
            if not _ok:
                _msg = f"{label} " if label else ""
                self.log(f"⚠️ {_msg}播放起始 {self._fmt_time(start_ms)} 超出文件时长 {self._fmt_time(_dur)}")
                return

        # 区间终点（positionChanged 回调中检查 _play_end_ms)
        self._play_end_ms = end_ms if end_ms > 0 else 0
        self._last_play_end = end_ms  # 供波形点击恢复边界检测

        target_url = QUrl.fromLocalFile(os.path.abspath(file_path))
        same_source = self.player.source() == target_url
        if not same_source:
            # 切源：setSource 异步加载,立即 play() 让 Qt 排队起播（不依赖
            # LoadedMedia 信号,避免状态机丢信号导致首次无声)
            self.player.setSource(target_url)
            self.player.play()
            # seek 必须等加载完成,LoadedMedia 回调中执行
            self._pending_seek = start_ms if start_ms > 0 else None
        else:
            # 同源已加载：清除异步残留,直接 seek + play
            self._pending_seek = None
            if start_ms > 0:
                self.player.setPosition(start_ms)
            self.player.play()

        # 日志 + 波形预览
        self.waveform_preview.setEnabled(True)
        if label and row is not None:
            dur_s = (end_ms - start_ms) / 1000 if end_ms > start_ms else 0
            self.log(f"▶ {label}试听: 第{row+1}行 {self._fmt_time(start_ms)} ({dur_s:.1f}s)")
            self._load_waveform_preview(file_path, start_ms, end_ms if end_ms > 0 else None)
        else:
            self._load_waveform_preview(file_path)

    def _play_cached_audio(self, row: int, label: str, start_ms_offset: int = 0, mixed: bool = False):
        """通用缓存音频播放"""
        idx = self._row_to_idx.get(row)
        if not idx:
            return
        cm = self._get_cache()
        if not cm:
            self.log("【警告】请先选择视频文件")
            return
        if idx > len(self.current_subtitles):
            return
        sub = self.current_subtitles[idx - 1]
        if sub is None:
            return
        path = cm.mixed_path(sub) if mixed else cm.tts_path(sub)
        if not os.path.exists(path):
            self.log(f"【警告】第 {row+1} 行 {label}不可用: {os.path.basename(path)}")
            return
        self._play_audio(path, start_ms=start_ms_offset, label=label, row=row)

    def _play_tts_segment(self, row: int):
        """播放指定行的 TTS 混合音频（跳过前置100ms扩展背景)"""
        self._play_cached_audio(row, "TTS", start_ms_offset=100, mixed=True)

    def _play_fullfile_audio(self, row: int, path: str, label: str):
        """通用全长文件播放：获取路径 → seek 区间播放"""
        if not path or not os.path.exists(path):
            self.log(f"【警告】第 {row+1} 行{label}文件不可用")
            return
        seg = self._subtitle_model.get_times(row)
        if seg == (0, 0):
            self.log(f"【警告】第 {row+1} 行字幕时间区间不可用")
            return
        start_ms, end_ms = seg
        self._play_audio(path, start_ms=start_ms, end_ms=end_ms, label=label, row=row)

    def _play_orig_audio(self, row: int):
        cm = self._get_cache()
        self._play_fullfile_audio(row, cm.vocals_path if cm else "", "人声")

    def _play_mix_audio(self, row: int):
        cm = self._get_cache()
        self._play_fullfile_audio(row, cm.mix_orig_path if cm else "", "原声")

    def _play_raw_tts(self, row: int):
        """播放指定行的原始TTS（混音前)"""
        self._play_cached_audio(row, "原始TTS")

    def _play_mixed(self):
        """播放混音后的音频"""
        cm = self._get_cache()
        path = cm.final_mix_path if cm else ""
        if path and os.path.exists(path):
            self._play_audio(path)
        else:
            self.log("【警告】混音音频不可用")

    def _play_src_segment(self, row: int):
        """播放 SRC 表中指定行的原人声区间（跟随视图模式: 原始/Qwen校准)"""
        if not hasattr(self, '_src_subs') or row >= len(self._src_subs):
            self.log(f"【警告】第 {row+1} 行 SRC 区间数据不可用")
            return
        sub = self._src_subs[row]
        if getattr(self, '_src_view_mode', 0) == 1:
            start_ms, end_ms = sub.eff_start_ms, sub.eff_end_ms
        else:
            start_ms, end_ms = sub.start_ms, sub.end_ms
        cm = self._get_cache()
        vocals_path = cm.vocals_path if cm else ""
        if not vocals_path or not os.path.exists(vocals_path):
            self.log("【警告】人声文件不可用")
            return
        self._play_audio(vocals_path, start_ms=start_ms, end_ms=end_ms, label="SRC 人声", row=row)

    def _play_ref_file(self, gender: str):
        """试听固定提示音"""
        path = self.cfg.fixed_ref_audio_male if gender == "male" else self.cfg.fixed_ref_audio_female
        if not path or not os.path.exists(path):
            self.log(f"【警告】{'男' if gender == 'male' else '女'}声提示音文件不存在")
            return
        label = "男声提示音" if gender == "male" else "女声提示音"
        self._play_audio(path, label=label)

    def _load_waveform_preview(self, audio_path: str, start_ms: int = None, end_ms: int = None):
        """加载音频文件并生成波形图数据（异步,不阻塞 UI)"""
        if not audio_path or not os.path.exists(audio_path):
            self.waveform_preview.set_waveform([], 1)
            return

        import threading
        _gen = self._wave_gen + 1
        self._wave_gen = _gen
        if self._wave_cancel is not None:
            self._wave_cancel.set()  # 取消上一次加载
        _cancel_event = threading.Event()
        self._wave_cancel = _cancel_event

        def _worker():
            """后台线程：分块读取音频并计算波形数据"""
            import soundfile as sf
            try:
                with sf.SoundFile(audio_path) as f:
                    sr = f.samplerate
                    total_frames = f.frames
                    if start_ms is not None and end_ms is not None:
                        _seek_pos = int(start_ms / 1000 * sr)
                        _read_frames = int((end_ms - start_ms) / 1000 * sr)
                        if _seek_pos >= total_frames:
                            return None
                        f.seek(_seek_pos)
                        total_frames = min(_read_frames, max(1, total_frames - _seek_pos))
                        total_ms = end_ms - start_ms
                    else:
                        total_ms = int(total_frames / sr * 1000)

                    target_n = 200

                    # 逐块读取完整波形
                    chunk_size = sr * 2
                    peak = 0.0
                    frames_read = 0
                    data = []
                    frames_per_bar = max(1, total_frames // target_n)
                    bar_buf = []

                    while frames_read < total_frames:
                        if _cancel_event.is_set():
                            return None
                        n = min(chunk_size, total_frames - frames_read)
                        chunk = f.read(n)
                        if chunk.ndim > 1:
                            chunk = chunk[:, 0]
                        frames_read += len(chunk)
                        for v in chunk:
                            bar_buf.append(abs(float(v)))
                            if len(bar_buf) >= frames_per_bar:
                                bar_max = max(bar_buf)
                                if bar_max > peak:
                                    peak = bar_max
                                data.append(bar_max)
                                bar_buf.clear()
                        import time as _wt
                        _wt.sleep(0)

                    if bar_buf:
                        bar_max = max(bar_buf)
                        if bar_max > peak:
                            peak = bar_max
                        data.append(bar_max)

                    if _cancel_event.is_set():
                        return None

                    if peak > 0:
                        data = [v / peak for v in data]

                    y_mono = None
                    f.seek(0 if start_ms is None else int(start_ms / 1000 * sr))
                    y_mono = f.read(total_frames)
                    if y_mono is not None and y_mono.ndim > 1:
                        y_mono = y_mono[:, 0]

                    return (data, total_ms, y_mono, sr)
            except Exception as e:
                if not _cancel_event.is_set():
                    print(f"【波形加载失败】{e}")
                return None

        _result_ref = [None]

        def _thread_entry():
            _result_ref[0] = _worker()
            self._wave_done_signal.emit(_gen)

        self._wave_result_ref = _result_ref
        self._wave_audio_path = audio_path
        self._wave_start_ms = start_ms
        t = threading.Thread(target=_thread_entry, daemon=True)
        t.start()

    def _on_wave_done(self, sig_gen: int):
        """波形加载完成回调（由 _wave_done_signal QueuedConnection 触发)"""
        if sig_gen != self._wave_gen:
            return
        ref = self._wave_result_ref
        result = ref[0] if ref else None
        if result is None:
            self.waveform_preview.set_waveform([], 1)
            return
        data, total_ms, y_mono, sr = result
        audio_path = self._wave_audio_path
        start_ms = self._wave_start_ms or 0
        if audio_path:
            self.waveform_preview._raw_file = audio_path
        if y_mono is not None and audio_path:
            self.waveform_preview.set_raw_audio(audio_path, y_mono, sr)
        self.waveform_preview.set_waveform(data, total_ms)
        self._waveform_offset = start_ms or 0
        self._display_duration = total_ms
        self.waveform_preview.set_segments([])

    def _stop_playback(self):
        """停止播放并重置 UI"""
        self.player.stop()
        self._play_end_ms = 0
        self.btn_stop_mix.setEnabled(False)
        self.lbl_play_time.setText("00:00 / 00:00")
        self.waveform_preview.set_position(0)

    def _on_player_position(self, pos_ms: int):
        """播放位置更新,到达终点时自动停止"""
        if self._play_end_ms > 0 and pos_ms >= self._play_end_ms:
            # 推迟到下一事件循环,避免在 positionChanged 回调内同步 stop 导致 MediaFoundation 重入
            QTimer.singleShot(0, self._stop_playback)
            return
        dur = self.player.duration()
        display_ms = self._display_duration or dur
        self.waveform_preview.set_position(pos_ms - self._waveform_offset)
        cur_ms = max(0, pos_ms - self._waveform_offset)
        self.lbl_play_time.setText(
            f"{cur_ms//60000:02d}:{(cur_ms//1000)%60:02d}.{cur_ms%1000:03d}"
            f" / {display_ms//60000:02d}:{(display_ms//1000)%60:02d}.{display_ms%1000:03d}"
        )

    def _on_player_duration(self, dur_ms: int):
        """音频总时长变化"""
        self.waveform_preview.setEnabled(dur_ms > 0)

    def _on_player_status(self, status):
        """播放器状态变化"""
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self.btn_stop_mix.setEnabled(True)
            # 媒体加载完成才执行 seek（切源时 play() 已调用)
            _seek = self._pending_seek
            if _seek is not None:
                self._pending_seek = None
                self.player.setPosition(_seek)
        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._stop_playback()
        elif status in (QMediaPlayer.MediaStatus.NoMedia,
                        QMediaPlayer.MediaStatus.InvalidMedia):
            self.btn_stop_mix.setEnabled(False)

    def _on_volume_changed(self, val: int):
        """音量滑块"""
        self.audio_output.setVolume(val / 100.0)

    def _on_waveform_seek(self, ms: int):
        """波形图拖动跳转（只 seek,不播放)"""
        if not self.player.source().isValid():
            return
        self.player.setPosition(ms + self._waveform_offset)
        cur_ms = max(0, ms)
        dur = self.player.duration()
        display_ms = self._display_duration or dur
        self.lbl_play_time.setText(
            f"{cur_ms//60000:02d}:{(cur_ms//1000)%60:02d}.{cur_ms%1000:03d}"
            f" / {display_ms//60000:02d}:{(display_ms//1000)%60:02d}.{display_ms%1000:03d}"
        )

    def _on_waveform_released(self, ms: int):
        """波形图松开鼠标 → seek + 播放/暂停切换"""
        if not self.player.source().isValid():
            return
        _pos = ms + self._waveform_offset
        if self.player.isPlaying():
            self.player.setPosition(_pos)
            self.player.pause()
            return
        self.player.setPosition(_pos)
        self.player.play()
        if self._last_play_end > 0 and _pos < self._last_play_end:
            self._play_end_ms = self._last_play_end

    def _show_waveform(self, row: int):
        """弹出音波对比对话框（委托给 waveform_dialog.py)"""
        from ui.waveform_dialog import show_waveform_dialog
        cm = self._get_cache()
        if not cm:
            self.log("【警告】请先选择视频文件")
            return
        show_waveform_dialog(self, cm, self._row_to_idx, self._subtitle_model, self.log, row)
