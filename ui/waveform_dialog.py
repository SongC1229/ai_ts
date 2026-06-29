"""波形对比对话框：弹出最多4个音频波形的可视化对比窗口"""
import os
import subprocess

from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QPushButton
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput


def show_waveform_dialog(parent, cm, row_to_idx, subtitle_model, log_fn, row: int):
    """弹出音波对比对话框（最多4个音频波形)

    Args:
        parent: 父窗口
        cm: CacheManager 实例
        row_to_idx: dict, 表格行号 → 字幕1-based索引
        subtitle_model: SubtitleTableModel
        log_fn: 日志回调 log(msg)
        row: 当前点击的表格行号
    """
    idx = row_to_idx.get(row)
    if idx is None:
        log_fn("【警告】无对应字幕索引")
        return

    # 获取该行字幕的时间区间
    sub_times = subtitle_model.get_times(row)
    seg_start_ms, seg_end_ms = sub_times

    # 收集4个音频源（按表头列顺序：试听 → 原声 → tts → 人声)
    audio_sources = []
    # 1. 试听（TTS合成混音,跳过前置100ms扩展)
    sub = subtitle_model.get_subtitle(row)
    if sub:
        tts_mixed = cm.mixed_path(sub)
    else:
        tts_mixed = None
    if tts_mixed and os.path.exists(tts_mixed):
        import soundfile as _sf
        with _sf.SoundFile(tts_mixed) as _fh:
            _tts_dur = int(len(_fh) / _fh.samplerate * 1000)
        audio_sources.append(("▶ 试听", "TTS合成 + 背景混音（跳过前置扩展)", (tts_mixed, 100, _tts_dur), (76, 175, 80)))
    # 2. 原声（原人声+背景混合,从全长音频seek)
    mix_full = cm.mix_orig_path
    if seg_end_ms > seg_start_ms and os.path.exists(mix_full):
        audio_sources.append(("🎶 原声", "原始人声 + 原始背景音（全长seek)", (mix_full, seg_start_ms, seg_end_ms), (156, 39, 176)))
    # 3. tts（纯TTS）
    raw_path = cm.tts_path(sub) if sub else None
    if os.path.exists(raw_path):
        audio_sources.append(("✨ tts", "AI合成语音（混音前)", raw_path, (255, 160, 0)))
    # 4. 人声（分离后人声参考,从全长音频seek)
    vocals_full = cm.vocals_path
    if seg_end_ms > seg_start_ms and os.path.exists(vocals_full):
        audio_sources.append(("🎵 人声", "原始人声（Demucs分离后,全长seek)", (vocals_full, seg_start_ms, seg_end_ms), (33, 150, 243)))

    if not audio_sources:
        log_fn("【警告】无可用音频")
        return

    from core.utils import read_wav_segment, downsample_waveform
    import soundfile as sf

    def load_wav(src):
        """加载音频,支持 (path, start_ms, end_ms) 元组以 seek 读取片段"""
        if isinstance(src, tuple):
            path, s_ms, e_ms = src
            if not path or not os.path.exists(path):
                return None, 0
            y, _ = read_wav_segment(path, s_ms, e_ms)
            if y is None:
                log_fn(f"⚠️ 波形加载跳过: {os.path.basename(path)} 起始时间 {s_ms}ms 超出文件时长")
            return y
        else:
            if not src or not os.path.exists(src):
                return None
            import soundfile as sf
            y, sr = sf.read(src)
            if y.ndim > 1:
                y = y[:, 0]
            return y

    durs = []
    for _, _, src, _ in audio_sources:
        if isinstance(src, tuple):
            _, s, e = src
            durs.append(e - s)
        else:
            durs.append(int(sf.info(src).duration * 1000))
    max_dur = max(durs) if durs else 1

    dlg = QDialog(parent)
    dlg.setWindowTitle("音波对比")
    dlg.setMinimumSize(1050, max(500, 225 * len(audio_sources)))
    lay = QVBoxLayout(dlg)

    player = QMediaPlayer()
    ao = QAudioOutput()
    ao.setVolume(1.0)
    player.setAudioOutput(ao)
    # 播放完毕时回到起始位置并暂停
    def _on_media_status(status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            ww = getattr(player, '_active_widget', None)
            if ww:
                player.setPosition(ww._play_start)
                player.pause()
    player.mediaStatusChanged.connect(_on_media_status)

    class WaveWidget(QWidget):
        def __init__(self, data, color, label, player, duration_ms, total_ms=1, play_start=0):
            super().__init__()
            self.data = data
            self.color = color
            self.label = label
            self.player = player
            self.dur = max(duration_ms, 1)
            self.total_ms = total_ms
            self.pos = 0
            self._play_start = play_start
            self._pressed = False
            self._active = False
            self.setMinimumHeight(80)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            player.positionChanged.connect(self._on_pos)
        def _on_pos(self, p):
            # 所有波形使用相同的 _active_play_start 参考点,播放时坐标同步移动
            ref = getattr(self.player, '_active_play_start', self._play_start)
            self.pos = p - ref
            # 当前活跃波形播放到区间末尾时自动暂停并复位
            if self._active and self.pos >= self.dur:
                self.player.pause()
                self.player.setPosition(self._play_start)
                self.pos = 0
            self.update()
        def _x_to_ms(self, x):
            return int(x / max(self.width(), 1) * self.total_ms)
        def mousePressEvent(self, e):
            self._pressed = True
            ms = self._x_to_ms(int(e.position().x()))
            self.player.setPosition(ms + self._play_start)
            self.update()
        def mouseMoveEvent(self, e):
            if self._pressed:
                ms = self._x_to_ms(int(e.position().x()))
                self.player.setPosition(ms + self._play_start)
                self.update()
        def mouseReleaseEvent(self, e):
            if self._pressed:
                self._pressed = False
                if hasattr(self.player, '_active_widget') and self.player._active_widget and self.player._active_widget != self:
                    self.player._active_widget._active = False
                self.player._active_widget = self
                self._active = True
                self.player._active_play_start = self._play_start
                ms = self._x_to_ms(int(e.position().x()))
                self.player.setPosition(ms + self._play_start)
                self.player.pause()
        def paintEvent(self, e):
            p = QPainter(self)
            w, h = self.width(), self.height()
            margin_bottom = 22
            wave_h = h - margin_bottom
            p.fillRect(0, 0, w, h, QColor(255, 255, 255))
            mid = wave_h // 2
            amp = mid * 0.75
            n = max(len(self.data), 1)
            step = w / n
            for i in range(len(self.data)):
                bar = int(self.data[i] * amp)
                x = int(i * step)
                bw = max(2, int(step * 0.7))
                if self.pos * n / self.total_ms > i:
                    clr = QColor(*self.color)
                else:
                    clr = QColor(180, 180, 180)
                p.fillRect(x, mid - bar, bw, bar * 2, clr)
            # 红色位置指示线
            px = int(self.pos * w / self.total_ms)
            line_clr = QColor(255, 0, 0) if self._active else QColor(200, 200, 200)
            p.setPen(QPen(line_clr, 2))
            p.drawLine(px, 0, px, wave_h)
            # 时间刻度（自适应精度)
            total_s = self.total_ms / 1000.0
            num_ticks = min(10, max(4, w // 80))
            if total_s < num_ticks * 0.5:
                num_ticks = max(2, int(total_s / 0.5))
            p.setPen(QPen(QColor(150, 150, 150), 1))
            for t in range(num_ticks + 1):
                sec = t * total_s / num_ticks
                tx = int(t * w / num_ticks)
                p.drawLine(tx, wave_h, tx, wave_h + 5)
                p.setPen(QColor(0, 0, 0))
                if total_s < 10:
                    label = f'{sec:.1f}s'
                elif sec < 60:
                    label = f'{sec:.0f}s'
                elif sec < 3600:
                    label = f'{int(sec//60)}m{sec%60:.0f}s'
                else:
                    label = f'{int(sec//3600)}h{int((sec%3600)//60)}m'
                p.drawText(max(2, min(w - 30, tx - 15)), wave_h + 16, label)
                p.setPen(QPen(QColor(150, 150, 150), 1))
            # 波形图内部不绘制标签（标题在外部 QLabel 中)

    def make_row(label, desc, wav_src, color):
        if isinstance(wav_src, tuple):
            path, s_ms, e_ms = wav_src
            dur = e_ms - s_ms
            play_path = path
            play_start = s_ms
        else:
            dur = int(sf.info(wav_src).duration * 1000) if os.path.exists(wav_src) else 1000
            play_path = wav_src
            play_start = 0
        data = downsample_waveform(load_wav(wav_src), dur, 200, max_dur)
        row_w = QWidget()
        row_lay = QVBoxLayout(row_w)
        row_lay.setContentsMargins(0, 0, 0, 0)
        hdr = QHBoxLayout()
        title = QLabel(f"<b>{label}</b>")
        title.setToolTip(desc)
        hdr.addWidget(title)
        desc_label = QLabel(f"<span style='color:gray;'>{desc}</span>")
        hdr.addWidget(desc_label)
        btn = QPushButton("▶")
        btn.setFixedSize(28, 20)
        def _play_wave(p, s, ww):
            if hasattr(player, '_active_widget') and player._active_widget and player._active_widget != ww:
                player._active_widget._active = False
            if ww:
                player._active_widget = ww
                ww._active = True
            player._active_play_start = s  # 全局 seek 参考点,四个波形共用
            target_url = QUrl.fromLocalFile(os.path.abspath(p))
            same_source = player.source() == target_url
            if s > 0:
                player.stop()
                if same_source:
                    # 同一文件已加载 — 直接 seek,不等待 LoadedMedia
                    player.setPosition(s)
                    player.play()
                else:
                    # 不同文件 — 等待加载完成后再 seek
                    def _on_loaded(status):
                        if status == QMediaPlayer.MediaStatus.LoadedMedia:
                            player.mediaStatusChanged.disconnect(_on_loaded)
                            player.setPosition(s)
                            player.play()
                    player.mediaStatusChanged.connect(_on_loaded)
                    player.setSource(target_url)
            else:
                player.stop()
                if not same_source:
                    # 不同文件 — 等待加载完成后再播放
                    def _on_loaded_zero(status):
                        if status == QMediaPlayer.MediaStatus.LoadedMedia:
                            player.mediaStatusChanged.disconnect(_on_loaded_zero)
                            player.play()
                    player.mediaStatusChanged.connect(_on_loaded_zero)
                    player.setSource(target_url)
                else:
                    player.play()
        _ww = None  # will be set after WaveWidget creation
        hdr.addWidget(btn)
        btn2 = QPushButton("⏹")
        btn2.setFixedSize(28, 20)
        btn2.clicked.connect(player.stop)
        hdr.addWidget(btn2)
        btn3 = QPushButton("📂")
        btn3.setFixedSize(28, 20)
        btn3.setToolTip("打开文件位置")
        btn3.clicked.connect(lambda checked=False, p=play_path: subprocess.Popen(['explorer', '/select,', os.path.abspath(p)]))
        hdr.addWidget(btn3)
        hdr.addStretch()
        row_lay.addLayout(hdr)
        _ww = WaveWidget(data, color, label, player, dur, max_dur, play_start)
        row_lay.addWidget(_ww)
        # 关联播放按钮与 WaveWidget
        btn.clicked.connect(lambda checked=False, p=play_path, s=play_start, ww=_ww: _play_wave(p, s, ww))
        return row_w

    for label, desc, src, color in audio_sources:
        lay.addWidget(make_row(label, desc, src, color))

    dlg.exec()
    player.stop()
