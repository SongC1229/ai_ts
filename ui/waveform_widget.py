"""波形预览 Widget"""

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics, QColor, QPainter, QPen


from core.utils import fmt_time_adaptive


class WaveformPreviewWidget(QWidget):
    """混音预览波形图：显示音频波形 + 语音片段区间 + 播放位置指示线,支持拖动 seek 与缩放"""
    seeked = Signal(int)  # emit seek position in ms
    released = Signal(int)  # emit final position on mouse release
    precise_reload_needed = Signal(str, int, int)  # (file_path, start_ms, end_ms) 需要加载精确波形

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = []           # downsampled waveform data (0.0~1.0) for current view
        self._segments = []       # [(start_ms, end_ms), ...] 语音片段区间
        self._total_duration = 1  # ms, 音频全长
        self._view_start = 0      # ms, 可见范围起点
        self._view_end = 1        # ms, 可见范围终点
        self._pos = 0             # current playback position ms (相对音频全长)
        self._pressed = False
        self._drag_start_x = 0    # drag 起始 x 坐标
        self._drag_start_view = (0, 1)  # drag 起始时的 view 范围
        self._drag_mode = "seek"
        # 原始音频数据（缩放时重新降采样)
        self._raw_y = None        # numpy array, 原始音频
        self._raw_sr = 0          # 原始采样率
        self._raw_file = None     # 原始文件路径
        self._preserve_view = False  # True 时 set_waveform 不重置 view
        self.setMinimumHeight(120)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    # ── 公共接口 ─────────────────────────────────────────

    def set_waveform(self, data: list, total_duration_ms: int):
        """Set waveform data and total duration (full view)"""
        self._data = data
        self._total_duration = max(total_duration_ms, 1)
        if not self._preserve_view:
            self._view_start = 0
            self._view_end = self._total_duration
        self._preserve_view = False
        self.update()

    def set_raw_audio(self, file_path: str, y, sr: int):
        """存储原始音频数据,供缩放时重新降采样"""
        self._raw_file = file_path
        self._raw_y = y
        self._raw_sr = sr

    def set_segments(self, segments: list):
        """设置语音片段区间"""
        self._segments = segments
        self.update()

    def set_position(self, pos_ms: int):
        """Update playback position (相对音频全长)"""
        self._pos = pos_ms
        self.update()

    def view_range(self):
        """返回当前可见范围 (start_ms, end_ms)"""
        return self._view_start, self._view_end

    def zoom_in(self, center_ms: int = None):
        """以 center_ms 为中心放大 1.5 倍"""
        if center_ms is None:
            center_ms = (self._view_start + self._view_end) // 2
        half_range = (self._view_end - self._view_start) / 3.0  # 缩小到 1/1.5
        if half_range < 50:  # 最小 100ms 范围
            return
        self._set_view(center_ms - half_range, center_ms + half_range)

    def zoom_out(self, center_ms: int = None):
        """以 center_ms 为中心缩小 1.5 倍"""
        if self._view_start <= 0 and self._view_end >= self._total_duration:
            return  # 已经全览
        if center_ms is None:
            center_ms = (self._view_start + self._view_end) // 2
        half_range = (self._view_end - self._view_start) * 1.5 / 2.0
        self._set_view(center_ms - half_range, center_ms + half_range)

    def zoom_fit(self):
        """重置为全览"""
        self._set_view(0, self._total_duration)

    # ── 内部 ─────────────────────────────────────────────

    def _set_view(self, start_ms: float, end_ms: float):
        """设置可见范围并重新生成波形数据"""
        start_ms = max(0, int(start_ms))
        end_ms = min(self._total_duration, int(end_ms))
        if end_ms - start_ms < 100:
            return
        # 固定可见范围宽度（缩放时不变,平移时保持)
        # 只有从 zoom_in/zoom_out 调用时才改变宽度
        self._view_start = start_ms
        self._view_end = end_ms
        # 缩放到 <=3min 且无原始音频时,请求加载精确波形
        if self._raw_y is None and self._raw_file and (end_ms - start_ms) <= 180_000:
            self.precise_reload_needed.emit(self._raw_file, start_ms, end_ms)
        self._rebuild_view_data()
        self.update()

    def _rebuild_view_data(self):
        """从原始音频重新降采样当前可见范围"""
        if self._raw_y is None or self._raw_sr == 0:
            return
        import numpy as np
        sr = self._raw_sr
        view_len_ms = self._view_end - self._view_start
        if view_len_ms <= 0:
            return
        start_sample = int(self._view_start / 1000 * sr)
        end_sample = int(self._view_end / 1000 * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(self._raw_y), end_sample)
        if end_sample - start_sample < 2:
            return
        chunk = self._raw_y[start_sample:end_sample]
        # 根据可见宽度决定柱子数量（~每 3px 一个柱子)
        w = self.width()
        target_n = max(20, w // 3) if w > 0 else 200
        step = max(1, len(chunk) // target_n)
        data = []
        for i in range(0, len(chunk), step):
            sub = chunk[i:i + step]
            data.append(float(np.max(np.abs(sub))) if len(sub) > 0 else 0.0)
        peak = max(data) if data else 1.0
        if peak > 0:
            data = [v / peak for v in data]
        self._data = data



    def _x_to_ms(self, x: int) -> int:
        """将 widget 内 x 坐标转为绝对时间 ms（考虑 view 范围)"""
        w = self.width()
        if w <= 0:
            return 0
        view_range = self._view_end - self._view_start
        ratio = max(0.0, min(1.0, x / w))
        return int(self._view_start + ratio * view_range)

    def mousePressEvent(self, e):
        self._pressed = True
        self._drag_start_x = int(e.position().x())
        self._drag_start_view = (self._view_start, self._view_end)
        self._drag_mode = "seek"  # 默认 seek 模式
        ms = self._x_to_ms(self._drag_start_x)
        self._pos = ms
        self.seeked.emit(ms)
        self.update()

    def mouseMoveEvent(self, e):
        if not self._pressed:
            return
        dx = int(e.position().x()) - self._drag_start_x
        view_range = self._view_end - self._view_start
        is_zoomed = view_range < self._total_duration * 0.99
        if is_zoomed and abs(dx) > 3:
            # 缩放状态下拖动 = 平移视野
            self._drag_mode = "pan"
            w = self.width()
            if w <= 0:
                return
            shift_ms = -int(dx * view_range / w)
            new_start = self._drag_start_view[0] + shift_ms
            new_end = self._drag_start_view[1] + shift_ms
            if new_start < 0:
                new_start = 0
                new_end = view_range
            if new_end > self._total_duration:
                new_end = self._total_duration
                new_start = self._total_duration - view_range
            self._set_view(new_start, new_end)
        else:
            # 全览状态下拖动 = 更新光标位置并实时显示时间
            self._drag_mode = "seek"
            ms = self._x_to_ms(int(e.position().x()))
            self._pos = ms
            self.seeked.emit(ms)
            self.update()

    def mouseReleaseEvent(self, e):
        self._pressed = False
        if self._drag_mode == "seek":
            self.released.emit(self._pos)

    def wheelEvent(self, e):
        """滚轮缩放：以鼠标位置为中心"""
        center_ms = self._x_to_ms(int(e.position().x()))
        delta = e.angleDelta().y()
        if delta > 0:
            self.zoom_in(center_ms)
        else:
            self.zoom_out(center_ms)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rebuild_view_data()
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        margin_top = 18       # 顶部绝对时间轴高度
        margin_bottom = 22    # 底部相对时间轴高度
        wave_h = h - margin_top - margin_bottom
        if wave_h < 10:
            return
        wave_y0 = margin_top  # 波形区域顶部 y 坐标
        mid = wave_y0 + wave_h // 2
        amp = (wave_h // 2) * 0.75
        view_range = self._view_end - self._view_start
        is_zoomed = view_range < self._total_duration * 0.99

        # 背景
        p.fillRect(0, 0, w, h, QColor(250, 250, 250))

        # ── 顶部时间轴（绝对时间,~5 个均匀刻度)──
        p.setPen(QPen(QColor(180, 180, 180), 1))
        r = QFontMetrics(p.font())
        num_top_ticks = 5  # 固定 5 个刻度
        for i in range(num_top_ticks + 1):
            t_ms = self._view_start + int(i * view_range / num_top_ticks)
            tx = int(i * w / num_top_ticks)
            # 刻度短线（朝下的)
            p.drawLine(tx, wave_y0 - 4, tx, wave_y0 - 1)
            # 时间标签
            label = fmt_time_adaptive(t_ms, self._total_duration)
            tw = r.horizontalAdvance(label)
            lx = max(2, min(w - tw - 2, tx - tw // 2))
            p.setPen(QColor(80, 80, 80))
            p.drawText(lx, wave_y0 - 6, label)
            p.setPen(QPen(QColor(180, 180, 180), 1))
        # 底部波形区域上边框
        p.drawLine(0, wave_y0 - 1, w, wave_y0 - 1)

        # 绘制语音片段区间（淡蓝色背景)
        if self._segments and self._total_duration > 1:
            seg_color = QColor(33, 150, 243, 40)
            p.fillRect(0, wave_y0, w, wave_h, QColor(245, 245, 245))
            p.setBrush(seg_color)
            p.setPen(Qt.PenStyle.NoPen)
            for start_ms, end_ms in self._segments:
                if end_ms <= self._view_start or start_ms >= self._view_end:
                    continue
                x1 = int(max(0, start_ms - self._view_start) / view_range * w)
                x2 = int(min(view_range, end_ms - self._view_start) / view_range * w)
                seg_w = max(2, x2 - x1)
                p.drawRect(max(0, x1), wave_y0, seg_w, wave_h)

        # 绘制波形柱状图
        n = max(len(self._data), 1)
        step_w = w / n
        for i in range(len(self._data)):
            bar = int(self._data[i] * amp)
            x = int(i * step_w)
            bw = max(1, int(step_w * 0.8))
            bar_center_ms = self._view_start + (i + 0.5) / n * view_range
            if self._pos > bar_center_ms:
                clr = QColor(33, 150, 243)
            else:
                clr = QColor(180, 180, 180)
            p.fillRect(x, mid - bar, bw, bar * 2, clr)

        # 播放位置指示线（红色)
        if view_range > 0:
            px = int((self._pos - self._view_start) * w / view_range)
            if 0 <= px <= w:
                p.setPen(QPen(QColor(255, 50, 50), 2))
                p.drawLine(px, wave_y0, px, wave_y0 + wave_h)

        # ── 底部时间轴（相对当前视图的时间)──
        p.setPen(QPen(QColor(150, 150, 150), 1))
        view_s = view_range / 1000.0
        num_ticks = min(10, max(4, w // 80))
        bottom_y = wave_y0 + wave_h
        for t in range(num_ticks + 1):
            sec = t * view_s / num_ticks
            tx = int(t * w / num_ticks)
            p.drawLine(tx, bottom_y, tx, bottom_y + 5)
            if view_s < 3:
                label = f'{sec:.2f}s'
            elif view_s < 10:
                label = f'{sec:.1f}s'
            elif sec < 60:
                label = f'{sec:.0f}s'
            elif sec < 3600:
                label = f'{int(sec//60)}m{sec%60:.0f}s'
            else:
                label = f'{int(sec//3600)}h{int((sec%3600)//60)}m'
            p.setPen(QColor(0, 0, 0))
            p.drawText(max(2, min(w - 30, tx - 15)), bottom_y + 16, label)
            p.setPen(QPen(QColor(150, 150, 150), 1))
        # 右下角缩放比例提示
        if is_zoomed:
            p.setPen(QColor(120, 120, 120))
            p.drawText(w - 50, bottom_y + 16, f'{view_s:.1f}s')

        # 无数据时提示
        if not self._data:
            p.setPen(QColor(180, 180, 180))
            p.drawText(w // 2 - 60, mid, "波形加载中…")

        p.end()
