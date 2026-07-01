"""字幕表格模型和委托 — Model-View 架构优化

使用 QTableView + QAbstractTableModel 替代 QTableWidget:
- 仅渲染可见行,大幅降低内存占用
- 通过 Delegate 绘制按钮列,避免创建数千个 QPushButton
- 通过 Model 的 data() 方法统一处理样式（颜色、字体、Tooltip)

包含:
- SubtitleTableModel: 右侧字幕表 (12列)
- SrcTableModel: 左侧原声字幕表 (5列)
- ButtonDelegate: 按钮列绘制委托
"""

from PySide6.QtCore import Qt, QAbstractTableModel, Signal, QModelIndex, QSize, QSortFilterProxyModel
from PySide6.QtGui import QColor, QFont, QPen, QPainter, QBrush
from PySide6.QtWidgets import QStyledItemDelegate, QStyle
from core.utils import fmt_time


# ══════════════════════════════════════════════════════════════════════════════
# 右侧字幕表 (subtitle_table) 定义
# ══════════════════════════════════════════════════════════════════════════════

SUBTITLE_COLUMNS = [
    ("序号", 44),
    ("开始: 点击试听", 109),
    ("结束", 109),
    ("性别", 64),
    ("字幕文本", -1),  # Stretch
    ("试听", 36),
    ("原声", 36),
    ("tts", 30),
    ("人声", 36),
    ("对比", 36),
    ("状态", 74),
    ("重试", 36),
]

# 列索引常量
COL_IDX = 0
COL_START = 1
COL_END = 2
COL_GENDER = 3
COL_TEXT = 4
COL_PLAY_TTS = 5
COL_PLAY_MIX = 6
COL_PLAY_RAW_TTS = 7
COL_PLAY_VOCAL = 8
COL_COMPARE = 9
COL_STATUS = 10
COL_REGEN = 11

# 按钮列集合
BUTTON_COLUMNS = {COL_PLAY_TTS, COL_PLAY_MIX, COL_PLAY_RAW_TTS, COL_PLAY_VOCAL, COL_COMPARE, COL_REGEN}

# 按钮文本和 Tooltip
BUTTON_TEXTS = {
    COL_PLAY_TTS: ("🌟", "播放 TTS 合成结果"),
    COL_PLAY_MIX: ("🎶", "播放原始混合音频片段"),
    COL_PLAY_RAW_TTS: ("✨", "播放纯 TTS（混音前)"),
    COL_PLAY_VOCAL: ("🎵", "播放原音频分离后的人声"),
    COL_COMPARE: ("🔍", "对比原声与 TTS 音波图"),
    COL_REGEN: ("🔄", "删除此条缓存并重新合成"),
}

# 状态映射
STATUS_MAP = {
    "pending": "等待中",
    "tts_done": "TTS 完成",
    "tts_synthesizing": "等待中",
    "skipped": "跳过",
    "mixed": "混音完成",
}

STATUS_COLORS = {
    "pending":          QColor(153, 153, 153),   #999999
    "tts_synthesizing": QColor(153, 153, 153),   #999999
    "tts_done":         QColor(76, 175, 80),      #4CAF50
    "skipped":          QColor(153, 153, 153),   #999999
    "mixed":            QColor(156, 39, 176),     #9C27B0
}

# 颜色常量
GREEN_COLOR = QColor(0, 160, 0)


class SubtitleTableModel(QAbstractTableModel):
    """右侧字幕表格数据模型

    数据源:
    - subtitles: 字幕列表 (SubtitleItem), gender/calib 存于字段
    - statuses: 状态字典 {row: status_str}
    - tts_ready: TTS 就绪集合 {row}
    - mix_ready: 混音就绪集合 {row}
    """

    # 按钮点击信号 (row, col)
    buttonClicked = Signal(int, int)
    # 性别点击信号 (row)
    genderClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._subtitles = []
        self._statuses = {}
        self._hide_text = False

    def set_data(self, subtitles):
        """设置字幕数据（批量更新)"""
        self.beginResetModel()
        self._subtitles = subtitles
        self._statuses.clear()

        for row, sub in enumerate(self._subtitles):
            self._statuses[row] = "pending"

        self.endResetModel()

    def update_times(self, row: int, start_ms: int, end_ms: int, changed: bool = False, idx: int = None):
        """更新单行校准时间"""
        if 0 <= row < len(self._subtitles):
            sub = self._subtitles[row]
            sub.calib_start_ms = start_ms
            sub.calib_end_ms = end_ms
            left = self.index(row, COL_START)
            right = self.index(row, COL_END)
            self.dataChanged.emit(left, right)
            if changed:
                idx_idx = self.index(row, COL_IDX)
                self.dataChanged.emit(idx_idx, idx_idx)


    def reset_all_calib(self):
        """批量清除所有校准（清缓存时用)"""
        for row in range(len(self._subtitles)):
            sub = self._subtitles[row]
            sub.calib_start_ms = 0
            sub.calib_end_ms = 0
        if self._subtitles:
            self.dataChanged.emit(self.index(0, 0),
                                  self.index(len(self._subtitles) - 1, self.columnCount() - 1))

    def get_original_times(self, row: int):
        """获取原始 SRT 时间"""
        if 0 <= row < len(self._subtitles):
            sub = self._subtitles[row]
            return (sub.start_ms, sub.end_ms)
        return None

    def update_gender(self, idx: int, gender: str):
        """更新性别"""
        row = idx - 1
        if 0 <= row < len(self._subtitles):
            self._subtitles[row].gender = gender
            self.dataChanged.emit(self.index(row, COL_GENDER), self.index(row, COL_GENDER))

    def set_status(self, row: int, status: str):
        """设置状态"""
        self._statuses[row] = status
        self.dataChanged.emit(self.index(row, COL_STATUS), self.index(row, COL_STATUS))


    # ── QAbstractTableModel 必须实现的方法 ──────────────

    def rowCount(self, parent=QModelIndex()):
        return len(self._subtitles)

    def columnCount(self, parent=QModelIndex()):
        return len(SUBTITLE_COLUMNS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()

        if row < 0 or row >= len(self._subtitles):
            return None

        sub = self._subtitles[row]
        idx = row + 1
        is_changed = sub.is_calibrated and (sub.calib_start_ms != sub.start_ms or sub.calib_end_ms != sub.end_ms)
        orig_start, orig_end = sub.start_ms, sub.end_ms
        cur_start, cur_end = sub.eff_start_ms, sub.eff_end_ms

        # ── 显示文本 ──
        if role == Qt.DisplayRole:
            if col == COL_IDX:
                return str(idx)
            elif col == COL_START:
                return fmt_time(cur_start)
            elif col == COL_END:
                return fmt_time(cur_end)
            elif col == COL_GENDER:
                return {"male": "男", "female": "女"}.get(sub.gender, "未定")
            elif col == COL_TEXT:
                return "·······" if self._hide_text else sub.text.replace('\n', ' ')
            elif col == COL_STATUS:
                return STATUS_MAP.get(self._statuses.get(row, "pending"), self._statuses.get(row, ""))
            elif col in BUTTON_COLUMNS:
                return BUTTON_TEXTS[col][0]

        # ── 前景色（绿色标记变化) ──
        elif role == Qt.ForegroundRole:
            if is_changed:
                if col == COL_IDX:
                    return GREEN_COLOR
                elif col == COL_START and cur_start != orig_start:
                    return GREEN_COLOR
                elif col == COL_END and cur_end != orig_end:
                    return GREEN_COLOR
            # 状态列颜色
            if col == COL_STATUS:
                _sk = self._statuses.get(row, "pending")
                if _sk in STATUS_COLORS:
                    return STATUS_COLORS[_sk]

        # ── 字体（粗体标记变化) ──
        elif role == Qt.FontRole:
            if is_changed:
                f = QFont()
                f.setBold(True)
                if col == COL_IDX:
                    return f
                elif col == COL_START and cur_start != orig_start:
                    return f
                elif col == COL_END and cur_end != orig_end:
                    return f

        # ── Tooltip ──
        elif role == Qt.ToolTipRole:
            if col == COL_TEXT:
                return sub.text
            elif col == COL_START and cur_start != orig_start:
                _diff = cur_start - orig_start
                _dir = f"向后{_diff}ms" if _diff > 0 else f"向前{-_diff}ms"
                return f"原始: {fmt_time(orig_start)} → {_dir}"
            elif col == COL_END and cur_end != orig_end:
                _diff = cur_end - orig_end
                _dir = f"向后{_diff}ms" if _diff > 0 else f"向前{-_diff}ms"
                return f"原始: {fmt_time(orig_end)} → {_dir}"
            elif col in BUTTON_COLUMNS:
                return BUTTON_TEXTS[col][1]

        # ── 文本对齐 ──
        elif role == Qt.TextAlignmentRole:
            if col in (COL_GENDER, COL_STATUS):
                return int(Qt.AlignCenter)
            if col in BUTTON_COLUMNS:
                return int(Qt.AlignCenter)

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return SUBTITLE_COLUMNS[section][0]
        return None

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def toggle_hide_text(self):
        """切换字幕文本列的隐藏/显示"""
        self._hide_text = not self._hide_text
        self.dataChanged.emit(self.index(0, COL_TEXT), self.index(len(self._subtitles) - 1, COL_TEXT))

    # ── 辅助方法 ──

    def get_subtitle(self, row: int):
        """获取指定行的字幕对象"""
        if 0 <= row < len(self._subtitles):
            return self._subtitles[row]
        return None

    def get_times(self, row: int):
        """获取显示时间（校准优先）"""
        if 0 <= row < len(self._subtitles):
            sub = self._subtitles[row]
            return (sub.eff_start_ms, sub.eff_end_ms)
        return (0, 0)

    def get_gender(self, idx: int) -> str:
        """获取性别"""
        row = idx - 1
        if 0 <= row < len(self._subtitles):
            return self._subtitles[row].gender
        return ""



# ══════════════════════════════════════════════════════════════════════════════
# 左侧原声字幕表 (src_table) 定义
# ══════════════════════════════════════════════════════════════════════════════

SRC_COLUMNS = [
    ("序号", 44),
    ("开始: 点击试听", 109),
    ("结束", 109),
    ("时长(s)", 70),
    ("文本", 100),
]

SRC_COL_IDX = 0
SRC_COL_START = 1
SRC_COL_END = 2
SRC_COL_DUR = 3
SRC_COL_TEXT = 4


class SrcTableModel(QAbstractTableModel):
    """左侧原声字幕表格数据模型

    数据源: List[SubtitleItem]
    - is_calibrated: False=显示原始时间, True=显示校准时间
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._subs = []
        self._is_calibrated = False

    def set_data(self, subs, is_calibrated=False):
        self.beginResetModel()
        self._subs = list(subs)
        self._is_calibrated = is_calibrated
        self.endResetModel()

    def clear_data(self):
        self.beginResetModel()
        self._subs.clear()
        self._is_calibrated = False
        self.endResetModel()

    def get_segments(self):
        return [(s.idx, s.eff_start_ms, s.eff_end_ms, s.text) for s in self._subs]

    # ── QAbstractTableModel ──────────────────────────

    def rowCount(self, parent=QModelIndex()):
        return len(self._subs)

    def columnCount(self, parent=QModelIndex()):
        return len(SRC_COLUMNS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        col = index.column()
        if row < 0 or row >= len(self._subs):
            return None

        sub = self._subs[row]
        is_calib = self._is_calibrated
        start_ms = sub.eff_start_ms if is_calib else sub.start_ms
        end_ms = sub.eff_end_ms if is_calib else sub.end_ms
        changed = is_calib and sub.is_calibrated and (sub.calib_start_ms != sub.start_ms or sub.calib_end_ms != sub.end_ms)

        if role == Qt.DisplayRole:
            if col == SRC_COL_IDX:
                return str(row + 1)
            elif col == SRC_COL_TEXT:
                return sub.text[:40]
            elif col == SRC_COL_DUR:
                return f"{(end_ms - start_ms) / 1000:.3f}"
            elif col == SRC_COL_END:
                return fmt_time(end_ms)
            elif col == SRC_COL_START:
                return fmt_time(start_ms)

        elif role == Qt.ForegroundRole:
            if changed:
                if col == SRC_COL_IDX:
                    return GREEN_COLOR
                elif col == SRC_COL_START and start_ms != sub.start_ms:
                    return GREEN_COLOR
                elif col == SRC_COL_END and end_ms != sub.end_ms:
                    return GREEN_COLOR

        elif role == Qt.FontRole:
            if changed:
                f = QFont()
                f.setBold(True)
                if col == SRC_COL_IDX:
                    return f
                elif col == SRC_COL_START and start_ms != sub.start_ms:
                    return f
                elif col == SRC_COL_END and end_ms != sub.end_ms:
                    return f

        elif role == Qt.ToolTipRole:
            if col == SRC_COL_TEXT:
                return sub.text
            if is_calib and changed:
                _src_name = "Qwen" if isinstance(is_calib, str) else "校准"
                if col == SRC_COL_START and start_ms != sub.start_ms:
                    _diff = start_ms - sub.start_ms
                    _dir = f"向后{_diff}ms" if _diff > 0 else f"向前{-_diff}ms"
                    return f"原声字幕 → {_src_name} {_dir}"
                if col == SRC_COL_END and end_ms != sub.end_ms:
                    _diff = end_ms - sub.end_ms
                    _dir = f"向后{_diff}ms" if _diff > 0 else f"向前{-_diff}ms"
                    return f"原声字幕 → {_src_name} {_dir}"
                    return f"原声字幕 → {_src_name} {_dir}"

        # ── 文本对齐 ──
        elif role == Qt.TextAlignmentRole:
            if col == SRC_COL_IDX:
                return int(Qt.AlignCenter)

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return SRC_COLUMNS[section][0]
        return None

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable


# ══════════════════════════════════════════════════════════════════════════════
# 按钮委托 (ButtonDelegate)
# ══════════════════════════════════════════════════════════════════════════════

class ButtonDelegate(QStyledItemDelegate):
    """按钮列委托 — 绘制按钮外观并处理点击事件"""

    # 按钮点击信号 (row, col)
    buttonClicked = Signal(int, int)

    def __init__(self, button_columns, parent=None):
        super().__init__(parent)
        self._button_columns = button_columns

    def paint(self, painter, option, index):
        """绘制按钮"""
        if index.column() not in self._button_columns:
            super().paint(painter, option, index)
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        enabled = True

        # 绘制按钮背景
        rect = option.rect.adjusted(3, 3, -3, -3)
        if rect.width() < 10 or rect.height() < 10:
            painter.restore()
            return

        # 按钮背景颜色
        if not enabled:
            bg_color = QColor(240, 240, 240)
            text_color = QColor(180, 180, 180)
        elif option.state & QStyle.State_MouseOver:
            bg_color = QColor(220, 235, 255)
            text_color = QColor(0, 0, 0)
        else:
            bg_color = QColor(255, 255, 255)
            text_color = QColor(0, 0, 0)

        # 绘制圆角矩形背景
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(rect, 3, 3)

        # 绘制文本
        text = index.data(Qt.DisplayRole)
        if text:
            painter.setPen(QPen(text_color))
            painter.drawText(rect, Qt.AlignCenter, text)

        painter.restore()

    def editorEvent(self, event, model, option, index):
        """处理鼠标事件"""
        if index.column() not in self._button_columns:
            return super().editorEvent(event, model, option, index)

        from PySide6.QtCore import QEvent
        if event.type() == QEvent.MouseButtonRelease:
            rect = option.rect.adjusted(3, 3, -3, -3)
            if rect.contains(event.pos()):
                self.buttonClicked.emit(index.row(), index.column())
                return True

        return super().editorEvent(event, model, option, index)

    def sizeHint(self, option, index):
        """返回按钮尺寸"""
        return QSize(28, 20)


class GenderFilterProxy(QSortFilterProxyModel):
    """按性别列（COL_GENDER)+ 状态列（COL_STATUS)过滤字幕行的代理模型

    支持“钉住”行：手动修改性别时把该行钉住,不受筛选影响,
    下次切换筛选时自动清除钉住。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filter_gender = ""  # ""=全部, "male", "female", "未检测"
        self._filter_status = ""  # ""=全部, 或 status key (pending/tts_done/mixed/...)
        self._pinned_rows: set[int] = set()
        self._pipeline_running = False  # pipeline 执行中自动钉住状态变化行

    def set_pipeline_running(self, running: bool):
        self._pipeline_running = running
        if not running:
            self._pinned_rows.clear()
            self.invalidateFilter()

    def dataChanged(self, topLeft, bottomRight, roles=None):
        """pipeline 执行时自动钉住状态变化的行,避免重筛后消失"""
        if self._pipeline_running and self._filter_status:
            top = topLeft.column()
            bottom = bottomRight.column()
            if top <= COL_STATUS <= bottom:
                for r in range(topLeft.row(), bottomRight.row() + 1):
                    self._pinned_rows.add(r)
        super().dataChanged(topLeft, bottomRight, roles)

    def pin_row(self, source_row: int):
        """钉住行,使其在当前筛选下始终可见"""
        self._pinned_rows.add(source_row)
        # 仅在存在筛选条件时才 invalidate,避免无筛选时列表跳动
        if self._filter_gender or self._filter_status:
            self.invalidateFilter()

    def set_filter_gender(self, gender: str):
        self._filter_gender = gender
        self._pinned_rows.clear()
        self.invalidateFilter()

    def set_filter_status(self, status: str):
        self._filter_status = status
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        if source_row in self._pinned_rows:
            return True
        # 性别筛选
        if self._filter_gender:
            g_idx = self.sourceModel().index(source_row, COL_GENDER, source_parent)
            g_data = g_idx.data(Qt.DisplayRole)
            if g_data != self._filter_gender:
                return False
        # 状态筛选
        if self._filter_status:
            # 直接比较内部状态键,避免经过 DisplayRole 映射
            s_data = self.sourceModel()._statuses.get(source_row, "pending")
            # tts_synthesizing 视为 pending（显示为"等待中")
            if s_data == "tts_synthesizing":
                s_data = "pending"
            if s_data != self._filter_status:
                return False
        return True
