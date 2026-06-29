"""通用配置面板 — 主界面第三行,左侧下拉选配置项,右侧动态显示控件"""

from dataclasses import dataclass, field
from typing import List, Optional

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QCheckBox, QLabel,
)


@dataclass
class _Item:
    """单个配置项定义

    range_min/range_max/step 仅对 int/float 有效;
    choices 仅对 choice 有效;string 类型三者都不用。
    """
    key: str
    label: str
    typ: str                                       # bool | int | float | choice | string
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    step: float = 0.1
    choices: List[str] = field(default_factory=list)


# 配置项定义
_ITEMS: List[_Item] = [
    # ── 本地引擎 ──
    _Item("tts_local_mode",             "本地引擎模式",      "choice", choices=["indextts", "dots"]),

    # ── Whisper 校准参数 ──
    _Item("whisper_vad_filter",         "Whisper VAD",       "bool"),
    _Item("whisper_vad_threshold",      "Whisper VAD阈值",   "float", 0.0, 1.0, 0.05),
    _Item("whisper_vad_min_silence_ms", "VAD最小静音",       "int",   100, 3000),
    _Item("whisper_vad_speech_pad_ms",  "VAD语音补白",       "int",   0, 1000),
    _Item("whisper_beam_size",          "Whisper波束宽度",   "int",   1, 20),

    # ── 性别检测 ──
    _Item("gender_detect_mode",         "性别检测模式",      "choice", choices=["wavlm", "gender_cls"]),
]


class ConfigPanel(QWidget):
    """通用配置面板 — 下拉选配置项 + 动态控件"""

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.key_combo = QComboBox()
        self.key_combo.setMinimumWidth(150)
        self.key_combo.currentIndexChanged.connect(self._on_key_changed)
        layout.addWidget(self.key_combo)

        self.value_widget = QWidget()
        self._value_layout = QHBoxLayout(self.value_widget)
        self._value_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.value_widget, 1)

        self._current_widget = None
        self._populate()

    def _populate(self):
        """填充下拉列表"""
        for item in _ITEMS:
            self.key_combo.addItem(item.label, item.key)

    def _on_key_changed(self, idx):
        """选中项变化 → 重建右侧控件"""
        # 清除旧控件
        while self._value_layout.count():
            w = self._value_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        self._current_widget = None

        if idx < 0 or idx >= len(_ITEMS):
            return

        item = _ITEMS[idx]
        key = item.key
        current = getattr(self._cfg, key)

        if item.typ == "bool":
            w = QCheckBox()
            w.setChecked(bool(current))
            w.setText("开启" if bool(current) else "关闭")
            w.toggled.connect(lambda c, k=key: self._on_changed_bool(k, c))
            self._value_layout.addWidget(w)
            self._current_widget = w

        elif item.typ == "int":
            w = QSpinBox()
            w.setRange(item.range_min, item.range_max)
            w.setValue(int(current))
            w.valueChanged.connect(lambda v, k=key: self._emit_changed(k, v))
            self._value_layout.addWidget(w)
            self._value_layout.addWidget(QLabel(f"({item.range_min}~{item.range_max})"))
            self._current_widget = w

        elif item.typ == "float":
            w = QDoubleSpinBox()
            w.setRange(item.range_min, item.range_max)
            w.setSingleStep(item.step)
            w.setValue(float(current))
            w.valueChanged.connect(lambda v, k=key: self._emit_changed(k, v))
            self._value_layout.addWidget(w)
            self._value_layout.addWidget(QLabel(f"({item.range_min}~{item.range_max})"))
            self._current_widget = w

        elif item.typ == "choice":
            w = QComboBox()
            w.addItems(item.choices)
            w.setCurrentText(str(current))
            w.currentTextChanged.connect(lambda v, k=key: self._emit_changed(k, v))
            self._value_layout.addWidget(w)
            self._current_widget = w

        elif item.typ == "string":
            w = QLineEdit()
            w.setText(str(current))
            w.textChanged.connect(lambda v, k=key: self._emit_changed(k, v))
            self._value_layout.addWidget(w, 1)
            self._current_widget = w

    def _emit_changed(self, key, value):
        setattr(self._cfg, key, value)

    def _on_changed_bool(self, key, checked):
        setattr(self._cfg, key, checked)
        if isinstance(self._current_widget, QCheckBox):
            self._current_widget.setText("开启" if checked else "关闭")
