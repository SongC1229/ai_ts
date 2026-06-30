"""主窗口 UI"""
import os
import glob
import json
import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QProgressBar,
    QTextEdit, QTableView, QHeaderView,
    QFileDialog, QGroupBox, QCheckBox, QSlider, QRadioButton, QComboBox,
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from ui.settings_dialog import SettingsDialog
from ui.table_models import (
    SubtitleTableModel, SrcTableModel, ButtonDelegate, GenderFilterProxy,
    SUBTITLE_COLUMNS, SRC_COLUMNS, SRC_COL_START, BUTTON_COLUMNS,
    COL_START, COL_GENDER, COL_TEXT, COL_PLAY_TTS, COL_PLAY_MIX, COL_PLAY_RAW_TTS, COL_PLAY_VOCAL, COL_COMPARE, COL_REGEN,
    COL_IDX,
)


from ui.waveform_widget import WaveformPreviewWidget
from core.srt_parser import parse_srt
from ui.playback_mixin import PlaybackMixin
from ui.cache_mixin import CacheMixin
from ui.pipeline_mixin import PipelineMixin
from ui.execution_mixin import ExecutionMixin
from core.log_utils import setup_logging
from config import cfg
from core.cache_manager import Step


class MainWindow(PlaybackMixin, CacheMixin, PipelineMixin, ExecutionMixin, QMainWindow):
    """主窗口 — Mixin 架构:
    - PlaybackMixin: 播放控制、波形预览
    - CacheMixin: 缓存管理、校准加载
    - PipelineMixin: 流水线回调、步骤/字幕状态 UI
    - ExecutionMixin: 流水线/单步执行、重混音/重合成
    """

    # ── Signals (必须在 MainWindow 中定义,PySide6 Mixin 兼容性) ──
    _log_signal = Signal(str)
    _wave_done_signal = Signal(int)

    def closeEvent(self, event):
        # 取消波形加载线程（daemon=True 会自动回收,但显式取消更安全)
        if self._wave_cancel is not None:
            self._wave_cancel.set()
        # 停止正在运行的单步/流水线线程
        for _worker_name in ('_single_step_worker', '_pipeline_worker'):
            _w = getattr(self, _worker_name, None)
            if _w is not None and _w.isRunning():
                # 取消并唤醒可能在 _error_cond / _config_cond 上等待的 worker
                try:
                    _w._cancelled = True
                    if hasattr(_w, 'ctx'):
                        _w.ctx.cancelled = True
                    _w._error_mutex.lock()
                    _w._error_cond.wakeAll()
                    _w._error_mutex.unlock()
                    if hasattr(_w, '_config_cond'):
                        _w._config_mutex.lock()
                        _w._config_cond.wakeAll()
                        _w._config_mutex.unlock()
                except Exception:
                    pass
                _w.quit()
                _w.wait(3000)
        super().closeEvent(event)

    def __init__(self):
        super().__init__()
        # _log_signal 用于跨线程安全
        self._log_signal.connect(self.log)
        # 波形加载完成信号（QueuedConnection 确保跨线程安全)
        self._wave_done_signal.connect(self._on_wave_done, Qt.ConnectionType.QueuedConnection)
        self.setWindowTitle("电影 AI 配音工具")
        self.setMinimumSize(1430, 900)

        self.cfg = cfg  # 全局配置实例
        self.current_subtitles = []
        # 缓存状态跟踪：避免step 0-3 未变化时重复打印详情
        self._last_cache_detail = []
        self._last_srt_path = ""     # 字幕文件路径跟踪,避免重复解析
        self._pipeline_running = False  # 管线运行时跳过缓存检查
        self._tts_cache_hits = {}       # idx -> True 已确认缓存命中,避免重复检查
        self._tts_error_dlg = None      # 当前显示的 TTS 错误弹窗（非模态)
        self.tts_paths = {}          # row_index -> tts_file_path (mixed_clip)
        self.raw_tts_paths = {}      # row_index -> raw_tts_file_path (before mixing)
        self.subtitle_row_map = {}   # subtitle_1based_index -> table_row
        self._row_to_idx = {}      # table_row -> subtitle_1based_index
        self._gender_click_count = {}  # idx -> 性别点击计数
        self._gender_cycle = {}        # idx -> 固定三态循环列表

        # CacheManager 实例缓存（路径未变时复用,避免重复创建)
        self._cache = None
        self._cache_video_path = ""

        # 音频播放器（用于试听)
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        # 系统输出设备变更时重建 QAudioOutput（播放前 _play_audio 中检查)
        # 播放状态
        self._play_end_ms = 0
        self._last_play_end = 0
        self._pending_seek = None
        self._wave_gen = 0
        self._wave_cancel = None
        self._setup_ui()
        self._setup_connections()
        self._setup_player_connections()
        # 同步提示音模式到设置（setup_ui 中 setChecked 时信号未连接)
        self._on_prompt_mode_changed()
        # 初始化缓存状态
        QTimer.singleShot(100, self._update_cache_status)
        # 设置日志系统：logging → QTextEdit + 文件（项目根目录)
        setup_logging(self.log_text, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 500ms 后清理残留的 Qwen API 进程和 GPU 显存
        QTimer.singleShot(500, self._cleanup_qwen_residual)
        # 窗口初始 X=460,为 SRC 预留左侧空间（在 show() 之前已定位)
        self.move(460, 120)
        # 触发默认视频的字幕搜索（setText 在 _setup_connections 之前,需显式调用)
        default_video = self.video_path_edit.text()
        if os.path.exists(default_video):
            self._auto_detect_srt(default_video)

    def _on_prompt_mode_changed(self):
        val = self.prompt_mode_fixed.isChecked()
        cfg.use_fixed_ref = val

    def _on_api_url_changed(self):
        val = self.api_url_edit.text().strip()
        cfg.tts_api_url = val

    def _default_settings(self) -> dict:
        import os as _os
        # 默认提示音路径
        cwd = _os.getcwd()
        m_wav = _os.path.join(cwd, "role", "男.wav")
        m_mp3 = _os.path.join(cwd, "role", "男.mp3")
        f_wav = _os.path.join(cwd, "role", "女.wav")
        f_mp3 = _os.path.join(cwd, "role", "女.mp3")
        m_path = m_wav if _os.path.exists(m_wav) else (m_mp3 if _os.path.exists(m_mp3) else "")
        f_path = f_wav if _os.path.exists(f_wav) else (f_mp3 if _os.path.exists(f_mp3) else "")
        if not cfg.fixed_ref_audio_male:
            cfg.fixed_ref_audio_male = m_path
        if not cfg.fixed_ref_audio_female:
            cfg.fixed_ref_audio_female = f_path

    # ── UI 搭建 ──────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 内容区容器（固定宽度,SRC 切换时保持位置不变)
        self._content_widget = QWidget()
        content_layout = QVBoxLayout(self._content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        # ===== 顶部: 文件选择 =====
        file_group = QGroupBox("输入文件")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(4)

        # 用 QGridLayout 对齐三行
        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setColumnStretch(1, 1)   # 视频/字幕路径列拉伸

        lbl1 = QLabel("视频文件:")
        lbl1.setFixedWidth(65)
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("选择视频文件...")
        self.video_path_edit.setText("pppe-080_2min.mp4")
        self.video_path_edit.setText("虞美人中文.mp4")
        self.btn_browse_video = QPushButton("浏览...")
        self.btn_browse_video.setFixedWidth(60)
        grid.addWidget(lbl1, 0, 0)
        grid.addWidget(self.video_path_edit, 0, 1)
        grid.addWidget(self.btn_browse_video, 0, 2)
        # 男声提示音（同排右侧)
        grid.addWidget(QLabel("男声:"), 0, 3)
        self.fixed_ref_male_edit = QLineEdit()
        self.fixed_ref_male_edit.setPlaceholderText("男.wav")
        self.fixed_ref_male_edit.setText(os.path.basename(cfg.fixed_ref_audio_male))
        self.fixed_ref_male_edit.textChanged.connect(lambda t: self._on_ref_path_changed("male", t))
        grid.addWidget(self.fixed_ref_male_edit, 0, 4)
        _bm = QPushButton("浏览...")
        _bm.setFixedWidth(60)
        _bm.clicked.connect(lambda: self._browse_ref_file("male"))
        grid.addWidget(_bm, 0, 5)
        _pm = QPushButton("▶")
        _pm.setFixedWidth(32)
        _pm.clicked.connect(lambda: self._play_ref_file("male"))
        grid.addWidget(_pm, 0, 6)

        lbl2 = QLabel("目标字幕:")
        lbl2.setFixedWidth(65)
        self.srt_path_edit = QLineEdit()
        self.srt_path_edit.setPlaceholderText("选择目标 SRT 字幕文件（配音用)...")
        self.btn_browse_srt = QPushButton("浏览...")
        self.btn_browse_srt.setFixedWidth(60)
        grid.addWidget(lbl2, 1, 0)
        grid.addWidget(self.srt_path_edit, 1, 1)
        grid.addWidget(self.btn_browse_srt, 1, 2)
        # 女声提示音（同排右侧)
        grid.addWidget(QLabel("女声:"), 1, 3)
        self.fixed_ref_female_edit = QLineEdit()
        self.fixed_ref_female_edit.setPlaceholderText("女.wav")
        self.fixed_ref_female_edit.setText(os.path.basename(cfg.fixed_ref_audio_female))
        self.fixed_ref_female_edit.textChanged.connect(lambda t: self._on_ref_path_changed("female", t))
        grid.addWidget(self.fixed_ref_female_edit, 1, 4)
        _bf = QPushButton("浏览...")
        _bf.setFixedWidth(60)
        _bf.clicked.connect(lambda: self._browse_ref_file("female"))
        grid.addWidget(_bf, 1, 5)
        _pf = QPushButton("▶")
        _pf.setFixedWidth(32)
        _pf.clicked.connect(lambda: self._play_ref_file("female"))
        grid.addWidget(_pf, 1, 6)

        lbl3 = QLabel("原声字幕:")
        lbl3.setFixedWidth(65)
        self.src_srt_path_edit = QLineEdit()
        self.src_srt_path_edit.setPlaceholderText("选择原声 SRT（可选,用于时间对齐)...")
        self.btn_browse_src_srt = QPushButton("浏览...")
        self.btn_browse_src_srt.setFixedWidth(60)
        grid.addWidget(lbl3, 2, 0)
        grid.addWidget(self.src_srt_path_edit, 2, 1)
        grid.addWidget(self.btn_browse_src_srt, 2, 2)

        # 输出目录隐藏控件（自动设为视频所在目录)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setVisible(False)

        # 与下方字幕列表等宽（左侧 panel : 右侧 panel ≈ 800:455)
        file_row = QHBoxLayout()
        grid_widget = QWidget()
        grid_widget.setLayout(grid)
        file_row.addWidget(grid_widget, 1)
        file_layout.addLayout(file_row)

        # ===== 顶部右侧：区域设置 =====
        cfg_group = QGroupBox("区域设置")
        cfg_layout = QVBoxLayout(cfg_group)
        cfg_layout.setSpacing(4)
        cfg_grid = QGridLayout()
        cfg_grid.setSpacing(4)

        _lbl_api = QLabel("API 地址:")
        _lbl_api.setFixedWidth(65)
        cfg_grid.addWidget(_lbl_api, 0, 0)
        self.api_url_edit = QLineEdit()
        self.api_url_edit.setPlaceholderText("http://localhost:9001")
        self.api_url_edit.setText(cfg.tts_api_url)
        cfg_grid.addWidget(self.api_url_edit, 0, 1)

        # 本地引擎开关
        self.local_tts_cb = QCheckBox("本地引擎 (IndexTTS2)")
        self.local_tts_cb.setChecked(cfg.use_local_tts)
        self.local_tts_cb.toggled.connect(self._on_local_tts_toggled)
        cfg_grid.addWidget(self.local_tts_cb, 0, 2)
        # 默认根据状态禁用 API 地址
        self.api_url_edit.setEnabled(not self.local_tts_cb.isChecked())

        self.prompt_mode_original = QRadioButton("原视频")
        self.prompt_mode_original.setFixedWidth(80)
        self.prompt_mode_original.setChecked(True)
        self.prompt_mode_fixed = QRadioButton("固定提示")
        self.prompt_mode_fixed.setFixedWidth(80)
        self.btn_settings = QPushButton("⚙ 设置")
        self.btn_settings.setFixedWidth(80)
        p_row = QHBoxLayout()
        p_row.setSpacing(2)
        p_row.addWidget(QLabel("提示音:"))
        p_row.addWidget(self.prompt_mode_original)
        p_row.addWidget(self.prompt_mode_fixed)
        p_row.addStretch()
        cfg_grid.addLayout(p_row, 1, 0, 1, 2)
        cfg_grid.addWidget(self.btn_settings, 1, 2, Qt.AlignRight | Qt.AlignVCenter)

        # ── 第3行: 通用配置 ──
        from ui.config_panel import ConfigPanel
        self.config_panel = ConfigPanel(cfg)
        cfg_grid.addWidget(self.config_panel, 2, 0, 1, 3)

        cfg_widget = QWidget()
        cfg_widget.setLayout(cfg_grid)
        cfg_layout.addWidget(cfg_widget)

        # 田字格布局：col0=vad | col1=file+preview | col2=cfg+right
        content_grid = QGridLayout()
        content_grid.setSpacing(0)
        content_grid.setVerticalSpacing(4)
        # 3列：vad(450px) + preview(970px) + right(500px)
        content_grid.addWidget(file_group, 0, 1)  # 仅在 preview 上方
        content_grid.addWidget(cfg_group, 0, 2)  # 仅在 right 上方

        # SRC语音区间列表（左侧)
        src_group = QGroupBox()
        src_group.setMinimumWidth(450)
        src_group.setMaximumWidth(450)
        src_inner = QVBoxLayout(src_group)
        src_inner.setContentsMargins(4, 0, 4, 4)

        # SRC 标题行（与右侧预览组标题行等高,保证表格 Y 对齐)
        src_title_bar = QHBoxLayout()
        src_title_bar.setContentsMargins(2, 2, 2, 2)
        self._btn_switch_src = QPushButton("切换到原始")
        self._btn_switch_src.setFixedWidth(78)
        self._btn_switch_src.setFixedHeight(22)
        self._btn_switch_src.setEnabled(False)
        self._btn_switch_src.setVisible(False)
        self._btn_switch_src.clicked.connect(self._toggle_src_view)
        self._src_view_mode = 0  # 0=原始, 1=Qwen校准
        self._src_subs = []
        src_title = QLabel("<b>原声字幕</b>")
        src_title.setFixedHeight(22)
        src_title.setFixedWidth(180)
        src_title_bar.addWidget(src_title)
        self._src_title_label = src_title  # 保存引用,用于校准后更新标题
        src_title_bar.addSpacing(4)
        src_title_bar.addWidget(self._btn_switch_src)
        src_title_bar.addStretch()
        src_title_w = QWidget()
        src_title_w.setFixedHeight(28)  # 与 preview_group 标题行等高,确保表格首行 Y 对齐
        src_title_w.setLayout(src_title_bar)
        src_inner.addWidget(src_title_w)
        # 使用 QTableView + Model 替代 QTableWidget
        self._src_model = SrcTableModel()
        self.src_table = QTableView()
        self.src_table.setModel(self._src_model)
        for ci in range(len(SRC_COLUMNS)):
            self.src_table.horizontalHeader().setSectionResizeMode(ci, QHeaderView.ResizeMode.Fixed)
        for ci, (_, w) in enumerate(SRC_COLUMNS):
            if w > 0:
                self.src_table.setColumnWidth(ci, w)
        self.src_table.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.src_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.src_table.setAlternatingRowColors(True)
        self.src_table.verticalHeader().hide()
        self.src_table.verticalHeader().setDefaultSectionSize(28)
        self.src_table.setStyleSheet("QTableView::item { padding: 2px 3px; }")
        self.src_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.src_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.src_table.setEnabled(True)
        src_inner.addWidget(self.src_table, 1)
        src_bottom_w = QWidget()
        src_bottom_w.setFixedHeight(30)
        src_bottom = QHBoxLayout(src_bottom_w)
        src_bottom.setContentsMargins(0, 0, 0, 0)
        self.src_count_label = QLabel("原声字幕: 0")
        src_bottom.addWidget(self.src_count_label)
        src_bottom.addStretch()
        src_inner.addWidget(src_bottom_w)
        self.src_group = src_group  # 保存引用用于隐藏/展开
        self.src_group.setVisible(False)  # 默认折叠
        # 字幕表格（右侧)
        preview_group = QGroupBox()
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(4, 0, 4, 4)

        # 自定义标题行：按钮 + "字幕预览" 文字,与左侧 SRC 表标题 Y 对齐
        title_bar = QHBoxLayout()
        title_bar.setContentsMargins(2, 2, 2, 2)
        self.btn_toggle_src = QPushButton("▶ 原声字幕")
        self.btn_toggle_src.setFixedSize(90, 22)
        self.btn_toggle_src.setToolTip("隐藏/展开左侧原声字幕表")
        self.btn_toggle_src.clicked.connect(self._toggle_src_panel)
        title_bar.addWidget(self.btn_toggle_src)
        title_bar.addSpacing(4)
        preview_title = QLabel("字幕预览 — 点击 ▶ 试听 TTS 合成结果")
        title_bar.addWidget(preview_title)
        title_bar.addStretch()
        preview_title_w = QWidget()
        preview_title_w.setFixedHeight(28)
        preview_title_w.setLayout(title_bar)
        preview_layout.addWidget(preview_title_w)
        # 使用 QTableView + Model + Delegate 替代 QTableWidget
        self._subtitle_model = SubtitleTableModel()
        # 性别筛选代理
        self._gender_proxy = GenderFilterProxy()
        self._gender_proxy.setSourceModel(self._subtitle_model)
        self.subtitle_table = QTableView()
        self.subtitle_table.setModel(self._gender_proxy)
        # 设置按钮委托
        self._btn_delegate = ButtonDelegate(BUTTON_COLUMNS, self.subtitle_table)
        self._btn_delegate.buttonClicked.connect(self._on_button_clicked)
        for col in BUTTON_COLUMNS:
            self.subtitle_table.setItemDelegateForColumn(col, self._btn_delegate)
        # 列宽设置
        self.subtitle_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        for ci in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]:
            self.subtitle_table.horizontalHeader().setSectionResizeMode(ci, QHeaderView.ResizeMode.Fixed)
        for ci, (_, w) in enumerate(SUBTITLE_COLUMNS):
            if w > 0:
                self.subtitle_table.setColumnWidth(ci, w)
        self.subtitle_table.setAlternatingRowColors(True)
        self.subtitle_table.verticalHeader().hide()
        self.subtitle_table.verticalHeader().setDefaultSectionSize(28)
        self.subtitle_table.setStyleSheet("QTableView::item { padding: 2px 3px; }")
        self.subtitle_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.subtitle_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        # 连接点击事件（用于性别列点击切换)
        self.subtitle_table.clicked.connect(self._on_table_clicked)
        # 表头点击：字幕文本列 → 切换隐藏/显示
        self.subtitle_table.horizontalHeader().sectionClicked.connect(
            lambda col: self._subtitle_model.toggle_hide_text() if col == COL_TEXT else None)
        preview_layout.addWidget(self.subtitle_table, 1)
        sub_info_w = QWidget()
        sub_info_w.setFixedHeight(30)
        sub_info_layout = QHBoxLayout(sub_info_w)
        sub_info_layout.setContentsMargins(0, 0, 0, 0)
        self.sub_count_label = QLabel("字幕数量: 0")
        sub_info_layout.addWidget(self.sub_count_label)
        # 性别筛选
        self.gender_filter = QComboBox()
        self.gender_filter.addItems(["全部", "男", "女", "未定"])
        # 用 activated 替代 currentIndexChanged,确保用户点击同一项也能触发筛选
        self.gender_filter.activated.connect(self._on_gender_filter_changed)
        sub_info_layout.addWidget(self.gender_filter)
        # TTS 状态筛选
        self.status_filter = QComboBox()
        self.status_filter.addItems(["全部状态", "等待中", "TTS 完成", "混音完成", "跳过"])
        self.status_filter.activated.connect(self._on_status_filter_changed)
        sub_info_layout.addWidget(self.status_filter)
        # 筛选无结果提示
        self.lbl_filter_empty = QLabel("无匹配结果")
        self.lbl_filter_empty.setStyleSheet("color: #FF5722; font-weight: bold;")
        self.lbl_filter_empty.setVisible(False)
        sub_info_layout.addWidget(self.lbl_filter_empty)
        sub_info_layout.addStretch()
        # 缓存状态 + 清空按钮
        self.lbl_cache_status = QLabel("缓存: 未启用")
        self.lbl_cache_status.setStyleSheet("color: #999999;")
        sub_info_layout.addWidget(self.lbl_cache_status)
        self.btn_clear_cache = QPushButton("🗑 清空缓存")
        self.btn_clear_cache.clicked.connect(self._clear_cache)
        self.btn_clear_cache.setEnabled(False)
        sub_info_layout.addWidget(self.btn_clear_cache)
        # 不清除缓存（默认不勾选)
        self.keep_temp_cb = QCheckBox("不清临时文件")
        self.keep_temp_cb.setChecked(True)
        self.keep_temp_cb.setToolTip("勾选后保留临时工作目录文件")
        sub_info_layout.addWidget(self.keep_temp_cb)
        preview_layout.addWidget(sub_info_w)
        content_grid.addWidget(preview_group, 1, 1)
        content_grid.setColumnStretch(1, 1)
        self.preview_group = preview_group
        self.preview_group.setMinimumWidth(920)

        # ── 右：控制按钮 + step进度 + 混音预览 + 日志 ──
        right_panel = QWidget()
        right_panel.setFixedWidth(500)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 开始/停止按钮放在右侧面板顶部
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(4)
        self.btn_start = QPushButton("▶ 开始配音")
        self.btn_start.setMinimumHeight(32)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; color: white; font-weight: bold;
                font-size: 13px;
                border-radius: 4px;
                padding: 4px 20px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        ctrl_row.addWidget(self.btn_start)
        self.btn_cancel = QPushButton("⏹ 停止")
        self.btn_cancel.setMinimumHeight(32)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #f44336; color: white; font-weight: bold;
                border-radius: 4px;
                padding: 4px 20px;
            }
            QPushButton:hover { background-color: #da190b; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        ctrl_row.addWidget(self.btn_cancel)
        self.btn_open_output = QPushButton("📁 打开输出")
        self.btn_open_output.setMinimumHeight(32)
        self.btn_open_output.setEnabled(False)
        self.btn_open_output.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; color: white; font-weight: bold;
                border-radius: 4px;
                padding: 4px 16px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        ctrl_row.addWidget(self.btn_open_output)
        ctrl_row.addStretch()
        right_layout.addLayout(ctrl_row)

        # ── 分段进度（2行 × 3步 + 共用进度条)──
        self.step_names = ["1.提取音频", "2.分离人声", "3.区间检测",
                           "4.TTS合成", "5.全长重建", "6.合并视频"]
        step_group = QGroupBox("分段进度")
        step_layout = QVBoxLayout(step_group)
        step_layout.setSpacing(4)
        step_layout.setContentsMargins(6, 6, 6, 6)

        self._step_cells = []  # (icon_label, text_label, step_btn)

        step_grid = QGridLayout()
        step_grid.setSpacing(0)
        step_grid.setContentsMargins(0, 0, 0, 0)

        for row_idx in range(2):
            for col_idx in range(3):
                step_idx = row_idx * 3 + col_idx
                # 步骤图标 + 文字
                cell = QWidget()
                cell_layout = QHBoxLayout(cell)
                cell_layout.setSpacing(3)
                cell_layout.setContentsMargins(4, 2, 4, 2)
                icon = QLabel("⏳")
                icon.setFixedWidth(16)
                cell_layout.addWidget(icon)
                text = QLabel(self.step_names[step_idx])
                text.setStyleSheet("color: #000000;")  # 初始 未完成
                cell_layout.addWidget(text)
                cell_layout.addStretch()
                # 单步执行按钮（3D 凸起效果)
                step_btn = QPushButton("▶")
                step_btn.setFixedWidth(22)
                step_btn.setFixedHeight(18)
                step_btn.setToolTip(f"单独执行 {self.step_names[step_idx]}")
                step_btn.setStyleSheet("""
                    QPushButton {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #f8f8f8, stop:0.4 #e8e8e8, stop:1 #d0d0d0);
                        border: 1px solid #b0b0b0;
                        border-radius: 3px;
                        padding: 1px 3px;
                        font-size: 10px;
                    }
                    QPushButton:hover {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #e0f0ff, stop:0.4 #c0e0ff, stop:1 #90c8f8);
                        border: 1px solid #6a9fd8;
                    }
                    QPushButton:pressed {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #c0d8f0, stop:1 #a0c0e8);
                        border: 1px solid #5a8fc8;
                        padding-top: 2px;
                        padding-left: 4px;
                    }
                """)
                step_btn.clicked.connect(
                    lambda checked, idx=step_idx: self._on_step_btn_clicked(idx)
                )
                cell_layout.addWidget(step_btn)
                grid_col = col_idx * 2  # 0, 2, 4
                step_grid.addWidget(cell, row_idx, grid_col)
                self._step_cells.append((icon, text, step_btn))
                # 箭头（除每行最后一个)
                if col_idx < 2:
                    arrow = QLabel("  →  ")
                    arrow.setStyleSheet("color: #BBBBBB; font-size: 13px;")
                    step_grid.addWidget(arrow, row_idx, grid_col + 1)
            # 空列占满剩余空间,保证3列左对齐
            step_grid.setColumnStretch(5, 1)

        step_layout.addLayout(step_grid)

        # 第三行：当前步骤名 | 进度条 | 百分比
        progress_row = QWidget()
        progress_h = QHBoxLayout(progress_row)
        progress_h.setContentsMargins(4, 2, 4, 0)
        progress_h.setSpacing(6)
        self._step_progress_label = QLabel(self.step_names[0])
        self._step_progress_label.setFixedWidth(80)
        self._step_progress_label.setStyleSheet("color: #000000;")
        progress_h.addWidget(self._step_progress_label)
        self._step_progress_bar = QProgressBar()
        self._step_progress_bar.setFixedHeight(16)
        self._step_progress_bar.setTextVisible(False)
        self._step_progress_bar.setValue(0)
        progress_h.addWidget(self._step_progress_bar, 1)
        self._step_progress_pct = QLabel("0%")
        self._step_progress_pct.setFixedWidth(90)
        self._step_progress_pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._step_progress_pct.setStyleSheet("color: #000000;")
        progress_h.addWidget(self._step_progress_pct)
        step_layout.addWidget(progress_row)

        self._update_step_ui(-1)
        right_layout.addWidget(step_group)


        # 混音预览
        preview_audio_group = QGroupBox("混音预览")
        preview_audio_layout = QVBoxLayout(preview_audio_group)
        preview_audio_layout.setSpacing(3)
        mix_row = QHBoxLayout()
        self.btn_play_mix = QPushButton("▶ 播放混音")
        self.btn_play_mix.setFixedWidth(100)
        mix_row.addWidget(self.btn_play_mix)
        self.btn_stop_mix = QPushButton("⏹ 停止")
        self.btn_stop_mix.setFixedWidth(100)
        self.btn_stop_mix.setEnabled(False)
        mix_row.addWidget(self.btn_stop_mix)
        mix_row.addStretch()
        preview_audio_layout.addLayout(mix_row)
        play_row = QHBoxLayout()
        self.lbl_play_time = QLabel("00:00.000 / 00:00.000")
        self.lbl_play_time.setFixedWidth(180)
        play_row.addWidget(self.lbl_play_time)
        self.lbl_volume = QLabel("音量")
        play_row.addWidget(self.lbl_volume)
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(80)
        play_row.addWidget(self.volume_slider)
        preview_audio_layout.addLayout(play_row)
        # 波形图（拖拽平移 + 滚轮缩放)
        self.waveform_preview = WaveformPreviewWidget()
        self.waveform_preview.setEnabled(False)
        self.waveform_preview.precise_reload_needed.connect(self._on_precise_waveform_needed)
        preview_audio_layout.addWidget(self.waveform_preview, 1)
        right_layout.addWidget(preview_audio_group)

        # 日志
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        log_btn_layout = QHBoxLayout()
        self.btn_clear_log = QPushButton("清空日志")
        self.btn_clear_log.clicked.connect(self.log_text.clear)
        log_btn_layout.addWidget(self.btn_clear_log)
        log_btn_layout.addStretch()
        log_layout.addLayout(log_btn_layout)
        right_layout.addWidget(log_group, 1)  # 日志占据剩余空间

        content_grid.addWidget(right_panel, 1, 2)

        # SRC 放入网格第 1 行第 0 列,与 preview/right 同水平线
        content_grid.addWidget(self.src_group, 1, 0)
        main_layout.addWidget(self._content_widget)
        content_layout.addLayout(content_grid)
        self._regen_queue = []
        self._segments_data = []  # [(start_ms, end_ms), ...] 语音片段区间
        self._waveform_offset = 0  # ms, 区间波形时偏移量
        self._display_duration = 0  # ms, 当前显示音频总时长

    def _fmt_time(self, ms: int) -> str:
        h = ms // 3600000
        m = (ms % 3600000) // 60000
        s = (ms % 60000) // 1000
        ms3 = ms % 1000
        return f"{h:02d}:{m:02d}:{s:02d}.{ms3:03d}"

    # ── 信号连接 ──────────────────────────────────────────

    def _setup_connections(self):
        self.btn_browse_video.clicked.connect(self._browse_video)
        self.video_path_edit.textChanged.connect(self._auto_detect_srt)
        self.btn_browse_srt.clicked.connect(self._browse_srt)
        self.btn_browse_src_srt.clicked.connect(self._browse_src_srt)
        self.src_srt_path_edit.textChanged.connect(self._on_src_srt_path_changed)
        self.srt_path_edit.textChanged.connect(self._on_srt_path_changed)
        self.btn_start.clicked.connect(self._start_dub)
        self.btn_cancel.clicked.connect(self._cancel_dub)
        self.btn_settings.clicked.connect(self._open_settings)
        self.prompt_mode_fixed.toggled.connect(self._on_prompt_mode_changed)
        self.btn_open_output.clicked.connect(self._open_output_file)
        self.btn_play_mix.clicked.connect(self._play_mixed)
        self.btn_stop_mix.clicked.connect(self._stop_playback)
        self.api_url_edit.textChanged.connect(self._on_api_url_changed)
        # SRC 表：点击开始列播放原人声
        self.src_table.clicked.connect(self._on_src_table_clicked)
        # 缓存：输出目录变化时更新缓存状态
        self.output_dir_edit.textChanged.connect(self._update_cache_status)
        # 两个字幕表同步滚动
        self._src_scrolling = False
        self._dst_scrolling = False
        self.src_table.verticalScrollBar().valueChanged.connect(
            lambda v: self._sync_scroll(v, self.subtitle_table, '_src_scrolling', '_dst_scrolling'))
        self.subtitle_table.verticalScrollBar().valueChanged.connect(
            lambda v: self._sync_scroll(v, self.src_table, '_dst_scrolling', '_src_scrolling'))
    def _sync_scroll(self, value: int, target_table, src_flag: str, dst_flag: str):
        """同步两个字幕表的垂直滚动"""
        if getattr(self, src_flag, False):
            return
        setattr(self, dst_flag, True)
        target_table.verticalScrollBar().setValue(value)
        setattr(self, dst_flag, False)

    # ── 文件浏览 ──────────────────────────────────────────

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件",
            getattr(self, '_last_video_dir', ""),
            "视频文件 (*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm);;所有文件 (*.*)"
        )
        if path:
            path = os.path.normpath(path)
            self._last_video_dir = os.path.dirname(path)
            self.video_path_edit.setText(path)
            # 停止旧播放器和波形加载,避免切换后线程崩溃
            self.player.stop()
            self._play_end_ms = 0
            self._pending_seek = None
            if self._wave_cancel is not None:
                self._wave_cancel.set()
            self._cache = None  # 视频路径变化,失效缓存实例
            if not self.output_dir_edit.text():
                self.output_dir_edit.blockSignals(True)
                self.output_dir_edit.setText(str(Path(path).parent))
                self.output_dir_edit.blockSignals(False)
    def _auto_detect_srt(self, video_path: str):
        """自动查找同目录下的 SRT 文件（按视频文件名第一段匹配)"""
        # 停止播放,仅当正在播放或暂停时（避免异步阻塞)
        _s = self.player.playbackState()
        if _s in (QMediaPlayer.PlaybackState.PlayingState, QMediaPlayer.PlaybackState.PausedState):
            self.player.stop()
        self._tts_cache_hits.clear()
        self._last_srt_path = ""
        self.current_subtitles = []
        self.subtitle_row_map.clear()
        self.btn_open_output.setEnabled(False)
        self._last_output_path = ""
        self.reset_preview_buttons()
        self._src_model.clear_data()
        self.src_count_label.setText("原声字幕: 0")
        self.waveform_preview.set_waveform([], 1)
        self.waveform_preview.set_position(0)
        self.waveform_preview.setEnabled(False)
        vid_dir = str(Path(video_path).parent)
        vid_stem = Path(video_path).stem.split('.')[0]
        candidates = [os.path.join(vid_dir, f"{vid_stem}.srt")] + glob.glob(os.path.join(vid_dir, f"{vid_stem}.*.srt"))
        for c in candidates:
            if os.path.exists(c):
                self.srt_path_edit.setText(os.path.normpath(c))
                break
        # 自动检测缓存 ASR 原声字幕 (cache/<hash>/subs/asr.srt)
        # 自动检测缓存 ASR 原声字幕 (cache/<hash>/subs/asr.srt)
        from core.cache_manager import CacheManager
        cm = CacheManager(video_path, os.path.join(os.getcwd(), ".cache"))
        _asr_exists, _asr_path, _ = cm.file_info(Step.SUBS, "asr.srt")
        if _asr_exists:
            self.src_srt_path_edit.setText(os.path.normpath(_asr_path))
        # NOTE: 原声字幕仅从缓存 subs/asr.srt 加载。不检测视频同目录 .ja.srt
        # 校准缓存在 _on_srt_loaded 完成后加载（此时 subtitle_row_map 已就绪)

    def _browse_srt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择字幕文件", "",
            "字幕文件 (*.srt);;所有文件 (*.*)"
        )
        if path:
            self.srt_path_edit.setText(os.path.normpath(path))

    def _browse_src_srt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择原声字幕文件", "",
            "字幕文件 (*.srt);;所有文件 (*.*)"
        )
        if path:
            path = os.path.normpath(path)
            self.src_srt_path_edit.setText(path)
            # 直接触发加载（兼容选择同文件时 textChanged 不触发)
            self.load_raw_srt(path)

    def _browse_ref_file(self, gender: str):
        """浏览固定提示音文件"""
        path, _ = QFileDialog.getOpenFileName(self, f"选择{'男' if gender == 'male' else '女'}声提示音",
                                              "", "音频文件 (*.wav *.mp3);;所有文件 (*.*)")
        if path:
            key = "fixed_ref_audio_male" if gender == "male" else "fixed_ref_audio_female"
            edit = self.fixed_ref_male_edit if gender == "male" else self.fixed_ref_female_edit
            edit.setText(os.path.basename(path))
            setattr(cfg, key, path)

    def _on_srt_path_changed(self, path: str):
        if path and Path(path).suffix.lower() == '.srt':
            self._load_subtitles()

    def _on_src_srt_path_changed(self, path: str):
        """原声字幕路径变化时,加载并显示到 SRC 表"""
        if path and Path(path).suffix.lower() == '.srt' and os.path.exists(path):
            self.load_raw_srt(path)
        else:
            # 路径为空 → 清空 SRC 表
            self._src_model.clear_data()
            self.src_count_label.setText("原声字幕: 0")

    def load_raw_srt(self, ja_path: str):
        """加载原声字幕文件,填充到左侧 SRC 表"""
        try:
            subs = parse_srt(ja_path)
            # 保存 SubtitleItem 列表（模型使用)
            self._src_subs = subs
            # 保存原始元组数据（用于兼容/备份)
            self._src_view_mode = 0
            self._btn_switch_src.setVisible(True)
            self._btn_switch_src.setEnabled(True)
            # 使用 Model 更新数据（传入 SubtitleItem 列表)
            self._src_model.set_data(subs, is_calibrated=False)
            self.src_count_label.setText(f"原声字幕: {len(subs)} 条")
            # 加载校准缓存到原声字幕
            cm = self._get_cache()
            has_calib = False
            if cm:
                has_calib = cm.load_calib_cache(raw_subs=subs)
            _calib_n = sum(1 for s in subs if s.is_calibrated)
            self.log(f"加载原声字幕: {len(subs)} 条, 缓存校准: {_calib_n} 条" if has_calib else f"加载原声字幕: {len(subs)} 条, 无校准缓存")

            for ci, (_, w) in enumerate(SRC_COLUMNS):
                if w > 0:
                    self.src_table.setColumnWidth(ci, w)
        except Exception as e:
            self.log(f"【原声字幕加载失败】{e}")
            self._src_model.clear_data()
            self.src_count_label.setText("原声字幕: 0")

    # ── 字幕加载 ──────────────────────────────────────────

    def _load_subtitles(self):
        srt_path = self.srt_path_edit.text().strip()
        if not srt_path or not os.path.exists(srt_path):
            return
        if srt_path == self._last_srt_path and self.current_subtitles:
            return
        self._last_srt_path = srt_path

        # 显示加载状态,立即刷新 UI
        self.sub_count_label.setText("⏳ 正在加载字幕...")

        try:
            items = parse_srt(srt_path)
        except Exception as e:
            self.log(f"【错误】字幕解析失败: {e}")
            self.sub_count_label.setText("字幕数量: 0")
            return

        self.current_subtitles = items
        self.tts_paths.clear()
        self.subtitle_row_map.clear()
        self._row_to_idx.clear()
        self._gender_click_count.clear()
        self._gender_cycle.clear()

        count = len(items)
        self.sub_count_label.setText(f"正在填充表格 ({count} 条)...")
        # 先处理事件,让"正在加载"文字显示出来
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.processEvents()

        # 使用 Model 设置数据（自动处理样式和按钮)
        self._subtitle_model.set_data(items)
        for idx, sub in enumerate(items, 1):
            row = idx - 1
            self.subtitle_row_map[idx] = row
            self._row_to_idx[row] = idx
        # 立即刷新界面,让字幕表先显示出来（后续缓存扫描可能卡几秒)
        QCoreApplication.processEvents()

        self.sub_count_label.setText(f"字幕数量: {count}")
        # 检查缓存,更新试听和状态
        cm = self._get_cache()
        has_demucs_vocals = False
        has_extract_mix = False
        gender_cache = {}
        if cm:
            demucs_vocals_path = cm.vocals_path
            has_demucs_vocals = os.path.exists(demucs_vocals_path)
            extract_mix_path = cm.mix_orig_path
            has_extract_mix = os.path.exists(extract_mix_path)

            # 一次性加载性别+校准到 current_subtitles
            gender_cache = cm.load_gender_cache(items)
            has_calib = cm.load_calib_cache(items)
            _calib_n = sum(1 for s in items if s.is_calibrated)
            self.log(f"加载目标字幕: {count} 条, 缓存校准: {_calib_n} 条" if has_calib else f"加载目标字幕: {count} 条, 无校准缓存")
            # 立即刷新界面显示性别和校准时间
            n = self._subtitle_model.rowCount()
            if n > 0:
                self._subtitle_model.dataChanged.emit(
                    self._subtitle_model.index(0, 0),
                    self._subtitle_model.index(n - 1, 10)
                )
            QCoreApplication.processEvents()

            # TTS 扫描放后台线程（较慢）
            from PySide6.QtCore import QThread, Signal as TSig
            class TtsScanner(QThread):
                done = TSig(object, object, bool)
                def run(self):
                    _tts_cache, _mixed_cache = cm.scan_tts_cache()
                    _exists, _, _ = cm.file_info(Step.MIX, "final_audio.wav")
                    self.done.emit(_tts_cache, _mixed_cache, _exists)

            def _on_scan_done(_tts_cache, _mixed_cache, _mixed_final_exists):
                self._subtitle_model.blockSignals(True)
                try:
                    for j, sub in enumerate(self.current_subtitles, 1):
                        row = self.subtitle_row_map.get(j)
                        if row is None:
                            continue
                        tts_matches = _tts_cache.get(j)
                        if tts_matches is not None:
                            self.raw_tts_paths[row] = tts_matches
                        mixed_matches = _mixed_cache.get(j)
                        if mixed_matches is not None:
                            g_val = sub.gender
                            tts_gender = {"male": "男", "female": "女"}.get(g_val, "?")
                            self._tts_cache_hits[j] = {"path": mixed_matches, "gender": tts_gender}
                            self.tts_paths[row] = mixed_matches
                            status_key = "mixed" if _mixed_final_exists else "tts_done"
                            self._subtitle_model.set_status(row, status_key)
                        elif tts_matches is not None:
                            self._subtitle_model.set_status(row, "tts_done")
                finally:
                    self._subtitle_model.blockSignals(False)
                n = self._subtitle_model.rowCount()
                if n > 0:
                    self._subtitle_model.dataChanged.emit(
                        self._subtitle_model.index(0, 0),
                        self._subtitle_model.index(n - 1, 10)
                    )
                cached_count = len(self._tts_cache_hits)
                self.log(f"TTS {cached_count}/{count} | 人声 {'✅' if has_demucs_vocals else '❌'} | 原声 {'✅' if has_extract_mix else '❌'} | 性别 {len(gender_cache)}/{count}")

            self._tts_scanner = TtsScanner()
            self._tts_scanner.done.connect(_on_scan_done)
            self._tts_scanner.start()

        self._update_cache_status()

    def _on_precise_waveform_needed(self, file_path: str, start_ms: int, end_ms: int):
        """缩放到 <=3min 时加载精确波形"""
        self.waveform_preview._preserve_view = True
        self._load_waveform_preview(file_path, start_ms, end_ms)

    # ── 试听播放 ──────────────────────────────────────────

    def _cleanup_qwen_residual(self):
        """启动后清理残留的 Qwen API 进程和 GPU 显存（后台线程)"""
        import threading as _th
        def _run():
            import urllib.request
            for port in [8765, 8766, 8767]:
                try:
                    r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.3)
                    if r.status == 200:
                        self._log_signal.emit(f"检测到残留 Qwen API (端口 {port}),发送卸载请求")
                        try:
                            urllib.request.urlopen(f"http://127.0.0.1:{port}/shutdown", timeout=2)
                        except Exception:
                            pass
                except Exception:
                    pass
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            self._log_signal.emit("GPU 显存已清理")
        _th.Thread(target=_run, daemon=True).start()
    def _on_ref_path_changed(self, gender: str, path: str):
        """固定提示音路径变更"""
        setattr(cfg, f"fixed_ref_audio_{gender}", path)
        self.log(f"  固定提示音({gender}): {path}")

    def _on_local_tts_toggled(self, checked: bool):
        """本地引擎开关切换：禁用/启用 API 地址输入"""
        self.api_url_edit.setEnabled(not checked)
        cfg.use_local_tts = checked

    def _open_settings(self):
        dialog = SettingsDialog(cfg, self)
        # 支持运行时修改参数（如 TTS API 地址/模式),用于错误重试时调整
        # 注意：运行时修改 Demucs/缓存等参数需等待下一次完整流水线
        dialog.load_from_cfg()

        if dialog.exec():
            # 设置对话框的修改同步到 cfg
            _dialog_settings = dialog.get_settings()
            for k, v in _dialog_settings.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            self.log(f"设置已更新: TTS 模式={cfg.tts_mode}, "
                     f"API={cfg.tts_api_url}")
    def _open_output_file(self):
        """打开文件管理器并选中输出文件"""
        output_path = getattr(self, '_last_output_path', '')
        if output_path and os.path.exists(output_path):
            import subprocess
            subprocess.run(['explorer', '/select,', os.path.abspath(output_path)])

    # ── 工作线程回调 ──────────────────────────────────────
    def _toggle_src_view(self):
        """切换原声字幕表：原始 ↔ Qwen校准"""
        if not hasattr(self, '_src_subs'):
            return


        if self._src_view_mode == 1:
            # 切换到原始
            self._src_view_mode = 0
            self._btn_switch_src.setText("切换到Qwen")
            if hasattr(self, '_src_title_label'):
                self._src_title_label.setText("<b>原声字幕</b>")
            self._src_model.set_data(self._src_subs, is_calibrated=False)
            self.src_count_label.setText(f"原声字幕: {len(self._src_subs)} 条")
        else:
            # 切换到 Qwen 校准
            if not any(s.is_calibrated for s in self._src_subs):
                self.log("Qwen 校准数据不存在")
                return
            self._src_view_mode = 1
            self._btn_switch_src.setText("切换到原始")
            self._src_model.set_data(self._src_subs, is_calibrated='Qwen')
            if hasattr(self, '_src_title_label'):
                self._src_title_label.setText("<b>原声字幕（Qwen校准)</b>")
            self.src_count_label.setText(f"原声字幕（Qwen): {len(self._src_subs)} 条")
            self.src_table.viewport().update()

        for ci, (_, w) in enumerate(SRC_COLUMNS):
            if w > 0:
                self.src_table.setColumnWidth(ci, w)
    def _on_gender_filter_changed(self, idx: int):
        """性别筛选下拉框切换"""
        _map = ["", "男", "女", "未定"]
        self._gender_proxy.set_filter_gender(_map[idx] if idx > 0 else "")
        self._update_filter_empty_label()

    def _on_status_filter_changed(self, idx: int):
        """TTS 状态筛选下拉框切换"""
        _map = ["", "pending", "tts_done", "mixed", "skipped"]
        self._gender_proxy.set_filter_status(_map[idx] if idx > 0 else "")
        self._update_filter_empty_label()

    def _update_filter_empty_label(self):
        """筛选无结果时显示提示,同时更新字幕计数"""
        _total = self._subtitle_model.rowCount()
        _visible = self._gender_proxy.rowCount()
        if _visible < _total:
            self.sub_count_label.setText(f"字幕: {_visible}/{_total}")
            self.lbl_filter_empty.setVisible(_visible == 0)
        else:
            self.sub_count_label.setText(f"字幕数量: {_total}")
            self.lbl_filter_empty.setVisible(False)

    def _on_src_table_clicked(self, index):
        """SRC 表：点击开始列播放原人声区间"""
        col = index.column()
        if col != SRC_COL_START:  # 开始列
            return
        row = index.row()
        self._play_src_segment(row)

    def _toggle_src_panel(self):
        """隐藏/展开 SRC 表,向左扩展/收缩"""
        SRC_W = 450
        visible = self.src_group.isVisible()
        x, y, w, h = self.x(), self.y(), self.width(), self.height()
        cw = self.centralWidget()
        cw.setVisible(False)
        from PySide6.QtCore import QCoreApplication as _QCA
        _QCA.processEvents()
        try:
            if visible:
                self.src_group.setVisible(False)
                nw = max(w - SRC_W, 800)
                self.setMinimumSize(nw, h)
                self.move(x + SRC_W, y)
                self.resize(nw, h)
                self.btn_toggle_src.setText("▶ 原声字幕")
            else:
                nw = self.width() + SRC_W
                self.setMinimumSize(nw, h)
                self.resize(nw, h)
                self.move(x - SRC_W, y)
                self.src_group.setVisible(True)
                self.btn_toggle_src.setText("◀ 原声字幕")
        finally:
            _QCA.processEvents()
            cw.setVisible(True)


    def _on_button_clicked(self, row: int, col: int):
        """处理按钮列点击事件（由 ButtonDelegate 发出)"""
        # 代理行 → 源模型行
        proxy_idx = self._gender_proxy.index(row, col)
        src_row = self._gender_proxy.mapToSource(proxy_idx).row()
        if col == COL_PLAY_TTS:
            self._play_tts_segment(src_row)
        elif col == COL_PLAY_MIX:
            self._play_mix_audio(src_row)
        elif col == COL_PLAY_RAW_TTS:
            self._play_raw_tts(src_row)
        elif col == COL_PLAY_VOCAL:
            self._play_orig_audio(src_row)
        elif col == COL_COMPARE:
            self._show_waveform(src_row)
        elif col == COL_REGEN:
            self._regen_segment(src_row)

    def _on_table_clicked(self, index):
        """处理表格点击事件（用于性别列切换和开始时间播放)"""
        # 代理模型 → 源模型行
        src_idx = self._gender_proxy.mapToSource(index)
        row = src_idx.row()
        col = index.column()
        # 点击序号 → 从原人声 seek 播放 + 展示全长原人声波形
        if col == COL_IDX:
            seg = self._subtitle_model.get_times(row)
            if seg == (0, 0):
                self.log(f"【警告】第 {row+1} 行字幕时间区间不可用")
                return
            start_ms, end_ms = seg
            cm = self._get_cache()
            if not cm:
                self.log("【警告】请先选择视频文件")
                return
            vocals_path = cm.vocals_path
            if not os.path.exists(vocals_path):
                self.log(f"【警告】第 {row+1} 行人声文件不可用")
                return
            self._play_audio(vocals_path, start_ms=start_ms, end_ms=end_ms, label="人声", row=row)
            self.waveform_preview.setEnabled(True)
            self._load_waveform_preview(vocals_path)
            return
        # 点击开始时间列 → 播放混音
        if col == COL_START:
            cm = self._get_cache()
            mix_path = cm.final_mix_path if cm else ""
            if not os.path.exists(mix_path):
                self.log(f"【警告】第 {row+1} 行 全长混音不可用")
                return
            seg = self._subtitle_model.get_times(row)
            if seg == (0, 0):
                self.log(f"【警告】第 {row+1} 行字幕时间区间不可用")
                return
            start_ms, end_ms = seg
            _edge = cfg.edge_ms
            self._play_audio(mix_path, start_ms=max(0, start_ms - _edge))
            dur_s = (end_ms - start_ms) / 1000
            self.log(f"▶ 全长混音: 第{row+1}行 ({dur_s:.1f}s)")
            self.waveform_preview.setEnabled(True)
            self.log("⏳ 正在加载全长波形...")
            self._load_waveform_preview(mix_path)
            return
        # 点击性别列切换
        if col != COL_GENDER:
            return
        idx = row + 1
        cur = self._subtitle_model.get_gender(idx)
        # 基于初始性别固定 cycle,避免每次点击后 cur 变化导致路径偏移
        if idx not in self._gender_cycle:
            if cur == "male":
                self._gender_cycle[idx] = ["female", "", "male"]
            elif cur == "female":
                self._gender_cycle[idx] = ["male", "", "female"]
            else:
                self._gender_cycle[idx] = ["female", "male", ""]
        fixed_cycle = self._gender_cycle[idx]
        _count = self._gender_click_count.get(idx, 0) % 3
        new_gender = fixed_cycle[_count]
        self._gender_click_count[idx] = _count + 1
        self._subtitle_model.update_gender(idx, new_gender)
        gender_display = {"male": "男", "female": "女", "": "空"}.get(new_gender, new_gender)
        self.log(f"第 {idx} 条性别手动更新为: {gender_display}")
        # 手动修改性别时钉住该行,使其在当前筛选下仍然可见
        self._gender_proxy.pin_row(row)
        # 同步到性别缓存,确保流水线读取最新性别
        cm = self._get_cache()
        if cm:
            cm.update_gender(idx, new_gender)
            # 性别改变后清除对应 TTS + 混音缓存,确保下一次重新合成
            import glob as _g
            for _f in _g.glob(os.path.join(cm.cache_dir, "tts", f"tts_{idx:04d}_*.wav")):
                try:
                    os.remove(_f)
                except Exception:
                    pass
            for _f in _g.glob(os.path.join(cm.cache_dir, "tts", f"mixed_{idx:04d}_*.wav")):
                try:
                    os.remove(_f)
                except Exception:
                    pass
            # 同时清理相似度缓存
            _sim = cm.get_path(Step.TTS, ".similarity_cache.json")
            if os.path.exists(_sim):
                try:
                    with open(_sim) as _sf:
                        _sc = json.load(_sf)
                    _sc.pop(str(idx), None)
                    with open(_sim, 'w') as _sf:
                        json.dump(_sc, _sf)
                except Exception:
                    pass
            # 状态始终重置为等待中
            self._subtitle_model.set_status(row, "pending")
            self.tts_paths.pop(row, None)
            self.raw_tts_paths.pop(row, None)
            self._tts_cache_hits.pop(idx, None)

    # ── 波形对比 ──
    def log(self, msg: str):
        """统一日志入口（薄包装,委托给 Python logging 模块)

        - 空 msg 仅插入空白行（UI 分隔线)
        """
        if not msg:
            self.log_text.append("")
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            return
        logging.getLogger(__name__).info(msg)

    def log_file(self, msg: str):
        """仅写入日志文件,不显示到 UI 控件（委托给 DEBUG 级别)"""
        if not msg:
            return
        logging.getLogger(__name__).debug(msg)
