"""TTS API 设置对话框"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QLabel, QGroupBox, QDialogButtonBox,
    QTabWidget, QWidget, QTextEdit, QCheckBox,
)


class SettingsDialog(QDialog):
    """TTS API 和 Demucs 设置对话框"""

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self.setWindowTitle("配音设置")
        self.setMinimumWidth(550)
        self._setup_ui()
        self.load_from_cfg()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ===== Tab 1: TTS API 设置 =====
        tts_tab = QWidget()
        self.tabs.addTab(tts_tab, "TTS API 设置")
        tts_layout = QVBoxLayout(tts_tab)

        # 预设选择
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("预设方案:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("手动配置")
        self.preset_combo.addItem("雨落版 dots.tts")
        self.preset_combo.addItem("CosyVoice (本地)")
        self.preset_combo.addItem("GPT-SoVITS")
        self.preset_combo.addItem("OpenAI TTS")
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo, 1)
        tts_layout.addLayout(preset_layout)

        # API 设置表单
        form = QFormLayout()

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("可选")
        form.addRow("API 密钥:", self.api_key_edit)

        self.api_mode_combo = QComboBox()
        self.api_mode_combo.addItems(["rainfall", "cosyvoice", "gpt-sovits", "openai", "custom"])
        form.addRow("API 模式:", self.api_mode_combo)

        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("如 Fun-CosyVoice3-0.5B")
        form.addRow("模型名:", self.model_name_edit)

        self.language_combo = QComboBox()
        self.language_combo.addItems(["auto", "zh", "en", "ja", "ko", "es", "fr", "de", "ru"])
        form.addRow("输出语言:", self.language_combo)
        lang_hint = QLabel("参考音频为日语时选 zh,为中文时选 ja（输入日语→输出中文)")
        lang_hint.setStyleSheet("color: #666666; font-size: 11px; margin-left: 80px;")
        tts_layout.addWidget(lang_hint)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(10, 600)
        self.timeout_spin.setSuffix(" 秒")
        form.addRow("超时:", self.timeout_spin)

        tts_layout.addLayout(form)

        # 额外参数
        tts_layout.addWidget(QLabel("额外 JSON 参数 (可选):"))
        self.extra_params_edit = QTextEdit()
        self.extra_params_edit.setPlaceholderText('{"temperature": 0.7, "top_p": 0.9}')
        self.extra_params_edit.setMaximumHeight(80)
        tts_layout.addWidget(self.extra_params_edit)

# ── 提示音模式 ──
        prompt_group = QGroupBox("固定提示音设置")
        prompt_layout = QVBoxLayout(prompt_group)


        fixed_text_m = QHBoxLayout()
        fixed_text_m.addSpacing(20)
        fixed_text_m.addWidget(QLabel("男声文本:"))
        self.fixed_ref_text_male = QLineEdit()
        self.fixed_ref_text_male.setPlaceholderText("男声提示音频的文字内容...")
        fixed_text_m.addWidget(self.fixed_ref_text_male)
        prompt_layout.addLayout(fixed_text_m)


        fixed_text_f = QHBoxLayout()
        fixed_text_f.addSpacing(20)
        fixed_text_f.addWidget(QLabel("女声文本:"))
        self.fixed_ref_text_female = QLineEdit()
        self.fixed_ref_text_female.setPlaceholderText("女声提示音频的文字内容...")
        fixed_text_f.addWidget(self.fixed_ref_text_female)
        prompt_layout.addLayout(fixed_text_f)

        # 是否发送提示文本到 API
        self.send_prompt_text_cb = QCheckBox("发送提示文本到 API")
        self.send_prompt_text_cb.setChecked(False)
        self.send_prompt_text_cb.setToolTip("取消勾选则只传参考音频,不传参考文本（部分模型可改善效果)")
        prompt_layout.addWidget(self.send_prompt_text_cb)


        tts_layout.addWidget(prompt_group)
        tts_layout.addStretch()

        # ===== Tab 2: 音频设置 =====
        audio_tab = QWidget()
        self.tabs.addTab(audio_tab, "音频设置")
        audio_layout = QVBoxLayout(audio_tab)

        # ── Demucs 设置 ──
        demucs_group = QGroupBox("Demucs")
        demucs_form = QFormLayout(demucs_group)

        self.demucs_model_combo = QComboBox()
        self.demucs_model_combo.addItems(["htdemucs", "htdemucs_ft", "htdemucs_6s"])
        demucs_form.addRow("模型:", self.demucs_model_combo)

        self.demucs_threads_spin = QSpinBox()
        self.demucs_threads_spin.setRange(1, 64)
        self.demucs_threads_spin.setSuffix(" 线程")
        demucs_form.addRow("CPU 线程:", self.demucs_threads_spin)

        self.demucs_segment_spin = QSpinBox()
        self.demucs_segment_spin.setRange(1, 7)
        self.demucs_segment_spin.setSuffix(" 秒")
        self.demucs_segment_spin.setToolTip("Demucs 内部处理窗口大小,htdemucs 最大 7 秒")
        demucs_form.addRow("处理片段:", self.demucs_segment_spin)

        self.demucs_overlap_spin = QDoubleSpinBox()
        self.demucs_overlap_spin.setRange(0.0, 1.0)
        self.demucs_overlap_spin.setSingleStep(0.05)
        self.demucs_overlap_spin.setToolTip("Demucs 片段间的重叠比例")
        demucs_form.addRow("片段重叠:", self.demucs_overlap_spin)

        audio_layout.addWidget(demucs_group)

        # ── TTS 音频处理 ──
        tts_audio_group = QGroupBox("TTS 音频处理")
        tts_audio_form = QFormLayout(tts_audio_group)

        self.vad_mode_combo = QComboBox()
        self.vad_mode_combo.addItems(["字幕对齐", "原声对齐"])
        self.vad_mode_combo.setToolTip("字幕对齐: 固定留白时长 / 原声对齐: 参考原声静音长度补充留白")
        tts_audio_form.addRow("VAD模式:", self.vad_mode_combo)

        self.vad_pad_spin = QSpinBox()
        self.vad_pad_spin.setRange(0, 500)
        self.vad_pad_spin.setSuffix(" ms")
        self.vad_pad_spin.setToolTip("字幕对齐模式下固定补充的前导静音时长")
        tts_audio_form.addRow("TTS 前导静音:", self.vad_pad_spin)

        self.edge_ms_spin = QSpinBox()
        self.edge_ms_spin.setRange(0, 500)
        self.edge_ms_spin.setSuffix(" ms")
        self.edge_ms_spin.setToolTip("扩展提取区间 ms。0=不扩展,自动避免与相邻字幕重叠")
        tts_audio_form.addRow("TTS 边缘噪声抑制:", self.edge_ms_spin)

        self.similarity_threshold_spin = QDoubleSpinBox()
        self.similarity_threshold_spin.setRange(0.0, 1.0)
        self.similarity_threshold_spin.setSingleStep(0.05)
        self.similarity_threshold_spin.setToolTip("TTS 声纹相似度低于此值时用固定提示音重试一次")
        tts_audio_form.addRow("声纹阈值:", self.similarity_threshold_spin)

        audio_layout.addWidget(tts_audio_group)

        # ── 音量 ──
        volume_group = QGroupBox("音量")
        volume_form = QFormLayout(volume_group)

        self.vocal_volume_spin = QDoubleSpinBox()
        self.vocal_volume_spin.setRange(0.0, 5.0)
        self.vocal_volume_spin.setSingleStep(0.1)
        volume_form.addRow("人声音量:", self.vocal_volume_spin)

        self.bg_volume_spin = QDoubleSpinBox()
        self.bg_volume_spin.setRange(0.0, 5.0)
        self.bg_volume_spin.setSingleStep(0.1)
        volume_form.addRow("背景音量:", self.bg_volume_spin)

        audio_layout.addWidget(volume_group)

        # ── 字幕校准 ──
        calib_group = QGroupBox("字幕校准")
        calib_form = QFormLayout(calib_group)

        self.align_mode_combo = QComboBox()
        self.align_mode_combo.addItems(["qwen", "whisper", "sensevoice"])
        self.align_mode_combo.setToolTip(
            "第3步区间矫正方式:\n"
            "qwen = Qwen3 强制对齐 (需原声字幕, 精确)\n"
            "whisper = faster-whisper 转写对齐 (无需原声字幕, 词级时间戳)\n"
            "sensevoice = SenseVoiceSmall ASR + VAD 对齐 (无需原声字幕)")
        calib_form.addRow("对齐方式:", self.align_mode_combo)

        self.asr_max_pad_spin = QSpinBox()
        self.asr_max_pad_spin.setRange(0, 1000)
        self.asr_max_pad_spin.setSuffix(" ms")
        self.asr_max_pad_spin.setToolTip("Qwen 送检音频单侧最大扩展毫秒数")
        calib_form.addRow("最大扩展:", self.asr_max_pad_spin)

        self.asr_safe_gap_spin = QSpinBox()
        self.asr_safe_gap_spin.setRange(0, 1000)
        self.asr_safe_gap_spin.setSuffix(" ms")
        self.asr_safe_gap_spin.setToolTip("两字幕间隔小于此值时不做扩展")
        calib_form.addRow("安全间隔:", self.asr_safe_gap_spin)

        self.asr_pad_ms_spin = QSpinBox()
        self.asr_pad_ms_spin.setRange(0, 500)
        self.asr_pad_ms_spin.setSuffix(" ms")
        self.asr_pad_ms_spin.setToolTip("Qwen 对齐结果前后各加的安全区")
        calib_form.addRow("安全区:", self.asr_pad_ms_spin)

        audio_layout.addWidget(calib_group)

        # ── 线程数 ──
        thread_group = QGroupBox("线程数")
        thread_layout = QFormLayout(thread_group)
        self.tts_threads_spin = QSpinBox()
        self.tts_threads_spin.setRange(1, 16)
        thread_layout.addRow("TTS 线程:", self.tts_threads_spin)
        self.mix_threads_spin = QSpinBox()
        self.mix_threads_spin.setRange(1, 16)
        thread_layout.addRow("混音线程:", self.mix_threads_spin)
        self.aligner_threads_spin = QSpinBox()
        self.aligner_threads_spin.setRange(1, 16)
        thread_layout.addRow("Step3 提取线程:", self.aligner_threads_spin)

        # 批处理大小
        self.gender_batch_spin = QSpinBox()
        self.gender_batch_spin.setRange(1, 64)
        self.gender_batch_spin.setToolTip("WavLM 性别检测每批并行处理条数,越大显存占用越高")
        thread_layout.addRow("性别批大小:", self.gender_batch_spin)

        self.qwen_batch_spin = QSpinBox()
        self.qwen_batch_spin.setRange(1, 128)
        self.qwen_batch_spin.setToolTip("Qwen 对齐每批并行处理条数,越大显存占用越高")
        thread_layout.addRow("Qwen批大小:", self.qwen_batch_spin)

        audio_layout.addWidget(thread_group)
        audio_layout.addStretch()

        # ===== 确定/取消按钮 =====
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self):
        """确定前校验额外 JSON 参数,失败则提示并不关闭对话框"""
        raw = self.extra_params_edit.toPlainText().strip()
        if raw:
            try:
                import json
                json.loads(raw)
            except json.JSONDecodeError as e:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self, "JSON 格式错误",
                    f"额外 JSON 参数解析失败:\n{e}\n\n请修正后再保存,或清空该输入框。"
                )
                return
        self.accept()

    def load_from_cfg(self):
        """从 cfg 加载全部字段到控件"""
        # TTS API
        self.api_key_edit.setText(self._cfg.tts_api_key)
        self.api_mode_combo.setCurrentText(self._cfg.tts_mode)
        self.model_name_edit.setText(self._cfg.tts_model)
        self.language_combo.setCurrentText(self._cfg.tts_language)
        self.timeout_spin.setValue(self._cfg.tts_timeout)
        extra = self._cfg.tts_extra_params
        if extra:
            import json
            self.extra_params_edit.setPlainText(json.dumps(extra, ensure_ascii=False, indent=2))
        else:
            self.extra_params_edit.clear()
        self.fixed_ref_text_male.setText(self._cfg.fixed_ref_text_male)
        self.fixed_ref_text_female.setText(self._cfg.fixed_ref_text_female)
        self.send_prompt_text_cb.setChecked(self._cfg.send_prompt_text)

        # 音频设置
        self.demucs_model_combo.setCurrentText(self._cfg.demucs_model)
        self.demucs_threads_spin.setValue(self._cfg.demucs_threads)
        self.demucs_segment_spin.setValue(self._cfg.demucs_segment)
        self.demucs_overlap_spin.setValue(self._cfg.demucs_overlap)

        self.vad_mode_combo.setCurrentText(self._cfg.vad_mode)
        self.vad_pad_spin.setValue(self._cfg.vad_pad_ms)
        self.edge_ms_spin.setValue(self._cfg.edge_ms)
        self.similarity_threshold_spin.setValue(self._cfg.tts_similarity_threshold)

        self.vocal_volume_spin.setValue(self._cfg.vocal_volume)
        self.bg_volume_spin.setValue(self._cfg.bg_volume)

        # 字幕校准
        self.align_mode_combo.setCurrentText(self._cfg.align_mode)
        self.asr_max_pad_spin.setValue(self._cfg.asr_max_pad)
        self.asr_safe_gap_spin.setValue(self._cfg.asr_safe_gap)
        self.asr_pad_ms_spin.setValue(self._cfg.asr_pad_ms)

        # 线程数
        self.tts_threads_spin.setValue(self._cfg.tts_threads)
        self.mix_threads_spin.setValue(self._cfg.mix_threads)
        self.aligner_threads_spin.setValue(self._cfg.qwen_aligner_threads)
        self.gender_batch_spin.setValue(self._cfg.gender_batch_size)
        self.qwen_batch_spin.setValue(self._cfg.qwen_batch_size)

    def _on_preset_changed(self, preset: str):
        """预设方案切换"""
        presets = {
            "雨落版 dots.tts": {
                "mode": "rainfall",
                "model": "",
                "language": "zh",
            },
            "CosyVoice (本地)": {
                # "url": "http://localhost:5000/api/tts",
                "mode": "cosyvoice",
                "model": "Fun-CosyVoice3-0.5B",
            },
            "GPT-SoVITS": {
                # "url": "http://localhost:9880/tts",
                "mode": "gpt-sovits",
                "model": "",
            },
            "OpenAI TTS": {
                # "url": "https://api.openai.com/v1/audio/speech",
                "mode": "openai",
                "model": "tts-1",
            },
        }
        if preset in presets:
            p = presets[preset]
            self.api_mode_combo.setCurrentText(p["mode"])
            self.model_name_edit.setText(p["model"])
            if "language" in p:
                self.language_combo.setCurrentText(p["language"])


    def get_settings(self) -> dict:
        """获取设置值"""
        extra_params = {}
        raw = self.extra_params_edit.toPlainText().strip()
        if raw:
            try:
                import json
                extra_params = json.loads(raw)
            except json.JSONDecodeError:
                pass

        return {
            "tts_api_key": self.api_key_edit.text().strip(),
            "tts_mode": self.api_mode_combo.currentText(),
            "tts_model": self.model_name_edit.text().strip(),
            "tts_language": self.language_combo.currentText(),
            "tts_timeout": self.timeout_spin.value(),
            "tts_extra_params": extra_params,
            "fixed_ref_text_male": self.fixed_ref_text_male.text().strip(),
            "fixed_ref_text_female": self.fixed_ref_text_female.text().strip(),
            "send_prompt_text": self.send_prompt_text_cb.isChecked(),

            "demucs_model": self.demucs_model_combo.currentText(),
            "demucs_threads": self.demucs_threads_spin.value(),
            "demucs_segment": self.demucs_segment_spin.value(),
            "demucs_overlap": self.demucs_overlap_spin.value(),
            "tts_threads": self.tts_threads_spin.value(),
            "mix_threads": self.mix_threads_spin.value(),
            "qwen_aligner_threads": self.aligner_threads_spin.value(),
            "gender_batch_size": self.gender_batch_spin.value(),
            "qwen_batch_size": self.qwen_batch_spin.value(),
            "vad_pad_ms": self.vad_pad_spin.value(),
            "edge_ms": self.edge_ms_spin.value(),
            "tts_similarity_threshold": self.similarity_threshold_spin.value(),
            "vad_mode": self.vad_mode_combo.currentText(),
            "vocal_volume": self.vocal_volume_spin.value(),
            "bg_volume": self.bg_volume_spin.value(),
            "asr_max_pad": self.asr_max_pad_spin.value(),
            "asr_safe_gap": self.asr_safe_gap_spin.value(),
            "asr_pad_ms": self.asr_pad_ms_spin.value(),
            "align_mode": self.align_mode_combo.currentText(),
        }
