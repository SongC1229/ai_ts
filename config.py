"""配置管理 — 集中存放所有默认值,全局共用同一实例"""


class Cfg:
    """所有可调设置,建议全局共用同一实例"""

    def __init__(self):
        # ── TTS ──
        self.tts_api_url = "http://localhost:9001"
        self.tts_api_key = ""
        self.tts_mode = "rainfall"
        self.tts_model = ""
        self.tts_language = "zh"
        self.tts_timeout = 60
        self.tts_extra_params = {}
        self.tts_similarity_threshold = 0.6
        self.tts_threads = 1
        self.use_local_tts = True
        self.tts_local_mode = "indextts"   # indextts | dots
        self.use_fixed_ref = False
        self.send_prompt_text = False

        # ── dots.tts 引擎参数 ──
        self.dots_num_steps = 10
        self.dots_guidance_scale = 1.2
        self.dots_speaker_scale = 1.5
        self.dots_precision = "bfloat16"   # bfloat16 | float16

        # ── Demucs ──
        self.demucs_model = "htdemucs"
        self.demucs_device = "auto"
        self.demucs_threads = 4
        self.demucs_segment = 7
        self.demucs_overlap = 0.25

        # ── VAD / 音频处理 ──
        self.vad_mode = "原声对齐"
        self.vad_pad_ms = 50
        self.edge_ms = 100

        # ── 音量 ──
        self.vocal_volume = 1.0
        self.bg_volume = 1.0

        # ── Qwen 校准 ──
        self.asr_max_pad = 500
        self.asr_safe_gap = 200
        self.asr_pad_ms = 100
        self.qwen_aligner_url = "http://localhost:8765"
        self.qwen_batch_size = 8

        # ── 字幕区间对齐方式 (第3步) ──
        self.align_mode = "qwen"   # qwen | whisper

        # ── Whisper 转写对齐参数 (align_mode=whisper 时生效) ──
        # Demucs 提纯人声无背景音, VAD 阈值低于默认 0.5 避免耳语/气声被滤掉
        self.whisper_vad_filter = True          # VAD 预过滤 (True=开启)
        self.whisper_vad_threshold = 0.4        # 语音概率阈值 (默认 0.5)
        self.whisper_vad_min_silence_ms = 500    # 静音持续多久才算分段 (默认 2000)
        self.whisper_vad_speech_pad_ms = 100     # 语音段前后补白 (默认 400)
        self.whisper_beam_size = 5               # beam search 宽度

        # ── 性别检测 ──
        self.gender_detect_mode = "wavlm"   # wavlm | gender_cls
        self.gender_batch_size = 6

        # ── 混音 ──
        self.mix_threads = 6
        self.qwen_aligner_threads = 4

        # ── 参考音频 ──
        self.fixed_ref_audio_male = ""
        self.fixed_ref_audio_female = ""
        self.fixed_ref_text_male = ""
        self.fixed_ref_text_female = ""


# 全局唯一实例,各模块引用同一份
cfg = Cfg()
