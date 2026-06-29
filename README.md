# 电影 AI 配音工具

基于 PySide6 的桌面 GUI 工具，用于电影 AI 翻译配音。将视频中的人声替换为 AI 合成的目标语言语音，
同时保留背景音乐和音效。非说话区间 100% 保留原始音频（含原声、背景音乐、环境音效），
TTS 片段与原始音频在拼接点做交叉淡变平滑过渡。

---

## 依赖

```bash
# 1. 安装基础依赖
pip install -r requirements.txt

# 2. 安装 Demucs（人声分离）
pip install demucs

# 3. 安装 ffmpeg
# Windows: winget install ffmpeg  或从 https://ffmpeg.org/download.html 下载
# macOS:   brew install ffmpeg
# Linux:   sudo apt install ffmpeg

# 4. 安装 ML 模型依赖
pip install transformers torch soundfile loralib qwen_asr

# 5. 安装 Resemblyzer（声纹相似度检测，需 PyTorch）
pip install resemblyzer

# 6. 安装 uv（Qwen 服务独立启动需要）
pip install uv

# 7. 下载模型权重
#     本地方案：使用本地 IndexTTS2 引擎 + Qwen 强制对齐
#     IndexTTS2:
#       modelscope download --model IndexTeam/IndexTTS-2 --local_dir indextts2_src/checkpoints
#       pip install descript-audiotools scipy
#     WavLM / Qwen 模型权重已托管在 models/ 目录下

# 8. Qwen 强制对齐服务（流水线自动启动，无需手动）
#     本地 TTS 模式自动用 uv run 启动，API 模式用系统 Python 启动

# 9. TTS 方案选择
#     A) 使用本地 IndexTTS2 引擎（GUI 勾选"本地引擎"）
#     B) 部署 TTS API 服务（任选其一）：
#        - CosyVoice:   https://github.com/FunAudioLLM/CosyVoice
#        - GPT-SoVITS:  https://github.com/RVC-Boss/GPT-SoVITS
#        - dots.tts:    https://github.com/rednote-hilab/dots.tts
#        - OpenAI TTS:  兼容 OpenAI 格式的服务
```

---

## 架构总览

### 流水线引擎

流水线采用 **BaseStep 解耦分步架构**，每步独立实现 `run()` 方法，遵循统一生命周期：

```
缓存检查(check_cache) → 依赖检查(dependencies) → 执行(run) → 标记完成(mark_completed)
```

6 个 Step 由 `PipelineOrchestrator` 按顺序调度：

```python
PipelineOrchestrator.STEPS = [
    ExtractAudioStep(),      # Step 1: ffmpeg 提取音频
    SeparateVocalsStep(),    # Step 2: Demucs 人声分离
    SubStep(),              # Step 3: 字幕解析 + Qwen对齐 + WavLM性别检测
    TTSSynthesisStep(),      # Step 4: TTS 合成
    AudioMixAndMergeStep(),  # Step 5: 混音 + 全长拼接
    VideoMergeStep(),        # Step 6: 合并回视频
]
```

**BaseStep 基类**（`core/pipeline.py`）：

| 属性/方法 | 说明 |
|---|---|
| `name` | 步骤显示名 |
| `step_index` | 步骤序号（0-based） |
| `cache_key` | CacheManager 中的缓存键 |
| `dependencies` | 前置步骤 cache_key 列表 |
| `check_cache(ctx)` → `CacheStatus` | 缓存状态：FULL / PARTIAL / NONE |
| `get_target_files(ctx)` → `list[str]` | 本步骤产出的目标文件（用于文件级缓存检查） |
| `run(ctx, progress_cb)` | 执行本步骤 |

**PipelineContext** 是流水线共享上下文，携带所有输入路径、运行时状态和配置参数。各 Step 通过 `ctx` 读写中间结果，而非直接依赖。

**PipelineOrchestrator** 支持两种运行模式：
- `run()` — 完整流水线，按序执行全部 6 步
- `run_single_step(step_index)` — 单步执行，用于断点续传/重试

**独立可复用函数**（模块级，供流水线、单条重生成、重新混音共用）：
- `synthesize_tts_segment()` — 合成单条 TTS
- `mix_tts_segment()` — 对单条 TTS 做 VAD 修剪 + 前导对齐 + 增益匹配 + 背景混音
- `regen_single_tts()` — 单条 TTS 重新生成（TTS + 混音全流程）
- `remix_from_cache()` — 从缓存重新拼接音频

### 配置系统

`core/config.py` 中的 `Cfg` 类集中管理所有可调参数的默认值，全局共用同一实例 `default_cfg`：

```python
class Cfg:
    # ── TTS ──
    tts_api_url = "http://localhost:9001"     # TTS 服务地址
    tts_api_key = ""                          # API 密钥
    tts_mode = "rainfall"                     # API 模式 (rainfall/cosyvoice/gpt-sovits/openai/custom)
    tts_model = ""                            # 模型名
    tts_language = "zh"                       # 目标语言
    tts_timeout = 60                          # API 超时（秒）
    tts_extra_params = {}                     # 额外请求参数
    tts_similarity_threshold = 0.6            # 声纹相似度阈值
    tts_threads = 1                           # TTS 合成并行线程数（上限 4）
    use_local_tts = True                      # 使用本地 IndexTTS2 引擎
    use_fixed_ref = False                     # 使用固定提示音替代原声参考
    send_prompt_text = False                  # 发送提示文本（部分 API 需要）
    ref_gender = "female"                     # 默认参考性别
    fixed_ref_audio_male = ""                 # 固定男声参考音频路径
    fixed_ref_audio_female = ""               # 固定女声参考音频路径
    fixed_ref_text_male = ""                  # 固定男声参考文本
    fixed_ref_text_female = ""                # 固定女声参考文本

    # ── Demucs ──
    demucs_model = "htdemucs"                 # 模型 (htdemucs/htdemucs_ft/htdemucs_6s)
    demucs_device = "auto"                    # 设备 (auto/cpu/cuda/mps)
    demucs_threads = 4                        # CPU 线程数
    demucs_segment = 7                        # 内部窗口大小（秒，htdemucs 最大 7）
    demucs_overlap = 0.25                     # 片段重叠比例

    # ── VAD / 音频处理 ──
    vad_mode = "原声对齐"                      # 前导对齐模式 (原声对齐/字幕对齐)
    vad_pad_ms = 50                           # 字幕对齐模式固定留白（毫秒）
    edge_ms = 100                             # 混音边界扩展（毫秒）

    # ── 音量 ──
    vocal_volume = 1.0                        # 人声音量
    bg_volume = 1.0                           # 背景音量

    # ── Qwen 校准 ──
    asr_max_pad = 500                        # 送检音频最大单侧 padding（毫秒）
    asr_safe_gap = 200                       # 两字幕间安全间距（毫秒）
    asr_pad_ms = 100                         # Qwen 结果安全区（毫秒）
    qwen_aligner_url = "http://localhost:8765" # Qwen API 服务地址
    qwen_batch_size = 8                       # Qwen 批处理大小
    qwen_aligner_threads = 4                  # Qwen 音频片段并行提取线程数

    # ── 性别检测 ──
    gender_batch_size = 6                     # WavLM 批处理大小

    # ── 混音 ──
    mix_threads = 6                           # 混音并行线程数
```

`PipelineContext.__init__()` 通过 `default_cfg.__dict__` 批量拷贝所有配置字段，UI 层修改 `default_cfg` 即可影响流水线。

### 缓存系统

`CacheManager`（`core/cache_manager.py`）基于视频文件 SHA256 指纹（头尾 64KB + 文件大小 + mtime）生成缓存目录：

```
.cache/{video_stem}.{hash_4char}/
├── .cache_info                    ← 各 step 完成状态 + 元数据（JSON）
├── extract/
│   └── mix_orig.wav               ← ffmpeg 提取的原始混合音频
├── demucs/
│   ├── vocals_orig.wav            ← 人声（单声道）
│   └── background.wav             ← 背景音（立体声）
├── subs/
│   ├── genders_cache.json         ← 性别检测缓存 {"0": "female", "1": "male", ...}
│   └── calib.json                 ← Qwen/Whisper 校准后的字幕时间（含目标+原声）
├── tts/
│   ├── .similarity_cache.json     ← 声纹相似度缓存（按 TTS 文件 mtime+size 失效）
│   ├── _ref_{idx}_{s}-{e}.wav    ← 参考音频片段（从人声截取）
│   ├── tts_{idx}_{s}-{e}.wav     ← TTS 原始合成
│   └── mixed_{idx}_{s}-{e}.wav   ← TTS+背景混音（最终片段）
└── mix/
    └── final_audio.wav            ← 全长拼接最终音频
```

**缓存策略**：
- 每步开始前调用 `check_cache()` → 文件级检查（`get_target_files`）+ 标记级检查（`.cache_info`）
- 同名同路径视频自动跳过已完成 step
- 清空某 step 会自动清空其后所有 step（依赖链）
- 部分缓存命中（PARTIAL）时自动跳过已完成条目
- 声纹相似度缓存按 TTS 文件的 mtime+size 失效

**CacheManager 快捷方法**：

| 方法 | 说明 |
|---|---|
| `get_path(step, filename)` | 通用路径获取（自动创建目录） |
| `get_tts_path(idx, start_ms, end_ms)` | TTS 文件快捷路径 |
| `get_mixed_path(idx, start_ms, end_ms)` | 混音文件快捷路径 |
| `vocals_path` / `bg_path` / `mix_orig_path` | 常用路径属性 |
| `load_gender_cache()` / `save_gender_cache()` | 性别缓存读写 |
| `restore_calib_subs()` | 从缓存恢复字幕列表 |
| `scan_tts_cache()` | 扫描 TTS 目录返回 `{idx: path}` |
| `clear_step(step_name)` | 清空指定 step 缓存 |

### TTSConfig

`TTSConfig`（`core/pipeline.py`）是 TTS 合成配置的统一 dataclass，消除多处重复构建：

```python
@dataclass
class TTSConfig:
    use_local_tts: bool = False       # 本地 IndexTTS2 引擎
    api_url: str = "..."              # API 地址
    api_key: str = ""                 # API 密钥
    mode: str = "rainfall"            # API 模式
    model_name: str = ""              # 模型名
    language: str = "zh"              # 目标语言
    timeout: int = 120                # 超时（秒）
    extra_params: dict = {}           # 额外参数
    send_prompt_text: bool = False    # 发送提示文本
    use_fixed_ref: bool = False       # 使用固定提示音
    fixed_ref_audio_male: str = ""    # 固定男声参考
    fixed_ref_audio_female: str = ""  # 固定女声参考
    vocals_path: str = ""             # 人声轨路径（用于提取参考片段）

    @classmethod
    def from_ctx(cls, ctx)            # 从 PipelineContext 提取
    @classmethod
    def from_dict(cls, d: dict)       # 从 settings dict 构建（用于单条重生成）
```

### 工具函数

`core/utils.py` 集中存放各模块复用的工具函数：

| 函数 | 说明 |
|---|---|
| `_ms_parts(ms)` | 毫秒拆解为 `(h, m, s, ms3)` |
| `fmt_ts(ms, sep)` | 毫秒 → `HH_MM_SS_MS`（文件名用） |
| `fmt_time(ms, ms_sep)` | 毫秒 → `HH:MM:SS.mmm`（日志/UI 用） |
| `fmt_time_adaptive(ms, total_ms)` | 智能格式化：短音频秒数，长音频时分秒 |
| `read_wav_np(path, mono)` | WAV → float64 numpy（优先 wave 模块，回退 soundfile） |
| `write_wav_np(path, y, sr)` | float64 numpy → 16-bit WAV |
| `read_wav_segment(path, start_ms, end_ms)` | seek 读取区间音频（避免加载全长） |
| `downsample_waveform(y, dur_ms, target_n)` | 音频降采样到 target_n 个峰值柱 |
| `tts_filename(idx, s, e)` | TTS 文件名生成 |
| `mixed_filename(idx, s, e)` | 混音文件名生成 |
| `resolve_device(device)` | 统一设备检测（cuda > cpu，首次缓存） |
| `cleanup_cuda()` | GPU 显存清理（gc + torch.cuda.empty_cache） |
| `make_logger(log_cb)` | 创建安全日志函数（None → 静默） |
| `get_threads(ctx, attr, max)` | 从 ctx 获取线程数（未配置 → 1 + 警告） |

---

## 流水线详细说明

### Step 1 — 提取音频（`ExtractAudioStep`）

| 输入 | 输出 | 缓存路径 | 缓存键 |
|---|---|---|---|
| 视频文件 | `mix_orig.wav` | `extract/mix_orig.wav` | `extract` |

**逻辑**：
1. 缓存检查：文件存在则跳过，同时恢复 `ctx.sample_rate`
2. 自动探测原始音频采样率（上限 48kHz），通过 `get_audio_info()` 一次 ffprobe 获取
3. ffmpeg 提取：`-vn -acodec pcm_s16le -ar {sr} -ac 2`
4. 采样率写入 `.cache_info` 元数据（`set_meta("sample_rate", sr)`）

**关键参数**：
- `ctx.sample_rate` — 采样率（自动探测，上限 48kHz，探测失败默认 48kHz）

### Step 2 — Demucs 人声分离（`SeparateVocalsStep`）

| 输入 | 输出 | 缓存路径 | 缓存键 |
|---|---|---|---|
| `mix_orig.wav` | `vocals_orig.wav` + `background.wav` | `demucs/` | `demucs` |

**逻辑**：
1. 缓存检查：两个文件都存在则跳过
2. 调用 `voice_separator.separate_vocals()`：
   - 正常音频（≤1h）：直接运行 Demucs
   - 超长音频（>1h）：自动分片（1h/片 + 2s 重叠），逐片 Demucs 后交叉淡变拼接
   - 内部使用 tqdm monkey-patch 获取进度
   - monkey-patch `demucs.audio.save_audio` 和 `torchaudio.save` 使用 soundfile 替代
3. 人声轨转单声道（`ffmpeg -ac 1`）
4. 背景轨直接复制
5. 清理 Demucs 中间输出目录
6. `cleanup_cuda()` 释放显存

**关键参数**：
- `ctx.demucs_model` — 模型名（htdemucs / htdemucs_ft / htdemucs_6s）
- `ctx.demucs_device` — 设备（auto / cpu / cuda / mps）
- `ctx.demucs_threads` — CPU 线程数（默认 4）
- `ctx.demucs_segment` — 内部窗口（秒，htdemucs 最大 7，默认 7）
- `ctx.demucs_overlap` — 片段重叠（默认 0.25）

**进度映射**：`(0/2)` Demucs → `(1/2)` 人声转单声道 → `(2/2)` 背景复制

### Step 3 — 字幕处理（`SubStep`）

| 输入 | 输出 | 缓存路径 | 缓存键 |
|---|---|---|---|
| SRT 字幕 + `vocals_orig.wav` | `genders_cache.json` + `calib.json` | `subs/` | `subs` |

**两条处理路径**：

**路径 A：原声字幕对齐（提供 `raw_src_path` 时）**

1. **Qwen 强制对齐**（`core/qwen_aligner.py`）
   - 自动启动 `qwen_api_server.py` 子进程（REST API，端口 8765）
   - 本地 TTS 模式用 `uv run python` 启动，API 模式用系统 Python 直接启动
   - 主程序退出时自动关闭服务
   - Phase 1：并行提取音频片段（`qwen_aligner_threads` 线程，soundfile seek）
     - 动态 padding：两字幕间隔 ≥200ms 才做 padding，单侧最大 500ms
     - 按时长排序分桶减少 batch padding 浪费
   - Phase 2：批量 Qwen 对齐（`qwen_batch_size` 条/批）
     - 头部裁剪 + `asr_pad_ms` 安全区
     - 尾部仅允许扩展（禁止裁剪原始结束时间）
     - 语速校验：<15 字符/秒才接受对齐结果
   - 对齐完成后立即 `unload_model()` 释放显存
   - 校准结果先写入 `calib.json`（防中断丢失）

2. **WavLM 性别检测**（`core/wavlm_gender_api.py`）
   - 使用对齐后的时间裁剪音频片段
   - Phase 1：并行提取音频片段（同上）
   - Phase 2：批量 WavLM 推理（`gender_batch_size` 条/批）
     - 置信度 ≥75% 写入缓存，<75% 置空（UI 显示供人工确认）
   - 完成后 `unload_model()` 释放显存

**路径 B：直接使用字幕时间（无原声字幕时）**

- 跳过 Qwen 对齐，直接 SRT 时间 + 性别检测

**Qwen 对齐参数**：
- `ctx.asr_max_pad` — 送检音频最大单侧 padding（默认 500ms）
- `ctx.asr_safe_gap` — 两字幕间安全间距（默认 200ms）
- `ctx.asr_pad_ms` — Qwen 结果前后安全区（默认 100ms）
- `ctx.qwen_batch_size` — 每批条数（默认 8）
- `ctx.qwen_aligner_threads` — 音频提取并行线程数（默认 4）

### Step 4 — TTS 合成（`TTSSynthesisStep`）

| 输入 | 输出 | 缓存路径 | 缓存键 |
|---|---|---|---|
| 字幕文本 + 参考人声 | `tts_{idx}_{s}-{e}.wav` | `tts/` | `tts` |

**三轮流程**：

**Phase 1：批量 TTS 合成**
- 预收集需要合成的任务（跳过已缓存、性别未定、括号注释）
- 多端口自动探测：从 `tts_api_url` 的端口号开始，扫描 `tts_threads` 个端口
- `ThreadPoolExecutor` 并行合成，每完成一条回调 `on_item(idx, tts_name)`
- 失败时调用 `ctx.on_tts_error_cb(failed_items)` 弹出非模态对话框（重试/跳过/终止）
- 参考音频优先级：固定提示音 > 原声参考片段（从 vocals 截取，缓存为 `_ref_{idx}_{s}-{e}.wav`）

**Phase 2：批量声纹校验**（`tts_similarity_threshold > 0` 时）
- 基于 Resemblyzer 的说话人嵌入余弦相似度
- 结果按 TTS 文件 mtime+size 缓存到 `.similarity_cache.json`
- 4 线程并行推理
- 低于阈值的条目标记为待重试

**Phase 3：固定提示音对比 + 批量重试**
- 先用固定提示音对比 TTS（而非原声），≥0.6 则跳过重试
- 不足的条用固定提示音重新合成
- 重试后清除对应混音缓存（`mixed_*.wav`）

**TTSClient 支持的 5 种 API 模式**：

| 模式 | 协议 | 说明 |
|---|---|---|
| `rainfall` | GET `/api/clone?text=&prompt_path=` | 雨落版，返回 WAV |
| `cosyvoice` | POST JSON `{text, ref_audio(base64)}` | CosyVoice |
| `gpt-sovits` | POST form-data `{ref_audio, text, text_lang}` | GPT-SoVITS |
| `openai` | POST JSON `{model, input, voice}` | OpenAI TTS 兼容 |
| `custom` | POST JSON（用户自定义模板） | 由 extra_params 控制 |

**本地引擎模式**（`use_local_tts=True`）：
- 调用 `core/tts_engine.py` 的 `tts_synthesize()`
- 参考音频读取为 bytes 直接传入（避免额外磁盘写）
- `target_duration_ms` 估算 max_mel_tokens 使生成接近原声时长
- 单例模型（MLModelHolder），首次加载约 25s

**TTSConfig 参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `use_local_tts` | `False` | 使用本地 IndexTTS2 |
| `api_url` | `http://localhost:9001` | TTS API 地址 |
| `api_key` | `""` | API 密钥 |
| `mode` | `rainfall` | API 模式 |
| `model_name` | `""` | 模型名 |
| `language` | `zh` | 目标语言 |
| `timeout` | `120` | 超时（秒） |
| `extra_params` | `{}` | 额外 API 参数 |
| `send_prompt_text` | `False` | 发送参考文本 |
| `use_fixed_ref` | `False` | 使用固定提示音 |
| `tts_threads` | `1` | 并行线程数（上限 4） |
| `tts_similarity_threshold` | `0.6` | 声纹阈值 |

### Step 5 — 全长重建（`AudioMixAndMergeStep`）

每条字幕的处理链（由 `mix_tts_segment()` 统一实现）：

```
TTS 原始合成
  │
  ├── 淡入淡出 ────────── 语音起始前 40ms 线性淡入 + 尾部 100ms 指数淡出
  │                        （写入 .faded.wav 副本，不覆盖缓存）
  │
  ├── 参考音频截取 ────── 从 vocals.wav 截取对应区间供 VAD 使用
  │
  ├── VAD 修剪 ────────── vad_trim_silence()
  │     ▸ 检测 TTS 和参考的前导静音长度
  │     ▸ 仅裁 TTS 比参考多出的前导静音
  │     ▸ 尾部不裁剪（保留完整音频）
  │     ▸ WAV 文件使用 numpy 能量阈值（-35dB, 20ms 帧），无子进程
  │
  ├── 前导对齐 ────────── add_leading_silence()
  │     ▸ "原声对齐"（默认）: 检测原声前导静音长度，补足 TTS 前导到等长
  │     ▸ "字幕对齐": 固定补充 vad_pad_ms 毫秒前导静音
  │
  ├── RMS 增益匹配 ────── match_rms_gain()
  │     ▸ 截取重叠长度，去除静音段，仅在有声部分对比 RMS
  │     ▸ 差值 > 0.3dB 时调整，否则直接复制
  │
  ├── 边界扩展 ────────── pad_audio_np() 头部补充 edge_ms 静音
  │
  └── 背景混音 ────────── mix_segment_clip()
        ▸ TTS(mono) + 背景(stereo/mono) → 直接 numpy 叠加
        ▸ 采样率不一致时 TTS 升采样到背景采样率
        ▸ 长度不一致时较短方补零到较长方长度
        ▸ 立体声背景时 TTS 广播到双声道再叠加
```

**输出文件**：`tts/mixed_{idx}_{start}-{end}.wav`

**多线程**：`mix_threads` 线程并行处理（默认 6）

**关键参数**：
- `ctx.vad_mode` — 前导对齐模式（默认 "原声对齐"）
- `ctx.vad_pad_ms` — 字幕对齐固定留白（默认 50ms）
- `ctx.edge_ms` — 混音边界扩展（默认 100ms）
- `ctx.mix_threads` — 混音线程数（默认 6）

**安全边界计算**（`_safe_edge()`）：
- 检查前后字幕间距，若 < edge_ms×2 则缩小扩展值
- 防止相邻片段的扩展区域重叠

### Step 6 — 合并视频（`VideoMergeStep`）

| 输入 | 输出 | 缓存键 |
|---|---|---|
| `mix_orig.wav` + 所有 `mixed_*.wav` | `{视频名}.ts.mp4` | `mix`（标记级） |

**逻辑**（`splice_segments_into_base()`）：

1. **tts_segments 重建**：从缓存扫描所有 `mixed_*.wav`，构建 `[(start_ms, end_ms, clip_path), ...]`
2. **以原始音频为基底**：加载 `mix_orig.wav`，非说话区间 100% 保留
3. **逐片段替换**：
   - 按起始时间排序，clip 裁剪/填充到区间长度
   - 声道适配：clip 单声道但基底立体声时复制到双声道
   - **段前交叉淡变** `[s-L_in, s+L_in)`：原始淡出 + clip 淡入（线性，默认 40ms）
   - **段后交叉淡变** `[e-L_out, e+L_out)`：clip 淡出 + 原始淡入（线性，默认 40ms）
   - L_in/L_out 自动收缩：`min(L, seg_len//4)`，且不与相邻片段重叠
   - `prev_end` 包含尾淡变区域，防止下一片段头淡变与之重叠
4. **ffmpeg 合并视频**：
   - `-c:v copy`（视频流不变）
   - `-c:a aac -b:a 192k`（音频重编码）
   - `-movflags +faststart`（Web 友好）
   - `-shortest`（取最短流）

---

## 辅助操作

### 单条重新生成

`regen_single_tts()` 实现单条 TTS 的完整重生成流程：
1. TTS 合成（复用 `synthesize_tts_segment`）
2. 声纹校验 + 固定提示音重试
3. VAD 修剪 + 前导对齐 + 增益匹配 + 背景混音（复用 `mix_tts_segment`）
4. 清理临时文件，回调 `done_cb(success, mixed_clip_path)`

### 重新混音

`remix_from_cache()` 从缓存重新拼接音频：
1. 从 `calib.json` 或 SRT 解析字幕时间
2. 扫描所有 `mixed_*.wav` 构建片段列表
3. 调用 `splice_segments_into_base()` 拼接
4. 标记 `mix` step 完成

---

## 音频处理模块

### `core/audio_tools.py`

| 函数 | 说明 |
|---|---|
| `get_audio_info(path, max_rate)` | 获取 AudioInfo(duration_ms, sample_rate, channels)，WAV 直接读头部，其他用 ffprobe |
| `split_audio_np(path, s, e, out)` | soundfile seek 裁剪音频片段 |
| `split_audio_by_time(path, s, e, out, ch)` | ffmpeg 裁剪（保持采样率） |
| `pad_audio_np(in, out, front, back)` | 前后补静音 |
| `vad_detect_speech(path, thresh)` | 检测语音起止位置，WAV 用 numpy，其他用 ffmpeg silencedetect |
| `vad_trim_silence(in, out, thresh, ref, pre)` | VAD 修剪（联合前导静音处理） |
| `add_leading_silence(in, out, ref, mode, margin)` | 前导静音对齐 |
| `match_rms_gain(tts, ref, out)` | RMS 增益匹配 |
| `mix_segment_clip(tts, bg, out, edge)` | TTS + 背景 numpy 叠加混音 |
| `splice_segments_into_base(base, segs, out, crossfade)` | 全长拼接（交叉淡变） |

### `core/srt_parser.py`

| 函数/类 | 说明 |
|---|---|
| `SubtitleItem` | dataclass：index, start_ms, end_ms, text, duration_ms |
| `parse_srt(path)` | 解析 SRT → List[SubtitleItem]，自动去除 HTML/格式标签 |

### `core/voice_separator.py`

| 函数 | 说明 |
|---|---|
| `separate_vocals(audio, out, model, device, ...)` | Demucs 人声分离（含超长音频自动分片） |
| `_run_demucs(...)` | 内部：分片逻辑（1h/片 + 2s 重叠 + 交叉淡变拼接） |
| `_run_demucs_single(...)` | 内部：单片 Demucs（含 tqdm 进度 monkey-patch） |
| `_stitch_chunks_from_files(paths, sr, cf)` | 流式磁盘读取 + 交叉淡变拼接（内存最多 2 块） |

### `core/voice_similarity.py`

| 函数 | 说明 |
|---|---|
| `compare_similarity(tts_wav, orig_wav)` | Resemblyzer 余弦相似度 → `{similarity, device, error}` |

### `core/qwen_aligner.py`

| 函数 | 说明 |
|---|---|
| `_load_model()` / `load_model()` | 启动 Qwen API 子进程服务 |
| `unload_model()` | 关闭子进程，释放显存 |
| `align_batch(items, language)` | 批量对齐 REST API 调用 → `[[{word, start_ms, end_ms}, ...], ...]` |
| `set_api_url(url)` | 设置 API 地址 |
| `set_log_cb(cb)` | 设置日志回调 |

**自动子进程管理**：
- 首次 `align_batch` 调用时自动启动 `qwen_api/qwen_api_server.py`
- CUDA 设备用 `uv run` 启动，CPU 设备用系统 Python
- 主程序退出时 `atexit` 自动关闭
- 支持外部预启动的服务（先检查 `/health`）

---

## 声道策略

流水线根据原始视频音频的声道数自动适配。

### 立体声输入

```
原始视频 (stereo)
  ├── mix_orig.wav (stereo, -ac 2) ─── 基底音频
  ├── Demucs: vocals_orig.wav (mono, -ac 1) + background.wav (stereo)
  ├── mix_segment_clip: mono TTS 广播到 stereo + stereo 背景 → stereo
  └── splice_segments_into_base: stereo 基底 + stereo clip → stereo 输出
```

### 单声道输入

全流程保持单声道。

---

## UI 界面布局

主窗口使用 QGridLayout 三列布局（最小 1430×900）：

| 左列 (450px) | 中列 (stretch) | 右列 (500px) |
|---|---|---|
| **VAD 原声字幕表**（默认隐藏） | **输入文件区** + **字幕预览表** | **设置区** + **控制按钮** + **6步分段进度** + **混音预览** + **日志** |

### 字幕预览表（12 列）

| 列 | 图标/内容 | 说明 |
|---|---|---|
| 序号 | 数字 | 字幕行号 |
| 开始 | 时间 | 字幕开始时间 |
| 结束 | 时间 | 字幕结束时间 |
| 性别 | 男/女/空/未检测 | 点击循环切换，影响 TTS 参考音频选择 |
| 字幕文本 | 文本 | 字幕内容（tooltip 显示完整文本） |
| **试听** | ▶ | TTS 合成 + 背景混音（最终配音效果） |
| **原声** | 🎶 | 原始混合音频片段 |
| **tts** | ✨ | 纯 TTS（混音前） |
| **人声** | 🎵 | Demucs 分离后的原始人声 |
| **对比** | 🔍 | 打开波形对比弹窗 |
| 状态 | pending/TTS完成/... | 当前处理状态 |
| **重试** | 🔄 | 删除缓存并重新合成 |

### 设置区

- **本地引擎 (IndexTTS2)** — 复选框，勾选后使用本地模型替代 API
- **API 地址** — TTS 服务 URL（本地引擎启用时自动禁用）
- **提示音模式** —「原视频」从视频提取｜「固定提示」使用预设音频
- **⚙ 设置** — 详细参数配置（含情绪参数生成 kwargs）

### 分段进度（6 步）

```
[▶ 提取] [▶ 分离] [▶ 区间] [▶ TTS] [▶ 重建] [▶ 合并]
   进度条 (共用)        ✅ 完成 / ▶ 进行中 / ⏳ 待处理
```

- 每步有独立 ▶ 按钮，支持断点续跑
- 共用进度条显示当前步骤百分比
- 完成自动跳转到下一步

### 混音预览区

- **播放混音** / **停止** / **重新混音** 按钮
- 音量滑块
- 波形预览图（`WaveformPreviewWidget`，支持拖动 seek + 滚轮缩放）

### 播放器架构

- 全局共享一个 `QMediaPlayer` + `QAudioOutput`
- 播放不同文件时，先 `player.stop()` 再重建 `QAudioOutput`，安全跟随系统默认输出设备
- 支持字幕区间终点自动停止（`_play_end_ms` + 定时器兜底）

### UI Mixin 架构

主窗口 `MainWindow` 通过 Mixin 模式拆分功能模块：

| Mixin | 文件 | 职责 |
|---|---|---|
| `PlaybackMixin` | `ui/playback_mixin.py` | 播放控制、波形加载、缓存音频播放 |
| `PipelineMixin` | `ui/pipeline_mixin.py` | 流水线回调、进度更新、完成处理 |
| `ExecutionMixin` | `ui/execution_mixin.py` | 执行控制、路径校验、UI 状态管理 |
| `CacheMixin` | `ui/cache_mixin.py` | 缓存管理、清空操作 |

**公共方法**（提取自重复逻辑）：
- `_validate_paths()` — 校验视频/字幕路径
- `_set_pipeline_running(running)` — 统一设置运行时 UI 状态
- `_reset_pipeline_ui_state()` — 重置运行后 UI 状态
- `_play_cached_audio()` — 通用缓存音频播放模板

---

## 波形对比弹窗

点击字幕表的 **🔍 对比** 按钮打开，最多显示 **4 个对齐波形**：

| 音轨 | 图标 | 颜色 | 来源 |
|---|---|---|---|
| 试听 | ▶ 试听 | 绿色 | TTS 合成 + 背景混音 |
| 原声 | 🎶 原声 | 紫色 | 原始人声 + 背景（全长文件 seek） |
| tts | ✨ tts | 橙色 | 纯 TTS（混音前） |
| 人声 | 🎵 人声 | 蓝色 | Demucs 分离后原始人声（全长文件 seek） |

特性：
- 每个波形独立 row，含 **▶ 播放** / **⏹ 停止** / **📂 打开文件位置** 按钮
- 所有波形共用同一个 `QMediaPlayer`，播放时 **四行坐标同步移动**
- 使用 `numpy` 降采样到 200 个柱状条绘制
- 已播放部分彩色、未播放灰色
- 红色竖线指示当前播放位置（活跃波形红色，其他灰色）
- 点击波形 **seek 到指定位置并暂停**（不自动播放）
- 播放到区间末尾 **自动暂停并复位到起点**
- 时间轴刻度自适应精度（秒/毫秒）
- 支持切换不同音轨播放，坐标自动同步

---

## 设置说明

### 主界面
- **本地引擎 (IndexTTS2)** — 勾选后使用本地 IndexTTS2 模型替代 API 服务
- **API 地址** — 仅本地引擎未勾选时可用
- **提示音模式** —「原视频」从视频提取｜「固定提示」使用预设音频
- **⚙ 设置** — 详细参数配置

### 设置弹窗（2 个 Tab）

**Tab 1: TTS API 设置**

| 参数 | 说明 |
|---|---|
| 预设方案 | 手动 / 雨落版 / CosyVoice / GPT-SoVITS / OpenAI，自动填充模式和模型 |
| API 密钥 | 认证密钥（密码模式） |
| API 模式 | rainfall / cosyvoice / gpt-sovits / openai / custom |
| 模型名 | 传递到 API 的 model 参数 |
| 语言 | 提示 TTS 服务目标语言 |
| 超时 | API 请求超时（秒） |
| 额外参数 | JSON 格式，合并到 API 请求体 |

**固定提示音设置**

| 参数 | 说明 |
|---|---|
| 男声/女声参考音频 | WAV/MP3 文件选择 + ▶ 试听 |
| 同时发送提示文本 | 部分 API（如 CosyVoice）需要参考文本 |
| 参考性别 | 默认参考音频性别（男/女/无） |

**Tab 2: 音频设置**

| 参数 | 默认值 | 说明 |
|---|---|---|
| Demucs 模型 | htdemucs | htdemucs / htdemucs_ft / htdemucs_6s |
| 运算设备 | auto | auto / cpu / cuda / mps |
| CPU 线程 | 4 | Demucs 并行线程数（1-64） |
| 处理片段 | 7 | Demucs 内部窗口大小（秒，htdemucs 最大 7） |
| 片段重叠 | 0.25 | Demucs 片段重叠比例 |
| VAD 模式 | 原声对齐 | 原声对齐 / 字幕对齐 |
| VAD 留白 | 50 | 字幕对齐模式下固定留白（毫秒） |
| 边界扩展 | 100 | 混音片段边界扩展（毫秒） |
| TTS 线程 | 1 | TTS 合成并行线程数（上限 4） |
| 混音线程 | 6 | 全长重建并行线程数 |
| Step3 提取线程 | 4 | 性别检测 / Qwen 对齐时音频片段并行提取线程数 |
| 性别批大小 | 6 | WavLM 性别检测每批条数 |
| Qwen 批大小 | 8 | Qwen 对齐每批条数 |
| Qwen 最大 padding | 500 | 送检音频最大单侧 padding（毫秒） |
| Qwen 安全间距 | 200 | 两字幕间安全间距（毫秒） |
| Qwen 安全区 | 100 | Qwen 结果前后安全区（毫秒） |
| 声纹阈值 | 0.6 | 低于此值时用固定提示音重试 |

---

## 声纹相似度检测

基于 **Resemblyzer** 的说话人嵌入对比，自动检测 TTS 合成质量。

### 流程（3 轮）

```
Phase 1: 批量 TTS 合成 ────── 多线程并行合成所有字幕
Phase 2: 批量声纹校验 ────── 逐一对比 TTS vs 原始人声
Phase 3: 固定提示音对比 ─── 低于阈值 → 与固定提示音比对 → ≥0.6 跳过 → 不足则重试
```

- 相似度结果按 TTS 文件的 mtime+size 缓存（`.similarity_cache.json`）
- TTS 文件未变时跳过重检
- 单条重生成（🔄 按钮）也带内联声纹校验
- 清除缓存对话框新增「4.1 声纹校验缓存」选项

### 日志示例

```
🎯 第6条 相似度满足: 0.747 (cpu)
⚠️ 第1条 相似度不足: 0.497 (cpu)
🔄 共 7 条相似度不足，开始重试...
🎯 第4条 重试相似度满足: 0.923
```

---

## 进度显示

各步骤进度条显示当前完成计数 `(x/y)`：

| 步骤 | 格式 |
|---|---|
| 提取音频 | `(0/1)` |
| 分离人声 | `(0/2)` Demucs → `(1/2)` 人声转单声道 → `(2/2)` 背景复制 |
| 字幕处理 | `(0/total)` → `VAD校准 (done/total)` → `强制对齐 (done/total)` → `性别检测 (done/total)` |
| TTS 合成 | `(0/total)` → `(done/total)` |
| 音频处理 | `(0/total)` → `(done/total)` |
| 拼接合并 | `(0/1)` |

大模型步骤标题自动标注实际设备：
- `=== 2.分离人声 (cuda) ===`
- `=== 3.字幕处理 (cuda) ===`
- `=== 4.TTS合成 (cuda) ===`

---

## 项目结构

```
movie-dub-gui/
├── main.py                      # 入口 — 启动 PySide6 QApplication + GPU 清理
├── requirements.txt             # 依赖: PySide6, requests
│
├── core/                        # 核心引擎（无 UI 依赖）
│   ├── config.py                # 全局配置管理 — Cfg 类 + default_cfg 实例
│   ├── pipeline.py              # 6 步流水线 + PipelineOrchestrator
│   │   ├── PipelineContext      # 共享上下文（路径、状态、配置、回调）
│   │   ├── BaseStep             # 步骤基类（缓存检查、依赖、执行）
│   │   ├── ExtractAudioStep     # Step 1: ffmpeg 提取音频
│   │   ├── SeparateVocalsStep   # Step 2: Demucs 人声分离
│   │   ├── SubStep              # Step 3: 字幕解析 + Qwen对齐 + WavLM性别检测
│   │   ├── TTSSynthesisStep     # Step 4: TTS 合成（含 3 轮声纹校验）
│   │   ├── AudioMixAndMergeStep  # Step 5: 混音 + 全长拼接
│   │   └── VideoMergeStep        # Step 6: 合并回视频
│   │
│   ├── audio_tools.py           # 音频处理函数集（裁剪、VAD、增益、混音、拼接）
│   ├── utils.py                 # 工具函数（时间格式化、WAV读写、降采样、设备检测、日志/线程）
│   ├── cache_manager.py         # 缓存管理器（基于文件Hash的逐step跳过）
│   ├── srt_parser.py            # SRT 字幕解析 → List[SubtitleItem]
│   ├── tts_client.py            # TTS API 客户端（5 种模式）
│   ├── voice_separator.py       # Demucs 人声分离（含超长音频分片拼接）
│   ├── voice_similarity.py      # Resemblyzer 声纹相似度检测
│   ├── qwen_aligner.py          # Qwen3-ForcedAligner（REST API 客户端 + 子进程管理）
│   ├── tts_engine.py            # IndexTTS2 本地推理引擎（单例 + 情绪控制 + 时长对齐）
│   ├── model_base.py            # MLModelHolder 基类（统一模型生命周期）
│   ├── wavlm_model/             # WavLM-Large + LoRA 架构定义
│   │   ├── wavlm_demographics.py
│   │   ├── revgrad.py / revgrad_func.py
│   │   └── __init__.py
│   └── wavlm_gender_api.py      # WavLM 性别检测 API（批处理）
│
├── ui/
│   ├── main_window.py           # 主窗口（Mixin 组合）
│   ├── execution_mixin.py       # 执行控制（路径校验、UI 状态、启动/取消）
│   ├── pipeline_mixin.py        # 流水线回调（进度更新、完成处理、UI 恢复）
│   ├── playback_mixin.py        # 播放控制（QMediaPlayer、波形加载、缓存播放）
│   ├── cache_mixin.py           # 缓存管理（清空操作）
│   ├── pipeline_worker.py       # PipelineWorker/TaskThread（QThread 封装）
│   ├── settings_dialog.py       # 设置对话框（2 Tab）
│   ├── waveform_dialog.py       # 4 波形同步对比弹窗
│   └── table_models.py          # 字幕表数据模型
│
├── qwen_api/                    # Qwen 对齐 API 服务（子进程）
│   └── qwen_api_server.py       # FastAPI 服务端
│
├── indextts2_src/               # IndexTTS2 源码（克隆自 upstream）
│   └── checkpoints/             # IndexTTS2 模型权重（需单独下载）
│
├── models/                      # 本地模型权重（不提交 git）
│   ├── wavlm-large/             # WavLM-Large 骨架（config + safetensors）
│   ├── wavlm-large-age-sex/     # 微调权重（config + safetensors + preprocessor_config）
│   ├── qwen3-forced-aligner/    # Qwen3ForcedAligner 模型
│
├── tts_api_server.py            # Windows SAPI TTS 服务器
├── .cache/                      # 缓存目录（自动创建）
└── .gitignore
```

---

## 已知问题 / 兼容性

### torchaudio.save 不兼容
torch 2.11 的 torchcodec 在 Windows 上找不到 FFmpeg DLL。引擎内部已 monkey-patch 为使用 `scipy.io.wavfile` 保存。


基于 PySide6 的桌面 GUI 工具，用于电影 AI 翻译配音。将视频中的人声替换为 AI 合成的目标语言语音，
