"""缓存管理器 — 基于文件 Hash 的逐step跳过"""

import glob
import hashlib
import json
import os
import shutil


# ── 文件名生成（模块内部使用）──

def _fmt_ts(ms: int) -> str:
    """毫秒转 HH_MM_SS_MS（用于文件名）"""
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}_{m:02d}_{s:02d}_{ms:03d}"


def _fast_file_hash(filepath: str) -> str:
    """快速文件指纹：sha256( 文件大小 + 修改时间 + 头64KB + 尾64KB )"""
    try:
        size = os.path.getsize(filepath)
        mtime = os.path.getmtime(filepath)
    except OSError:
        return hashlib.sha256(filepath.encode()).hexdigest()[:16]
    h = hashlib.sha256()
    h.update(f"{size}:{mtime}".encode())
    try:
        with open(filepath, 'rb') as f:
            head = f.read(65536)
            h.update(head)
            if size > 65536:
                f.seek(-65536, 2)
                tail = f.read(65536)
                h.update(tail)
    except OSError:
        pass
    return h.hexdigest()[:4]


class Step:
    """缓存步骤目录名常量 — get_path 的第一个参数用这些代替裸字符串"""
    EXTRACT = "extract"
    DEMUCS = "demucs"
    SUBS = "subs"
    TTS = "tts"
    MIX = "mix"


class CacheManager:
    """\u6309step缓存中间文件,同名同文件时可跳过已完成step
    
    缓存结构:
      {cache_root}/{video_hash}/
        ├── .cache_info
        ├── extract/
        │   └── mix_orig.wav            ← 原始混合音频
        ├── demucs/
        │   ├── vocals_orig.wav        ← 原始人声轨
        │   └── background.wav         ← 背景轨 (音乐/环境音/音效)
        ├── subs/
        │   ├── genders_cache.json     ← 性别缓存
        │   └── calib.json             ← 校准后的字幕时间
        ├── tts/
        │   ├── .similarity_cache.json ← 声纹相似度缓存（按 TTS 文件 mtime/size 失效)
        │   ├── _ref_*.wav             ← 参考音频片段（人声轨裁剪）
        │   ├── tts_*.wav              ← TTS 原始合成
        │   └── mixed_*.wav            ← TTS+背景 片段混音
        ├── mix/
        │   └── final_audio.wav        ← 全长拼接后最终音频
        └── video/   (不缓存,始终重新生成)
    """

    STEP_NAMES = [
        Step.EXTRACT,
        Step.DEMUCS,
        Step.SUBS,
        Step.TTS,
        Step.MIX,
    ]

    STEP_CHECK_FILES = {
        Step.EXTRACT: ["mix_orig.wav"],
        Step.DEMUCS:  ["vocals_orig.wav", "background.wav"],
        Step.SUBS:    ["genders_cache.json"],
        Step.TTS:     ["tts_*.wav"],
        Step.MIX:     ["final_audio.wav"],
    }

    def __init__(self, video_path: str, cache_root: str):
        self.video_path = video_path
        self.cache_root = cache_root
        self._hash = None

        os.makedirs(self.cache_root, exist_ok=True)

        self.cache_dir = os.path.join(self.cache_root, self.video_hash)
        self.info_path = os.path.join(self.cache_dir, '.cache_info')
        self._info = self._load_info()

    @property
    def video_hash(self) -> str:
        if self._hash is None:
            from pathlib import Path
            stem = Path(self.video_path).stem
            h = _fast_file_hash(self.video_path)
            self._hash = f"{stem}.{h[:4]}"
        return self._hash

    # ── 信息读写 ──

    def _load_info(self) -> dict:
        if os.path.exists(self.info_path):
            try:
                with open(self.info_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_info(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.info_path, 'w', encoding='utf-8') as f:
            json.dump(self._info, f, indent=2, ensure_ascii=False)

    def is_step_completed(self, step_name: str) -> bool:
        if step_name not in self.STEP_NAMES:
            return False
        return self._info.get(step_name, False)

    def verify_step_files(self, step_name: str) -> bool:
        if not self._info.get(step_name, False):
            return False
        for fn in self.STEP_CHECK_FILES.get(step_name, []):
            if not os.path.exists(os.path.join(self.cache_dir, step_name, fn)):
                return False
        return True

    def mark_completed(self, step_name: str):
        self._info[step_name] = True
        self._save_info()

    def mark_incomplete(self, step_name: str):
        self._info[step_name] = False
        self._save_info()

    def get_meta(self, key: str, default=None):
        """从缓存元数据读取值（不依赖 step 完成状态)"""
        return self._info.get(f"meta_{key}", default)

    def set_meta(self, key: str, value):
        """写入缓存元数据并持久化"""
        self._info[f"meta_{key}"] = value
        self._save_info()

    def get_first_incomplete_step(self) -> int:
        for i, name in enumerate(self.STEP_NAMES):
            if not self.is_step_completed(name):
                return i
        return len(self.STEP_NAMES)

    # ── 路径获取 ──

    def get_path(self, step_name: str, filename: str) -> str:
        step_dir = os.path.join(self.cache_dir, step_name)
        os.makedirs(step_dir, exist_ok=True)
        return os.path.join(step_dir, filename)

    def get_step_dir(self, step_name: str) -> str:
        path = os.path.join(self.cache_dir, step_name)
        os.makedirs(path, exist_ok=True)
        return path


    @property
    def vocals_path(self) -> str:
        """原始人声文件路径"""
        return self.get_path(Step.DEMUCS, "vocals_orig.wav")

    @property
    def bg_path(self) -> str:
        """背景音轨文件路径"""
        return self.get_path(Step.DEMUCS, "background.wav")

    @property
    def mix_orig_path(self) -> str:
        """原始混合音频路径"""
        return self.get_path(Step.EXTRACT, "mix_orig.wav")

    # ── 路径格式化（日志用,不含 .cache/ 前缀)──

    def rel_path(self, step_name: str, filename: str) -> str:
        """返回相对路径字符串,如 subs/calib.json（日志用)"""
        return f"{step_name}{os.sep}{filename}"

    def step_dir_rel(self, step_name: str) -> str:
        """返回步骤目录相对路径,如 subs/（日志用)"""
        return f"{step_name}{os.sep}"

    # ── 字幕时间感知的文件路径 ──

    def tts_path(self, sub) -> str:
        """TTS 文件路径"""
        return self.get_path(Step.TTS, f"tts_{sub.idx:04d}_{_fmt_ts(sub.eff_start_ms)}-{_fmt_ts(sub.eff_end_ms)}.wav")

    def ref_path(self, sub) -> str:
        return self.get_path(Step.TTS, f"_ref_{sub.idx:04d}_{_fmt_ts(sub.eff_start_ms)}_{_fmt_ts(sub.eff_end_ms)}.wav")

    def mixed_path(self, sub) -> str:
        return self.get_path(Step.TTS, f"mixed_{sub.idx:04d}_{_fmt_ts(sub.eff_start_ms)}-{_fmt_ts(sub.eff_end_ms)}.wav")

    @property
    def final_mix_path(self) -> str:
        """全长混音文件路径"""
        return self.get_path(Step.MIX, "final_audio.wav")

    def file_info(self, step_name: str, filename: str) -> tuple:
        """统一文件检查：返回 (exists, full_path, rel_path)"""
        full = self.get_path(step_name, filename)
        rel = self.rel_path(step_name, filename)
        return os.path.exists(full), full, rel

    # ── 缓存管理 ──

    def clear(self):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        self._info = {}

    def clear_step(self, step_name: str):
        if step_name not in self.STEP_NAMES:
            return
        step_dir = os.path.join(self.cache_dir, step_name)
        if os.path.exists(step_dir):
            shutil.rmtree(step_dir, ignore_errors=True)
        self._info[step_name] = False
        self._save_info()

    # ── 高层查询方法 ──

    def load_gender_cache(self, subs: list = None) -> dict:
        """加载性别缓存, 若提供 subs 则直接写入 sub.gender, 始终返回 {str(idx): gender_str}"""
        path = self.get_path(Step.SUBS, "genders_cache.json")
        genders = {}
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    genders = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        if subs:
            for sub in subs:
                sub.gender = genders.get(str(sub.idx), '')
        return genders

    def load_calib_cache(self, subs: list = None, raw_subs: list = None) -> bool:
        """从 calib.json 加载校准时间（subs 和 raw_subs 共用), 直接写入 calib_start_ms/calib_end_ms"""
        path = self.get_path(Step.SUBS, "calib.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding='utf-8') as f:
                calib_list = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        calib_map = {c['idx']: c for c in calib_list if c.get('idx')}

        for sub in (subs or []):
            c = calib_map.get(sub.idx)
            if c:
                sub.calib_start_ms = c.get('calib_start_ms', 0)
                sub.calib_end_ms = c.get('calib_end_ms', 0)
                sub.calib_vad_ms = c.get('calib_vad_ms', -1)

        for sub in (raw_subs or []):
            c = calib_map.get(sub.idx)
            if c:
                sub.calib_start_ms = c.get('calib_start_ms', 0)
                sub.calib_end_ms = c.get('calib_end_ms', 0)
                sub.calib_vad_ms = c.get('calib_vad_ms', -1)

        return True

    def save_gender_cache(self, genders: dict):
        """保存性别缓存（空 dict 不写入)"""
        if not genders:
            return
        path = self.get_path(Step.SUBS, "genders_cache.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(genders, f, indent=2, ensure_ascii=False)

    def update_gender(self, idx: int, gender: str):
        """更新单条性别（读→改→写)

        Args:
            idx: 1-based 字幕索引 (sub.idx)
            gender: 性别字符串 ("male"/"female"/"")
        """
        cache = self.load_gender_cache()
        cache[str(idx)] = gender
        self.save_gender_cache(cache)


    def restore_calib_subs(self, dst_srt_path: str,
                           parse_srt_fn, raw_src_path: str = "") -> tuple:
        """从校准缓存恢复 subs 和 raw_subs 列表

        Returns:
            (subs, raw_subs, has_calib)
        """
        # 1. 解析 SRT 得到原始字幕
        subs = parse_srt_fn(dst_srt_path) if dst_srt_path else []
        raw_subs = parse_srt_fn(raw_src_path) if raw_src_path else []

        # 2. 从 calib.json + genders_cache.json 写入字段
        self.load_gender_cache(subs)
        has_calib = self.load_calib_cache(subs, raw_subs)

        return (subs, raw_subs, has_calib)

    def find_raw_tts(self, idx: int) -> str:
        """查找指定 idx 的原始 TTS 文件路径,不存在返回空字符串"""
        tts_dir = self.get_step_dir("tts")
        matches = glob.glob(os.path.join(tts_dir, f"tts_{idx:04d}_*.wav"))
        return matches[0] if matches else ""

    def scan_tts_cache(self) -> tuple:
        """预扫描 TTS 目录,返回 (raw_cache, mixed_cache) 两个 {idx: path} 字典"""
        raw, mixed = {}, {}
        tts_dir = self.get_step_dir("tts")
        for p in glob.glob(os.path.join(tts_dir, "tts_*.wav")):
            try:
                idx = int(os.path.basename(p).split('_')[1])
                raw[idx] = p
            except (IndexError, ValueError):
                pass
        for p in glob.glob(os.path.join(tts_dir, "mixed_*.wav")):
            try:
                idx = int(os.path.basename(p).split('_')[1])
                mixed[idx] = p
            except (IndexError, ValueError):
                pass
        return raw, mixed

    def has_step_file(self, step_name: str) -> bool:
        """基于文件系统检查 step 是否有缓存文件"""
        step_dir = os.path.join(self.cache_dir, step_name)
        if not os.path.exists(step_dir):
            return False
        for pattern in self.STEP_CHECK_FILES.get(step_name, []):
            if '*' in pattern:
                if glob.glob(os.path.join(step_dir, pattern)):
                    return True
            else:
                if os.path.exists(os.path.join(step_dir, pattern)):
                    return True
        # 无预定义检查项时,检查目录是否非空
        if step_name not in self.STEP_CHECK_FILES:
            return any(os.path.isfile(os.path.join(step_dir, f))
                       for f in os.listdir(step_dir))
        return False

    def get_all_step_status(self) -> dict:
        """返回所有 step 的状态（文件检查优先,降级到 .cache_info 标记)"""
        result = {}
        for step in self.STEP_NAMES:
            if step in self.STEP_CHECK_FILES:
                result[step] = self.has_step_file(step)
            else:
                result[step] = self._info.get(step, False)
        return result

    def calculate_total_cache_size(self) -> tuple:
        """计算 cache_root 下所有缓存的总大小,返回 (total_bytes, file_count)"""
        total = 0
        count = 0
        if not os.path.exists(self.cache_root):
            return 0, 0
        for entry in os.listdir(self.cache_root):
            path = os.path.join(self.cache_root, entry)
            if os.path.isdir(path):
                for dirpath, _, fns in os.walk(path):
                    for fn in fns:
                        fp = os.path.join(dirpath, fn)
                        try:
                            total += os.path.getsize(fp)
                            count += 1
                        except OSError:
                            pass
        return total, count

