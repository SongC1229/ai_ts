"""SRT 字幕解析模块"""

import re
from dataclasses import dataclass
from typing import List


# 预编译正则（parse_srt 每条字幕都会用到,避免重复编译)
_TIME_FULL_RE = re.compile(
    r'(\d{1,2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{3})')
_TIME_SHORT_RE = re.compile(
    r'(\d{1,2}:\d{2}[.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}[.,]\d{3})')
_HTML_TAG_RE = re.compile(r'<[^>]+>')
_FORMAT_TAG_RE = re.compile(r'\{[^}]*\}')


@dataclass
class SubtitleItem:
    """单条字幕

    字段生命周期:
    - idx, text: 初次加载后不更改
    - start_ms, end_ms: 始终存储原始 SRT 时间戳
    - calib_start_ms, calib_end_ms: 校准时间（0 = 未校准）
    - gender: 性别检测结果（空串 = 未检测）
    """
    idx: int              # 1-based 序号
    start_ms: int         # 原始 SRT 开始时间
    end_ms: int           # 原始 SRT 结束时间
    text: str             # 字幕文本
    calib_start_ms: int = 0   # 校准开始时间（0 = 未校准）
    calib_end_ms: int = 0     # 校准结束时间（0 = 未校准）
    gender: str = ""          # 性别 "male"/"female"/""

    @property
    def is_calibrated(self) -> bool:
        return self.calib_start_ms != 0 or self.calib_end_ms != 0

    @property
    def eff_start_ms(self) -> int:
        return self.calib_start_ms or self.start_ms

    @property
    def eff_end_ms(self) -> int:
        return self.calib_end_ms or self.end_ms

    def to_cache_dict(self) -> dict:
        """导出到校准缓存 JSON（只含 idx + 校准时间）"""
        d = {'idx': self.idx}
        if self.is_calibrated:
            d['calib_start_ms'] = self.calib_start_ms
            d['calib_end_ms'] = self.calib_end_ms
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'SubtitleItem':
        """从 JSON 导出的 dict 创建 SubtitleItem（初始用 parse_srt)"""
        return cls(
            idx=d.get('idx', 0),
            start_ms=d.get('start_ms', 0),
            end_ms=d.get('end_ms', 0),
            text=d.get('text', ''),
            gender=d.get('gender', ''),
            calib_start_ms=d.get('calib_start_ms', 0),
            calib_end_ms=d.get('calib_end_ms', 0),
        )


def _timestamp_to_ms(ts: str) -> int:
    """将 SRT 时间戳转为毫秒  01:23.456 或 01:02:03.456"""
    parts = ts.replace(',', '.').split(':')
    if len(parts) == 3:
        h, m, s = parts
    else:
        h, m, s = '0', parts[0], parts[1]
    return int(h) * 3600000 + int(m) * 60000 + int(float(s) * 1000)


def parse_srt(filepath: str) -> List[SubtitleItem]:
    """解析 SRT 文件,返回字幕列表"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    # 按空行分割成块
    blocks = re.split(r'\n\s*\n', content.strip())
    items = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        # 第一行：序号
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # 第二行：时间轴 00:01:23.456 --> 00:01:25.678
        time_match = _TIME_FULL_RE.match(lines[1])
        if not time_match:
            # 尝试不带小时的格式 00:01.456
            time_match = _TIME_SHORT_RE.match(lines[1])
        if not time_match:
            continue

        start_ms = _timestamp_to_ms(time_match.group(1))
        end_ms = _timestamp_to_ms(time_match.group(2))

        # 剩余行：字幕文本
        text = '\n'.join(lines[2:]).strip()
        # 去掉 HTML 标签
        text = _HTML_TAG_RE.sub('', text)
        # 去掉 {...} 格式标签
        text = _FORMAT_TAG_RE.sub('', text)

        if text:
            items.append(SubtitleItem(
                idx=index,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text
            ))

    return items


def fmt_srt_time(ms: int) -> str:
    """毫秒 → SRT 时间戳  HH:MM:SS,mmm"""
    if ms < 0:
        ms = 0
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    msec = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def write_srt(subs: List, out_path: str) -> int:
    """把字幕列表写成 SRT 文件 (id 从 1 开始连续编号, 文本可空)

    Args:
        subs: SubtitleItem 列表, 或 (start_ms, end_ms, text) 元组列表
        out_path: 输出 .srt 路径

    Returns:
        写入的条数 (0 表示无内容或失败)
    """
    if not subs:
        return 0
    lines = []
    for i, sub in enumerate(subs):
        # 兼容 SubtitleItem 和 (start_ms, end_ms, text) 元组
        if isinstance(sub, SubtitleItem):
            s_ms, e_ms, text = sub.start_ms, sub.end_ms, sub.text
        else:
            s_ms, e_ms, text = sub
        lines.append(str(i + 1))
        lines.append(f"{fmt_srt_time(s_ms)} --> {fmt_srt_time(e_ms)}")
        lines.append(text)
        lines.append("")  # 空行分隔
    try:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        return 0
    return len(subs)


