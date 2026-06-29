"""editdistance 纯 Python 回退 — 当 C 扩展不可用时使用"""


def eval(s1: str, s2: str) -> int:
    """Levenshtein distance"""
    if len(s1) < len(s2):
        return eval(s2, s1)
    if not s2:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


# ── 注入到 editdistance 模块 ──
import sys
if 'editdistance' not in sys.modules:
    import types
    mod = types.ModuleType('editdistance')
    mod.eval = eval
    mod.distance = eval
    sys.modules['editdistance'] = mod
