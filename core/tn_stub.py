"""tn 模块桩 — 替代 WeTextProcessing（Python 3.13+ 不兼容)"""
import sys
import types

class Normalizer:
    """空 normalizer, 直接返回原文"""
    def normalize(self, text: str) -> str:
        return text

def _install():
    for name in ['tn', 'tn.chinese', 'tn.english',
                  'tn.chinese.normalizer', 'tn.english.normalizer']:
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.Normalizer = Normalizer
            pkg.__path__ = []
            pkg.normalizer = pkg
            sys.modules[name] = pkg

_install()
