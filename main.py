#!/usr/bin/env python3
"""电影 AI 配音工具 — PySide6 GUI"""
import sys
import os
import atexit

# 确保在项目根目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 强制离线模式，避免 HuggingFace 网络请求卡住
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# 屏蔽 Qt FFmpeg 调试信息
os.environ["QT_LOGGING_RULES"] = "qt.multimedia.ffmpeg=false"

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.main_window import MainWindow  # noqa: E402
from core.utils import cleanup_cuda  # noqa: E402


def _cleanup_gpu():
    """强制释放 GPU 显存"""
    try:
        from core.tts_engine import unload_tts_engine
        unload_tts_engine()
    except Exception:
        pass
    try:
        cleanup_cuda()
    except Exception:
        pass


def main():
    # 退出时清理 GPU 显存(开机无需清理)
    atexit.register(_cleanup_gpu)

    app = QApplication(sys.argv)

    # 设置全局样式
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
