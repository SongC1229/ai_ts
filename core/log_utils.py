"""日志工具 — 统一使用 Python logging 模块

提供:
- QtLogHandler: 将日志输出到 QTextEdit（跨线程安全)
- setup_logging(): 配置根 logger,添加 UI + 文件两个 handler
"""

import logging
import os
from datetime import datetime

from PySide6.QtCore import QObject, Signal


class _Signaller(QObject):
    """内部信号对象,确保跨线程安全"""
    append = Signal(str)


class QtLogHandler(logging.Handler):
    """将 logging 记录追加到 QTextEdit 控件（跨线程安全)"""

    def __init__(self, text_widget=None):
        super().__init__()
        self._signaller = _Signaller()
        self._text_widget = text_widget
        if text_widget:
            self._signaller.append.connect(self._on_append)

    def _on_append(self, msg: str):
        widget = self._text_widget
        if widget is None:
            return
        sb = widget.verticalScrollBar()
        at_bottom = sb.maximum() < 1 or sb.value() >= sb.maximum() - 4
        widget.append(msg)
        if at_bottom:
            sb.setValue(sb.maximum())

    def set_text_widget(self, widget):
        """延迟绑定 QTextEdit（控件创建后才可调用)"""
        self._text_widget = widget
        self._signaller.append.connect(self._on_append)

    def emit(self, record):
        try:
            msg = self.format(record)
            self._signaller.append.emit(msg)
        except Exception:
            self.handleError(record)


class _AppFilter(logging.Filter):
    """仅允许应用自身日志通过,过滤第三方库（torch/transformers/indextts 等)的 INFO/WARNING 噪声

    同时应用于 UI handler 和文件 handler,ERROR 及以上仍会记录。
    """

    def filter(self, record):
        name = record.name
        # 应用日志：ui.* / core.* / __main__ / root
        if name.startswith(('ui.', 'core.')) or name in ('__main__', 'root'):
            return True
        # 第三方库日志只留存 ERROR 及以上
        return record.levelno >= logging.ERROR


class _LogFormatter(logging.Formatter):
    """自定义格式器

    - UI handler:  HH:MM:SS.m# message
    - File handler: YYYY-MM-DD HH:MM:SS.m# message
    """

    _FILE_DATEFMT = "FILE"

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        if datefmt == self._FILE_DATEFMT:
            return dt.strftime("%y-%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
        return dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 100000}"

    def format(self, record):
        return f"{self.formatTime(record, self.datefmt)}# {record.getMessage()}"


def setup_logging(text_widget, log_dir: str):
    """配置根 logger：UI handler + 文件 handler

    Args:
        text_widget: QTextEdit 实例,用于显示日志
        log_dir: 日志文件存储目录
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 清除已有 handlers（避免重复添加)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()

    # 1) UI handler（INFO 及以上显示到界面,过滤第三方库噪声)
    ui_handler = QtLogHandler(text_widget)
    ui_handler.addFilter(_AppFilter())
    ui_handler.setFormatter(_LogFormatter())
    ui_handler.setLevel(logging.INFO)
    logger.addHandler(ui_handler)

    # 2) 文件 handler（DEBUG 及以上写入文件,仅保留应用日志)
    os.makedirs(log_dir, exist_ok=True)
    log_name = "gui_" + datetime.now().strftime("%Y%m%d") + ".log"
    file_handler = logging.FileHandler(
        os.path.join(log_dir, log_name),
        encoding='utf-8',
    )
    file_handler.addFilter(_AppFilter())
    file_handler.setFormatter(_LogFormatter(datefmt=_LogFormatter._FILE_DATEFMT))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
