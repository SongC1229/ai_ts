"""Qwen3-ForcedAligner REST API 客户端

自动管理 qwen_api_server.py 子进程的生命周期：
- 首次 align_batch 调用时自动启动服务
- 主程序退出时自动关闭服务
"""

import os
import sys
import time
import atexit
import signal
import subprocess
from typing import List
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    requests = None


_API_BASE_URL = "http://127.0.0.1:8765"
_SERVER_PROC = None
_SERVER_STARTED = False
_log_func = print  # 默认 print,可由 set_log_cb 覆盖为文件日志
_use_uv_first = False  # CUDA 时用 uv run，CPU 时用 sys.python


def set_prefer_uv():
    """根据设备设置 Qwen 启动偏好：CUDA→uv run，CPU→sys.python"""
    global _use_uv_first
    from .utils import resolve_device
    _dev = resolve_device("auto")
    _use_uv_first = (_dev == "cuda")
    _log(f"Qwen 启动方式: {'uv run' if _use_uv_first else 'sys.python'} (设备={_dev})")


def set_log_cb(cb):
    """设置日志回调,用于写入 UI + 日志文件"""
    global _log_func
    _log_func = cb


def _log(msg: str):
    _log_func(f"  {msg}")


def _get_qwen_api_dir() -> str:
    """返回 qwen_api/ 目录的绝对路径"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qwen_api")


def set_api_url(url: str):
    """设置 API 服务地址（默认 http://127.0.0.1:8765)"""
    global _API_BASE_URL
    _API_BASE_URL = url.rstrip("/")


def _check_requests():
    if requests is None:
        raise ImportError("需要 requests 库: pip install requests")


def _ensure_server():
    """确保 API 服务已启动（首次调用时自动启动)"""
    global _SERVER_PROC, _SERVER_STARTED

    if _SERVER_STARTED:
        return True

    # 先检查是否已有外部启动的服务
    try:
        r = requests.get(urljoin(_API_BASE_URL, "/health"), timeout=2)
        if r.status_code == 200:
            _SERVER_STARTED = True
            return True
    except Exception:
        pass

    # 自动启动服务
    api_dir = _get_qwen_api_dir()
    server_script = os.path.join(api_dir, "qwen_api_server.py")
    if not os.path.exists(server_script):
        _log(f"API 服务脚本不存在: {server_script}")
        return False

    port = _API_BASE_URL.split(":")[-1] if ":" in _API_BASE_URL else "8765"
    _log(f"启动 API 服务 (port={port}) ...")

    # 根据 TTS 模式选择启动顺序
    # CUDA 用 uv run 启动，CPU 用 sys.executable
    _launch_cmds = [
        ["uv", "run", "python", "qwen_api_server.py", "--port", port],
        [sys.executable, "qwen_api/qwen_api_server.py", "--port", port],
    ] if _use_uv_first else [
        [sys.executable, "qwen_api/qwen_api_server.py", "--port", port],
        ["uv", "run", "python", "qwen_api_server.py", "--port", port],
    ]
    _SERVER_PROC = None
    for _cmd in _launch_cmds:
        try:
            _SERVER_PROC = subprocess.Popen(
                _cmd,
                cwd=api_dir if _cmd[0] == "uv" else None,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env={**os.environ, "PYTHONWARNINGS": "ignore", "UVICORN_LOG_LEVEL": "warning"},
            )
            if _cmd[0] == "uv":
                _log("使用 uv run 启动")
            break
        except FileNotFoundError:
            continue
    if _SERVER_PROC is None:
        _log("启动失败：python 和 uv 均不可用")
        return False

    # 启动线程读取服务端日志
    import threading as _th
    def _read_stderr():
        import re as _re
        _skip_patterns = [
            'DeprecationWarning', 'on_event is deprecated',
            'lifespan event', 'Read more about it',
            'FastAPI docs', '@app.on_event',
            'INFO:', 'INFO ',
        ]
        _ansi_escape = _re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        try:
            for _line in iter(_SERVER_PROC.stderr.readline, b''):
                _msg = _line.decode('utf-8', errors='replace').strip()
                _clean = _ansi_escape.sub('', _msg)
                if not _clean:
                    continue
                if any(p in _clean for p in _skip_patterns):
                    continue
                if '[qwen-api]' in _clean:
                    _log(_clean.replace('[qwen-api]', '').strip())
        except Exception:
            pass
    _th.Thread(target=_read_stderr, daemon=True).start()

    # 注册退出时清理（仅一次)
    if not getattr(_ensure_server, '_registered', False):
        atexit.register(_stop_server)
        _ensure_server._registered = True

    # 等待服务就绪（最多 20 秒,模型首次请求时懒加载)
    for i in range(20):
        time.sleep(1)
        try:
            r = requests.get(urljoin(_API_BASE_URL, "/health"), timeout=2)
            if r.status_code == 200:
                _log(f"API 服务就绪 ({i+1}s)")
                _SERVER_STARTED = True
                return True
        except Exception:
            pass
        if i % 5 == 4:
            _log(f"等待服务启动 ... ({i+1}s)")

    _log(f"API 服务启动超时,请检查 {api_dir}/qwen_api_server.py")
    return False


def _stop_server():
    """关闭 API 服务子进程"""
    global _SERVER_PROC, _SERVER_STARTED
    if _SERVER_PROC is not None:
        try:
            if sys.platform == "win32":
                _SERVER_PROC.terminate()
                _SERVER_PROC.wait(timeout=5)
            else:
                _SERVER_PROC.send_signal(signal.SIGTERM)
                _SERVER_PROC.wait(timeout=5)
        except Exception:
            try:
                _SERVER_PROC.kill()
                _SERVER_PROC.wait(timeout=3)
            except Exception:
                pass
        _log("API 服务已关闭")
        _SERVER_PROC = None
    _SERVER_STARTED = False


def is_server_ready() -> bool:
    """检查 API 服务是否就绪"""
    _check_requests()
    try:
        r = requests.get(urljoin(_API_BASE_URL, "/health"), timeout=3)
        return r.status_code == 200
    except Exception:
        return _SERVER_STARTED


# ── MLModelHolder 兼容入口 ──

def _load_model(device: str = "auto", dtype: str = "float32"):
    """启动 Qwen API 服务（兼容 MLModelHolder 接口)"""
    if _ensure_server():
        return True
    raise ConnectionError("Qwen API 服务启动失败")


def load_model(device: str = "auto", dtype: str = "float32"):
    """加载 Qwen 强制对齐模型（自动启动子进程服务)"""
    return _load_model(device, dtype)


def unload_model():
    """关闭 Qwen API 服务,释放显存"""
    _stop_server()


def is_loaded() -> bool:
    """检查 Qwen API 服务是否就绪"""
    return is_server_ready()


def align_batch(
    items: list,
    language: str = "ja",
    device: str = "auto",
    dtype: str = "float32",
) -> List[list]:
    """批量对齐,自动启动/调用 REST API"""
    _check_requests()

    # 自动启动服务
    if not _ensure_server():
        return [[] for _ in items]

    # 构建请求体
    payload_items = []
    for item in items:
        audio_path = item[0]
        text = item[1]
        lang = item[2] if len(item) > 2 else language
        if not os.path.exists(audio_path):
            _log(f"音频不存在: {audio_path}")
            payload_items.append(["", text, lang])
            continue
        payload_items.append([os.path.abspath(audio_path), text, lang])

    try:
        r = requests.post(
            urljoin(_API_BASE_URL, "/align"),
            json={"items": payload_items},
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("results", [[] for _ in items])
    except requests.ConnectionError:
        _log("API 服务连接失败")
        return [[] for _ in items]
    except Exception as e:
        _log(f"API 调用失败: {e}")
        return [[] for _ in items]
