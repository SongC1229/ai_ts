"""Qwen3 ForcedAligner REST API 服务

启动:  cd qwen_api && uv run python qwen_api_server.py [--port 8765]

特性:
  - 首次 /align 请求时自动加载模型
  - 空闲 3 秒后自动卸载释放显存
  - 默认 CUDA + FP16 推理

API:
  POST /align  {"items": [["audio_path", "text", "ja"], ...]}
  GET /health  {"status": "ok", "model_loaded": true}
"""
import os
import time
import argparse
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Qwen ForcedAligner API", version="0.1.0")


def _resolve_device(device: str) -> str:
    """设备检测：cuda > cpu"""
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# 模型状态
_model = None
_loaded = False
_unload_timer: Optional[threading.Timer] = None
_IDLE_UNLOAD_SEC = 5  # 空闲 5 秒后卸载,下次请求自动重新加载


class AlignRequest(BaseModel):
    items: list[list[str]]


class AlignResponse(BaseModel):
    results: list[list[dict]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def load_model(device: str = "auto", dtype: str = "bfloat16"):
    """加载 Qwen3ForcedAligner 模型（单例)"""
    global _model, _loaded
    if _loaded and _model is not None:
        return _model

    from qwen_asr import Qwen3ForcedAligner

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "qwen3-forced-aligner"
    )
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    if device == "auto":
        device = _resolve_device("auto")

    dtype_map = {"float32": None, "bfloat16": "bfloat16", "float16": "float16"}
    model_dtype = dtype_map.get(dtype)

    attn_impl = None
    if device.startswith("cuda"):
        try:
            import flash_attn  # noqa
            attn_impl = "flash_attention_2"
        except ImportError:
            pass

    load_kwargs = {"device_map": device if device.startswith("cuda") else None}
    if model_dtype is not None:
        load_kwargs["dtype"] = model_dtype
    if attn_impl is not None:
        load_kwargs["attn_implementation"] = attn_impl

    print("[qwen-api] 加载模型 ...")
    print(f"[qwen-api] 设备: {device}, Dtype: {dtype}")
    t0 = time.time()
    _model = Qwen3ForcedAligner.from_pretrained(model_path, **load_kwargs)
    _loaded = True
    print(f"[qwen-api] 加载完成 ({time.time()-t0:.1f}s)")
    print(f"[qwen-api] 会话空闲 {_IDLE_UNLOAD_SEC}s 后自动卸载")
    return _model


def unload_model():
    """卸载模型释放显存"""
    global _model, _loaded
    if _model is None:
        return
    try:
        import torch
        _model = _model.cpu() if hasattr(_model, 'cpu') else _model
    except Exception:
        pass
    _model = None
    _loaded = False
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("[qwen-api] 模型已卸载,显存已释放")


def _cancel_unload():
    """取消待执行的卸载计时器"""
    global _unload_timer
    if _unload_timer is not None:
        _unload_timer.cancel()
        _unload_timer = None


def _schedule_unload():
    """N 秒后自动卸载"""
    global _unload_timer
    _cancel_unload()
    _unload_timer = threading.Timer(_IDLE_UNLOAD_SEC, unload_model)
    _unload_timer.daemon = True
    _unload_timer.start()


@app.post("/align", response_model=AlignResponse)
def align(req: AlignRequest):
    """对齐接口：首次调用自动加载模型,空闲 N 秒卸载"""
    _cancel_unload()

    try:
        if not _loaded or _model is None:
            load_model(device=args.device, dtype=args.dtype)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"模型加载失败: {e}")

    audio_paths, texts, languages = [], [], []
    for item in req.items:
        if len(item) >= 3:
            audio_paths.append(item[0])
            texts.append(item[1])
            languages.append(item[2])
        elif len(item) == 2:
            audio_paths.append(item[0])
            texts.append(item[1])
            languages.append("ja")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid item: {item}")

    try:
        raw_results = _model.align(audio_paths, texts, languages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    all_segments = []
    for result in raw_results:
        segments = []
        if result:
            for r in result:
                segments.append({
                    "word": getattr(r, 'text', ''),
                    "start_ms": getattr(r, 'start_time', 0) * 1000,
                    "end_ms": getattr(r, 'end_time', 0) * 1000,
                    "confidence": getattr(r, 'confidence', 0) or 0,
                })
        all_segments.append(segments)

    _schedule_unload()
    return AlignResponse(results=all_segments)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=_loaded)


@app.post("/shutdown")
def shutdown():
    """关闭服务（供客户端清理残留用)"""
    import os
    _cancel_unload()
    unload_model()
    # 延迟退出,让响应返回
    import threading as _th
    _th.Timer(0.5, os._exit, args=[0]).start()
    return {"status": "shutting_down"}


def main():
    global args
    parser = argparse.ArgumentParser(description="Qwen ForcedAligner API")
    parser.add_argument("--port", type=int, default=8765, help="服务端口")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    parser.add_argument("--device", type=str, default="auto", help="推理设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="推理精度（bfloat16/float16/float32)")
    parser.add_argument("--idle", type=int, default=5, help="空闲 N 秒后卸载,下次请求自动重载")
    args = parser.parse_args()
    global _IDLE_UNLOAD_SEC
    _IDLE_UNLOAD_SEC = args.idle
    print("[qwen-api] Qwen ForcedAligner API 启动中 ...")
    print(f"[qwen-api] 空闲卸载: {_IDLE_UNLOAD_SEC}s")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
