"""声纹性别检测 — WavLM-Large-Age-Sex 模型直接推理（仅性别)

模型权重: models/wavlm-large-age-sex/  (本地加载,不依赖 HuggingFace 下载)
依赖: pip install transformers torch soundfile loralib
"""

import os
import soundfile as sf

from .model_base import MLModelHolder
from .utils import resolve_device


class _WavLMHolder(MLModelHolder):
    """WavLM 性别检测模型的单例生命周期管理"""

    @classmethod
    def _load_impl(cls, device, dtype):
        from .wavlm_model import WavLMWrapper

        local_path = _model_dir()
        if not os.path.exists(os.path.join(local_path, "model.safetensors")):
            raise FileNotFoundError(
                f"模型权重未找到: {local_path}。"
                f"请将 tiantiaf/wavlm-large-age-sex 的 model.safetensors 和 config.json 放入该目录。"
            )

        m = WavLMWrapper.from_pretrained(local_path)
        m.eval()
        _device = resolve_device(device or "auto")
        if _device != "cpu":
            m = m.to(_device)
        return m


def _model_dir() -> str:
    """返回模型权重所在目录"""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "wavlm-large-age-sex",
    )


def _load_model():
    """懒加载 WavLM 模型（仅从本地加载)"""
    return _WavLMHolder.load()


def unload_model():
    """释放模型显存,调用后模型将被卸载"""
    _WavLMHolder.unload(move_to_cpu=True)


def _load_audio(audio_path: str):
    """读取并预处理音频 → 16kHz 单声道 numpy 数组"""
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if len(audio) > 15 * 16000:
        audio = audio[:15 * 16000]
    return audio


def _parse_sex_logits(sex_logits):
    """从 sex_logits 解析性别结果列表"""
    import torch
    sex_probs = torch.softmax(sex_logits, dim=1)
    results = []
    for i in range(sex_logits.shape[0]):
        fp = sex_probs[i, 0].item()
        mp = sex_probs[i, 1].item()
        conf = max(fp, mp)
        gender = "female" if fp > mp else "male"
        results.append({"gender": gender, "confidence": conf, "error": None})
    return results


def detect_gender(audio_path: str) -> dict:
    """使用 WavLM 模型检测音频的性别

    Args:
        audio_path: 音频文件路径（16kHz 单声道 WAV 最佳, 自动重采样)

    Returns:
        {"gender": "male" | "female" | "", "confidence": float, "error": str | None}
    """
    if not audio_path or not os.path.exists(audio_path):
        return {"gender": "", "confidence": 0.0,
                "error": f"文件不存在: {audio_path}"}

    try:
        model = _load_model()
    except Exception as e:
        return {"gender": "", "confidence": 0.0,
                "error": f"模型加载失败: {e}"}

    try:
        import torch
        device = next(model.parameters()).device
        audio = _load_audio(audio_path)
        input_tensor = torch.from_numpy(audio).float().to(device).unsqueeze(0)

        with torch.no_grad():
            _, sex_logits = model(input_tensor)

        return _parse_sex_logits(sex_logits)[0]

    except Exception as e:
        return {"gender": "", "confidence": 0.0,
                "error": f"{type(e).__name__}: {e}"}


def detect_genders_batch(audio_paths: list) -> list:
    """批量检测性别,一批音频共享一次模型推理（自动 padding)

    Args:
        audio_paths: 音频文件路径列表

    Returns:
        [{"gender": ..., "confidence": ..., "error": ...}, ...]
    """
    if not audio_paths:
        return []

    try:
        model = _load_model()
    except Exception as e:
        return [{"gender": "", "confidence": 0.0, "error": f"模型加载失败: {e}"}
                for _ in audio_paths]

    try:
        import torch
        import torch.nn.functional as F

        device = next(model.parameters()).device

        # 加载并预处理所有音频
        tensors = []
        for path in audio_paths:
            if not path or not os.path.exists(path):
                tensors.append(None)
                continue
            try:
                audio = _load_audio(path)
                tensors.append(torch.from_numpy(audio).float().to(device))
            except Exception:
                tensors.append(None)

        # 排除无效项
        valid_indices = [i for i, t in enumerate(tensors) if t is not None]
        if not valid_indices:
            return [{"gender": "", "confidence": 0.0, "error": "所有音频文件无效"}
                    for _ in audio_paths]

        valid_tensors = [tensors[i] for i in valid_indices]

        # padding 到 batch 内最大长度
        max_len = max(t.shape[0] for t in valid_tensors)
        padded = torch.stack([
            F.pad(t, (0, max_len - t.shape[0])) for t in valid_tensors
        ])

        # 批推理
        with torch.no_grad():
            _, sex_logits = model(padded)

        batch_results = _parse_sex_logits(sex_logits)

        # 映射回原始顺序
        results = [{"gender": "", "confidence": 0.0, "error": "音频无效"}
                   for _ in audio_paths]
        for orig_idx, result in zip(valid_indices, batch_results):
            results[orig_idx] = result

        return results

    except Exception as e:
        return [{"gender": "", "confidence": 0.0,
                 "error": f"{type(e).__name__}: {e}"} for _ in audio_paths]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <音频文件.wav> [...]")
        sys.exit(1)
    if len(sys.argv) == 2:
        result = detect_gender(sys.argv[1])
        g = {"male": "男", "female": "女"}.get(result["gender"], "未定")
        print(f"性别: {g}  ({result['confidence']:.1%})")
        if result["error"]:
            print(f"错误: {result['error']}")
    else:
        results = detect_genders_batch(sys.argv[1:])
        for path, r in zip(sys.argv[1:], results):
            g = {"male": "男", "female": "女"}.get(r["gender"], "未定")
            print(f"{path}: {g} ({r['confidence']:.1%})")
            if r["error"]:
                print(f"  └─ 错误: {r['error']}")
