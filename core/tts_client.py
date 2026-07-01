"""TTS API 客户端模块 — 支持多种 API 模式"""

import json
import os
import sys
from datetime import datetime
from typing import Optional, Callable

import requests


class TTSClient:
    """灵活的 TTS API 客户端

    支持的 API 模式:
    - cosyvoice: JSON POST, 返回音频
    - gpt-sovits: Form-Data 上传, 返回音频
    - openai: OpenAI TTS 兼容
    - rainfall: 雨落版 API, GET /api/clone?text=&prompt_path=&seed=, 返回 WAV
    - custom: 用户自定义请求模板
    """

    def __init__(
        self,
        api_url: str = "http://localhost:9000",
        api_key: str = "",
        mode: str = "rainfall",
        model_name: str = "",
        language: str = "auto",
        timeout: int = 20,
        headers: Optional[dict] = None,
        extra_params: Optional[dict] = None,
    ):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.mode = mode
        self.model_name = model_name
        self.language = language
        self.timeout = timeout
        # 连接超时 50ms（本机服务快速失败),读取超时沿用 timeout（TTS 生成可能较慢)
        self._conn_timeout = 0.05
        self.headers = headers or {}
        self.extra_params = extra_params or {}

        # 默认头
        if self.api_key and 'Authorization' not in self.headers:
            self.headers['Authorization'] = f'Bearer {self.api_key}'

    def synthesize(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: Optional[str] = None,
        item_idx: int = 0,
        progress_callback: Optional[Callable] = None,
    ) -> bytes:
        """调用 TTS API 合成语音

        Args:
            text: 要合成的文本
            ref_audio_path: 参考音频路径（用于声音克隆)
            ref_text: 参考音频的文本（部分模型需要)
            item_idx: 字幕序号
            progress_callback: 进度回调

        Returns:
            音频字节数据 (WAV/MP3)
        """
        if self.mode == "rainfall":
            return self._call_rainfall(text, ref_audio_path, item_idx)
        elif self.mode == "cosyvoice":
            return self._call_cosyvoice(text, ref_audio_path, ref_text)
        elif self.mode == "gpt-sovits":
            return self._call_gptsovits(text, ref_audio_path, ref_text)
        elif self.mode == "openai":
            return self._call_openai(text, ref_audio_path)
        elif self.mode == "custom":
            return self._call_custom(text, ref_audio_path, ref_text)
        else:
            return self._call_cosyvoice(text, ref_audio_path, ref_text)

    def _encode_audio_base64(self, audio_path: str) -> str:
        """将音频文件编码为 base64"""
        import base64
        with open(audio_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _call_rainfall(self, text: str, ref_audio_path: str, item_idx: int = 0) -> bytes:
        """雨落版 API 调用 — GET /api/clone?text=&prompt_path=

        接口说明:
          GET /api/clone
          - text: 待合成文本
          - prompt_path: 参考音频路径
            - 绝对路径如 D:/voice.wav → 直接用
            - 相对路径如 demo_boy.wav → API 在自身 resources 中查找
          - seed: 随机种子（-1 表示随机)
          返回: audio/wav 原始数据
        """
        # 如果是绝对路径且文件存在,传绝对路径；否则传原值（让 API 自己查找)
        if os.path.isabs(ref_audio_path) and os.path.exists(ref_audio_path):
            prompt_path = os.path.abspath(ref_audio_path)
        else:
            prompt_path = ref_audio_path  # 相对路径,API 自己处理

        params = {
            "text": text,
            "prompt_path": prompt_path,
        }
        if self.language and self.language != "auto":
            params["language"] = self.language
        # 合并额外参数（如 seed)
        params.update(self.extra_params)

        # URL 构建: base 是 http://host:port,拼接 /api/clone
        base = self.api_url.rstrip('/')
        # 如果用户配的是 http://localhost:9000,就用 /api/clone
        # 如果配的是 http://localhost:9000/api/clone,直接用
        if base.endswith('/api/clone'):
            url = base
        else:
            url = f"{base}/api/clone"

        # debug: 打印请求参数（隐藏文本内容)
        _safe_params = {k: v for k, v in params.items() if k not in ('text', 'ref_text', 'prompt_text')}
        _ts = datetime.now().strftime("%m-%d %H:%M:%S")
        print(f"[{_ts}] [TTS API] [{item_idx}] GET {url} params={_safe_params}", file=sys.stderr)

        resp = requests.get(
            url,
            params=params,
            headers=self.headers,
            timeout=(self._conn_timeout, self.timeout)
        )

        # debug: 打印返回信息
        ct = resp.headers.get('Content-Type', '')
        print(f"[{_ts}] [TTS API] [{item_idx}] response: {resp.status_code} {ct} {len(resp.content)}bytes", file=sys.stderr)

        resp.raise_for_status()

        # 检查是否返回 WAV 音频（以 "RIFF" 开头)
        if not resp.content or len(resp.content) < 100:
            raise RuntimeError(f"API 返回数据过短 ({len(resp.content)} bytes)")

        content_type = resp.headers.get('Content-Type', '')
        if 'audio' not in content_type and not resp.content.startswith(b'RIFF'):
            # 可能返回了 JSON 错误
            try:
                err = resp.json()
                raise RuntimeError(f"API 错误: {err}")
            except (json.JSONDecodeError, ValueError):
                raise RuntimeError(
                    f"API 返回非音频数据 (Content-Type: {content_type})"
                )

        return resp.content

    def _call_cosyvoice(
        self, text: str, ref_audio_path: str, ref_text: Optional[str]
    ) -> bytes:
        """CosyVoice 风格 API 调用"""
        b64_audio = self._encode_audio_base64(ref_audio_path)

        payload = {
            "text": text,
            "mode": "sft",
            "ref_audio": b64_audio,
            "audio_format": "wav",
        }
        if self.model_name:
            payload["model"] = self.model_name
        if ref_text:
            payload["ref_text"] = ref_text
        if self.language and self.language != "auto":
            payload["language"] = self.language
        payload.update(self.extra_params)

        resp = requests.post(
            self.api_url,
            json=payload,
            headers={**self.headers, 'Content-Type': 'application/json'},
            timeout=(self._conn_timeout, self.timeout)
        )
        resp.raise_for_status()
        return resp.content

    def _call_gptsovits(
        self, text: str, ref_audio_path: str, ref_text: Optional[str]
    ) -> bytes:
        """GPT-SoVITS 风格 API 调用（form-data)"""
        text_lang = self.language if self.language != "auto" else "zh"

        with open(ref_audio_path, 'rb') as f:
            files = {
                'ref_audio': ('ref.wav', f, 'audio/wav'),
                'text': (None, text),
                'text_lang': (None, text_lang),
            }
            if ref_text:
                files['ref_text'] = (None, ref_text)
                files['ref_lang'] = (None, text_lang)

            for k, v in self.extra_params.items():
                files[k] = (None, str(v))

            resp = requests.post(
                self.api_url,
                files=files,
                headers=self.headers,
                timeout=(self._conn_timeout, self.timeout)
            )
        resp.raise_for_status()
        return resp.content

    def _call_openai(
        self, text: str, ref_audio_path: str
    ) -> bytes:
        """OpenAI TTS 兼容 API"""
        payload = {
            "model": self.model_name or "tts-1",
            "input": text,
            "voice": "alloy",
            "response_format": "wav",
        }
        payload.update(self.extra_params)

        resp = requests.post(
            self.api_url,
            json=payload,
            headers={**self.headers, 'Content-Type': 'application/json'},
            timeout=(self._conn_timeout, self.timeout)
        )
        resp.raise_for_status()
        return resp.content

    def _call_custom(
        self, text: str, ref_audio_path: str, ref_text: Optional[str]
    ) -> bytes:
        """自定义 API 调用 — 由 extra_params 中的模板控制"""
        b64_audio = self._encode_audio_base64(ref_audio_path)

        payload = {
            "text": text,
            "ref_audio": b64_audio,
        }
        if ref_text:
            payload["ref_text"] = ref_text
        if self.model_name:
            payload["model"] = self.model_name
        payload.update(self.extra_params)

        resp = requests.post(
            self.api_url,
            json=payload,
            headers={**self.headers, 'Content-Type': 'application/json'},
            timeout=(self._conn_timeout, self.timeout)
        )
        resp.raise_for_status()
        return resp.content

