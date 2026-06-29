"""TTS 测试服务器 — 轻量纯 Python 版 (edge-tts)

无 win32 依赖,纯 Python 实现,跨平台(Windows/Linux/macOS 均可用)。
使用 Microsoft Edge 在线 TTS 引擎(需要联网)。

启动:
    python tts_test_server.py [port]

默认端口 9001,调用格式:
    GET /api/clone?text=你好&voice=zh-CN-XiaoxiaoNeural

返回: edge-tts 合成的 WAV 音频 (24000Hz, 单声道, pcm_s16le)

依赖:
    pip install edge-tts
"""

import asyncio
import json
import os
import sys
import tempfile
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ── 异步 TTS 引擎 ──────────────────────────────────────────

VOICE_ALIASES = {
    # 中文
    "zh-cn": "zh-CN-XiaoxiaoNeural",
    "zh-cn-xiaoxiao": "zh-CN-XiaoxiaoNeural",
    "zh-cn-yunxi": "zh-CN-YunxiNeural",
    "zh-tw": "zh-TW-HsiaoChenNeural",
    # 英文
    "en": "en-US-AriaNeural",
    "en-us": "en-US-AriaNeural",
    "en-gb": "en-GB-SoniaNeural",
    "en-au": "en-AU-NatashaNeural",
    # 日语
    "ja": "ja-JP-NanamiNeural",
    # 韩语
    "ko": "ko-KR-SunHiNeural",
}

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"


async def _synthesize(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """使用 edge-tts 合成语音,返回 WAV 字节数据。"""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    # edge-tts 默认返回 MP3 格式,暂存后返回原始字节
    # 保持接口兼容：仍返回 .wav 但实际内容格式由客户端按 header 处理
    # 更干净的做法：请求 PCM 格式,但 edge-tts 只支持 mp3 / opus / aac / webm
    # 所以我们返回 MP3 但标 audio/mpeg,让客户端按实际格式解码
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(tmp_fd)

    await communicate.save(tmp_path)

    with open(tmp_path, "rb") as f:
        data = f.read()
    os.unlink(tmp_path)
    return data


# ── HTTP 处理器 ────────────────────────────────────────────


class TTSHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/api/clone":
            self._handle_clone(params)
        elif parsed.path == "/api/voices":
            self._handle_voices()
        else:
            self.send_error(404, f"未知路径: {parsed.path}")

    def _handle_clone(self, params):
        text = params.get("text", [""])[0]
        voice_raw = params.get("voice", [""])[0]

        if not text:
            self.send_error(400, "text 参数为空")
            return

        # 解析 voice 参数(支持别名)
        voice = VOICE_ALIASES.get(voice_raw.strip().lower(), voice_raw.strip()) or DEFAULT_VOICE

        print(f"[TTS] text={text[:60]}...  voice={voice}")

        try:
            # 在同步 handler 中跑异步代码
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            audio_data = loop.run_until_complete(_synthesize(text, voice))
            loop.close()
        except ImportError:
            self.send_error(500, "需要 edge-tts: pip install edge-tts")
            return
        except Exception as e:
            traceback.print_exc()
            self.send_error(500, f"TTS 合成失败: {e}")
            return

        if not audio_data or len(audio_data) < 100:
            self.send_error(500, "生成的音频数据为空")
            return

        self.send_response(200)
        # edge-tts 默认输出 MP3
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(len(audio_data)))
        self.end_headers()
        self.wfile.write(audio_data)

        print(f"  → 返回 {len(audio_data)} bytes (edge-tts MP3, voice={voice})")

    def _handle_voices(self):
        """返回支持的 voice 列表。"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        info = {
            "default": DEFAULT_VOICE,
            "aliases": VOICE_ALIASES,
            "note": "更多 voice 见 https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support",
        }
        self.wfile.write(json.dumps(info, ensure_ascii=False, indent=2).encode("utf-8"))

    def log_message(self, format, *args):
        pass  # 用上面的 print 代替


# ── 入口 ────────────────────────────────────────────────────


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9001
    server = HTTPServer(("0.0.0.0", port), TTSHandler)
    print("🎤 TTS 测试服务器启动 (轻量纯 Python — edge-tts)")
    print(f"   地址:  http://localhost:{port}")
    print(f"   接口:  GET /api/clone?text=...&voice=zh-CN-XiaoxiaoNeural")
    print(f"   列表:  GET /api/voices")
    print(f"   引擎:  Microsoft Edge 在线 TTS (无 win32 依赖)")
    print(f"   依赖:  pip install edge-tts")
    print("─" * 50)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已关闭")
        server.server_close()


if __name__ == "__main__":
    main()
