"""Windows SAPI (pywin32) TTS API 服务器 — 快速测试用

启动:
    python tts_api_server.py [port]

兼容 rainfall 模式: GET /api/clone?text=你好&prompt_path=xxx
"""

import os
import sys
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

try:
    import win32com.client
except ImportError:
    print("需要 pywin32: pip install pywin32")
    sys.exit(1)


class TTSHandler(BaseHTTPRequestHandler):
    """处理 TTS 合成请求,返回 WAV 音频"""

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        text = params.get("text", [""])[0]
        if not text:
            self._send_error(400, "缺少 text 参数")
            return
        try:
            wav_data = self._synthesize(text)
            self._send_wav(wav_data)
        except Exception as e:
            self._send_error(500, str(e))

    def do_POST(self):
        """兼容 JSON POST 请求(CosyVoice 风格)"""
        import json
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_error(400, "请求体为空")
            return
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_error(400, "JSON 解析失败")
            return
        text = data.get("text", "")
        if not text:
            self._send_error(400, "缺少 text 字段")
            return
        try:
            wav_data = self._synthesize(text)
            self._send_wav(wav_data)
        except Exception as e:
            self._send_error(500, str(e))

    def _synthesize(self, text: str) -> bytes:
        """使用 Windows SAPI 合成语音,返回 WAV 字节数据"""
        # 生成临时 WAV 文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            # 16kHz 16-bit mono WAV
            fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
            fmt.Type = 4  # SAFT16kHz16BitMono
            stream.Format = fmt
            stream.Open(tmp_path, 3)  # SSFMCreateForWrite
            speaker.AudioOutputStream = stream
            speaker.Speak(text)
            stream.Close()
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _send_wav(self, data: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code: int, msg: str):
        import json
        body = json.dumps({"error": msg}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[TTS] {args[0]} {args[1]} {args[2]}")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9001
    server = HTTPServer(("0.0.0.0", port), TTSHandler)
    print(f"Windows SAPI TTS 服务器启动: http://localhost:{port}")
    print("测试: curl 'http://localhost:{}/api/clone?text=你好世界'".format(port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()


if __name__ == "__main__":
    main()
