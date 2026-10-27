"""Local browser UI and HTTP bridge for live microphone digit recognition."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .recognize import load_digit_templates, recognize_utterance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DESIGN_DIR = Path(__file__).resolve().parent / "web" / "design_sources"
PAGE_ROUTES = {
    "/": "mobile.html",
    "/index.html": "mobile.html",
    "/speaker.html": "mobile.html",
    "/digits.html": "mobile.html",
    "/docs.html": "mobile.html",
    "/templates.html": "mobile.html",
}
MAX_AUDIO_BYTES = 8 * 96_000 * 4
INTEGRATION_TAG = b'<script src="/integration.js?v=mobile-spa-1"></script>'
TRANSITION_STYLE = b"""<style id=\"e3-page-transition\">
    :root{--e3-enter:cubic-bezier(.23,1,.32,1)}
    html,body{width:100%;height:100%;height:100dvh;min-height:100%;overflow:hidden}
    #app-container{height:100%;height:100dvh;min-height:100%;overflow:hidden}
    #app-container .screen{transform:none!important;transition:opacity 240ms var(--e3-enter)!important;will-change:opacity}
    #app-container .screen.hidden-left,#app-container .screen.hidden-right{visibility:hidden;opacity:0;transform:none!important}
    #app-container .screen.active,#app-container .screen.is-entering,#app-container .screen.is-leaving{transform:none!important}
    #app-container .screen.is-entering,#app-container .screen.is-leaving{visibility:visible;opacity:0;pointer-events:none}
    #app-container .screen.active{visibility:visible;opacity:1;pointer-events:auto}
    #app-container .screen.active .stagger-el{transform:none!important;transition-property:opacity!important;transition-duration:220ms!important;transition-delay:0ms!important}
    html[data-start-view="screen-speaker"] #screen-speaker,
    html[data-start-view="screen-digit"] #screen-digit,
    html[data-start-view="screen-docs"] #screen-docs{visibility:visible!important;opacity:1!important;pointer-events:auto!important;transform:none!important}
    html[data-start-view="screen-speaker"] #screen-overview,
    html[data-start-view="screen-digit"] #screen-overview,
    html[data-start-view="screen-docs"] #screen-overview{visibility:hidden!important;opacity:0!important;pointer-events:none!important}
    .mobile-mic-controls button,.mobile-record-controls button{touch-action:manipulation;min-height:44px}
    .page-content{overscroll-behavior:contain;-webkit-overflow-scrolling:touch;padding-bottom:max(24px,env(safe-area-inset-bottom))}
    #glcanvas{z-index:0}
    @media(prefers-reduced-motion:reduce){#app-container .screen,#app-container .screen.active .stagger-el{transition-duration:1ms!important}}
    </style>"""


class RecognitionHandler(BaseHTTPRequestHandler):
    templates: dict[str, list[np.ndarray]] = {}

    def log_message(self, format: str, *args: object) -> None:
        print(f"[网页] {self.address_string()} - {format % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in PAGE_ROUTES:
            page_path = DESIGN_DIR / PAGE_ROUTES[path]
            try:
                body = page_path.read_bytes()
            except OSError as error:
                self._json(500, {"error": f"界面文件读取失败：{error}"})
                return
            if b"</head>" in body:
                initial_view = {
                    "/speaker.html": "screen-speaker",
                    "/digits.html": "screen-digit",
                    "/docs.html": "screen-docs",
                    "/templates.html": "screen-docs",
                }.get(path, "screen-overview")
                boot = f'<script>document.documentElement.dataset.startView="{initial_view}";</script>'.encode("ascii")
                body = body.replace(b"</head>", TRANSITION_STYLE + boot + b"</head>", 1)
            if b"</body>" in body:
                body = body.replace(b"</body>", INTEGRATION_TAG + b"</body>", 1)
            self._send(200, body, "text/html; charset=utf-8")
            return
        if path == "/integration.js":
            try:
                body = (DESIGN_DIR / "integration.js").read_bytes()
            except OSError as error:
                self._json(500, {"error": f"界面接口文件读取失败：{error}"})
                return
            self._send(200, body, "text/javascript; charset=utf-8")
            return
        if path == "/health":
            self._json(
                200,
                {
                    "ready": bool(self.templates),
                    "template_count": sum(len(items) for items in self.templates.values()),
                    "digits": {label: len(items) for label, items in self.templates.items()},
                },
            )
            return
        self._json(404, {"error": "页面不存在"})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/recognize":
            self._json(404, {"error": "接口不存在"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            sample_rate = int(self.headers.get("X-Sample-Rate", "0"))
            if content_length <= 0 or content_length > MAX_AUDIO_BYTES:
                raise ValueError("录音长度不正确，请控制在8秒以内")
            if sample_rate < 8_000 or sample_rate > 96_000:
                raise ValueError("浏览器音频采样率不受支持")
            if content_length % 4:
                raise ValueError("收到的音频数据不完整")
            raw = self.rfile.read(content_length)
            if len(raw) != content_length:
                raise ValueError("录音数据传输不完整，请重新录制")
            samples = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
            result = recognize_utterance(samples, sample_rate, self.templates)
            self._json(200, result)
        except (ValueError, OSError) as error:
            self._json(400, {"error": str(error)})
        except Exception as error:  # Surface an actionable message without exposing a traceback.
            self._json(500, {"error": f"识别流程发生错误：{error}"})


def main() -> None:
    port = int(os.environ.get("PORT", "8765"))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    print("正在读取 data/processed/metadata.csv 并建立数字模板……")
    templates = load_digit_templates(PROJECT_ROOT)
    RecognitionHandler.templates = templates
    server = ThreadingHTTPServer((host, port), RecognitionHandler)
    print(f"实验三实时识别已启动：http://{host}:{port}")
    print(f"模板数量：{sum(len(items) for items in templates.values())} 条（数字0—9）")
    print("在浏览器打开页面并允许麦克风权限；按 Ctrl+C 可退出。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭实验三程序……")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
