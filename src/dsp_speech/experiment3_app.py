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
    "/": "overview",
    "/index.html": "overview",
    "/speaker.html": "speaker",
    "/digits.html": "digits",
    "/docs.html": "docs",
    "/templates.html": "docs",
}
MAX_AUDIO_BYTES = 8 * 96_000 * 4
INTEGRATION_TAG = b'<script src="/integration.js?v=mobile-cycle-saturation-3"></script>'
DESKTOP_STYLE = b"""<style id=\"e3-desktop-mic-style\">
    #page-digits .desktop-mic-row{display:flex;align-items:center;gap:10px;min-width:0}
    #page-digits .desktop-mic-controls{display:flex;align-items:center;gap:10px;min-width:0;font-size:12px;white-space:nowrap}
    #page-digits .desktop-mic-controls select{flex:1;min-width:0;max-width:100%;min-height:42px;padding:0 28px 0 10px;border:1px solid rgba(24,34,30,.28);border-radius:2px;appearance:none;background-color:rgba(250,226,177,.24);background-image:linear-gradient(45deg,transparent 50%,#1f201b 50%),linear-gradient(135deg,#1f201b 50%,transparent 50%);background-position:calc(100% - 13px) 50%,calc(100% - 8px) 50%;background-size:5px 5px;background-repeat:no-repeat;backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);color:#111;font:inherit}
    #page-digits .desktop-mic-refresh{min-height:42px;padding:0 14px;font-size:12px;white-space:nowrap;flex:0 0 auto}
    #page-digits .desktop-mic-hint{font-size:11px;line-height:1.45;opacity:.72}
    #page-digits .frame-top .sys-title,#page-digits .frame-top .sys-subtitle{text-shadow:none;filter:none;mix-blend-mode:normal}
    #page-digits .recognition-progress{display:none;width:100%;height:4px;overflow:hidden;background:rgba(0,0,0,.14)}
    #page-digits .recognition-progress.is-active{display:block}
    #page-digits .recognition-progress span{display:block;width:34%;height:100%;background:#17574c;animation:e3-progress 1.2s ease-in-out infinite}
    @keyframes e3-progress{0%{transform:translateX(-110%)}100%{transform:translateX(330%)}}
    #page-digits #wave-digit canvas{display:block;width:100%;height:100%}
    #page-digits .dtw-row[hidden]{display:none!important}
    #page-digits .dtw-row.is-best .dtw-label,#page-digits .dtw-row.is-best .dtw-value{font-weight:700}
    @media(max-width:900px){#page-digits .desktop-mic-row{flex-wrap:wrap}#page-digits .desktop-mic-controls{flex:1 1 100%;white-space:normal}#page-digits .desktop-mic-refresh{flex:1 1 100%}}
    </style>"""
TRANSITION_STYLE = b"""<style id=\"e3-page-transition\">
    :root{--e3-enter:cubic-bezier(.23,1,.32,1)}
    html,body{width:100%;height:100%;height:100dvh;min-height:100%;overflow:hidden}
    #app-container{height:100%;height:100dvh;min-height:100%;overflow:hidden}
    #app-container .screen{transform:none!important;transition:opacity 260ms var(--e3-enter)!important;will-change:opacity}
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
        self.send_header("Vary", "User-Agent, Sec-CH-UA-Mobile")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in PAGE_ROUTES:
            mobile = self._is_mobile_client()
            page_path = DESIGN_DIR / ("mobile.html" if mobile else "index.html")
            try:
                body = page_path.read_bytes()
            except OSError as error:
                self._json(500, {"error": f"界面文件读取失败：{error}"})
                return
            if b"</head>" in body:
                if mobile:
                    initial_view = {
                        "/speaker.html": "screen-speaker",
                        "/digits.html": "screen-digit",
                        "/docs.html": "screen-docs",
                        "/templates.html": "screen-docs",
                    }.get(path, "screen-overview")
                    boot = f'<script>document.documentElement.dataset.startView="{initial_view}";</script>'.encode("ascii")
                else:
                    initial_page = {
                        "/speaker.html": "page-speaker",
                        "/digits.html": "page-digits",
                        "/docs.html": "page-docs",
                        "/templates.html": "page-docs",
                    }.get(path, "page-home")
                    boot = f'<script>document.documentElement.dataset.startPage="{initial_page}";</script>'.encode("ascii")
                body = body.replace(b"</head>", (TRANSITION_STYLE if mobile else DESKTOP_STYLE) + boot + b"</head>", 1)
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

    def _is_mobile_client(self) -> bool:
        user_agent = self.headers.get("User-Agent", "").lower()
        client_hint = self.headers.get("Sec-CH-UA-Mobile", "")
        return client_hint == "?1" or any(token in user_agent for token in ("iphone", "ipad", "ipod", "android", "mobile"))

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
