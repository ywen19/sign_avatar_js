import http.server
import socketserver
import threading
import webbrowser
import time
import os
import json
from pathlib import Path
from urllib.parse import urlparse

from switch_anim import TestAnimLoader

PORT = 8000
httpd = None

loader = TestAnimLoader(
    default_json="Dancing_mixamo_com_frames.json"
)


class FrontendHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format, *args):
        print("[HTTP]", format % args)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/start":
            try:
                payload = loader.get_default_payload()
                self._send_json(payload, status=200)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/end":
            try:
                payload = loader.load_payload(
                    "Headbutt_mixamo_com_frames.json",
                    animation_name="headbutt",
                    camera_state="end",
                )
                self._send_json(payload, status=200)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if parsed.path == "/__shutdown__":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"shutting down")
            threading.Thread(target=shutdown_app, daemon=True).start()
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"not found")


def shutdown_app():
    global httpd
    print("Browser page closed. Shutting down app...")

    if httpd is not None:
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception as e:
            print("Error while shutting down server:", e)

    print("Server stopped.")
    os._exit(0)


def main():
    global httpd

    base_dir = Path(__file__).resolve().parent
    index_file = base_dir / "templates" / "index.html"

    if not index_file.exists():
        print(f"Error: {index_file} not found.")
        return

    class HandlerFactory(FrontendHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(base_dir), **kwargs)

    with socketserver.TCPServer(("127.0.0.1", PORT), HandlerFactory) as server:
        httpd = server
        url = f"http://127.0.0.1:{PORT}/templates/index.html"

        print(f"Serving frontend from: {base_dir}")
        print(f"Open in browser: {url}")

        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        time.sleep(0.5)
        webbrowser.open(url)

        try:
            print("App is running. Close the browser tab/window to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping app manually...")
        finally:
            if httpd is not None:
                try:
                    httpd.shutdown()
                    httpd.server_close()
                except Exception:
                    pass
            print("App stopped.")


if __name__ == "__main__":
    main()