import http.server
import socketserver
import threading
import webbrowser
import time
import os
import signal
from pathlib import Path
from urllib.parse import urlparse

PORT = 8000

# 如果你后面有自己启动的子进程，放到这里统一清理
child_processes = []

class FrontendHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/__shutdown__":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"shutting down")

            # 单独开线程，避免在当前请求线程里直接把 server 干掉
            threading.Thread(target=shutdown_app, daemon=True).start()
            return

        self.send_response(404)
        self.end_headers()

def shutdown_app():
    print("Browser page closed. Shutting down...")

    # 先停掉你自己维护的子进程
    for proc in child_processes:
        try:
            proc.terminate()
        except Exception:
            pass

    # 关闭 HTTP server
    global httpd
    if httpd is not None:
        httpd.shutdown()
        httpd.server_close()

    print("Server stopped.")
    os._exit(0)

httpd = None

def main():
    global httpd
    base_dir = Path(__file__).resolve().parent

    class HandlerFactory(FrontendHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(base_dir), **kwargs)

    with socketserver.TCPServer(("127.0.0.1", PORT), HandlerFactory) as server:
        httpd = server
        url = f"http://127.0.0.1:{PORT}/index.html"

        print(f"Serving frontend from: {base_dir}")
        print(f"Open in browser: {url}")

        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        time.sleep(0.5)
        webbrowser.open(url)

        try:
            print("Server is running. Close the browser tab/window to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping server manually...")
        finally:
            if httpd is not None:
                httpd.shutdown()
                httpd.server_close()
            print("Server stopped.")

if __name__ == "__main__":
    main()