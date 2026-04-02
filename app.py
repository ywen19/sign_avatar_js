import http.server
import socketserver
import threading
import webbrowser
import time
import os
import json
from pathlib import Path
from urllib.parse import urlparse
import signal
import atexit

from switch_anim import TestAnimLoader
from language_utils.smollm_service import load_model, get_response, classify_context_need
from language_utils.chat_history_store import ChatHistoryStore


PORT = 8000
httpd = None
latest_llm_answer = ""
conversation_history = []

MAX_HISTORY_MESSAGES = 3
TRIM_TO_MESSAGES = 5

history_store = ChatHistoryStore("chat_history.jsonl")

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

        if parsed.path == "/api/text":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(content_length)
                data = json.loads(raw_body.decode("utf-8"))

                user_text = data.get("text", "").strip()
                if not user_text:
                    self._send_json({"error": "Empty text input"}, status=400)
                    return

                print("[TEXT INPUT]", user_text)

                global latest_llm_answer, conversation_history

                context_type = classify_context_need(user_text)
                print("[CONTEXT TYPE]", context_type)

                if context_type == "SELF_CONTAINED":
                    # no history needed
                    latest_llm_answer = get_response(user_text)

                elif context_type == "RECENT_CONTEXT":
                    # use recent in-memory history
                    latest_llm_answer = get_response(
                        user_text,
                        conversation_history=conversation_history[-10:]
                    )

                elif context_type == "ARCHIVE_CONTEXT":
                    archived_messages = history_store.search_messages(
                        user_text,
                        limit=6
                    )

                    combined_history = archived_messages + conversation_history[-10:]

                    latest_llm_answer = get_response(
                        user_text,
                        conversation_history=combined_history
                    )

                else:
                    # fallback safety
                    latest_llm_answer = get_response(
                        user_text,
                        conversation_history=conversation_history[-10:]
                    )

                user_message = {
                    "role": "user",
                    "content": user_text
                }
                assistant_message = {
                    "role": "assistant",
                    "content": latest_llm_answer
                }

                conversation_history.append(user_message)
                conversation_history.append(assistant_message)

                history_store.append_messages([user_message, assistant_message])

                if len(conversation_history) > MAX_HISTORY_MESSAGES:
                    conversation_history = conversation_history[-TRIM_TO_MESSAGES:]

                self._send_json({
                    "ok": True,
                    "received_text": user_text,
                    "answer_text": latest_llm_answer
                }, status=200)

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


def cleanup_app_resources():
    try:
        history_store.delete_file()
        print("Deleted chat history JSONL.")
    except Exception as e:
        print("Failed to delete chat history JSONL:", e)

def handle_exit_signal(signum, frame):
    global httpd
    print(f"Received signal {signum}. Shutting down app...")

    cleanup_app_resources()

    if httpd is not None:
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception as e:
            print("Error while shutting down server:", e)

    os._exit(0)

def shutdown_app():
    global httpd
    print("Browser page closed. Shutting down app...")

    cleanup_app_resources()

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

    load_model()

    atexit.register(cleanup_app_resources)
    signal.signal(signal.SIGINT, handle_exit_signal)   # Ctrl+C
    signal.signal(signal.SIGTERM, handle_exit_signal)

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