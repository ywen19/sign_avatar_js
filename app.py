"""
This module provides the local backend server and application entry point for the
sign avatar application.

The backend serves frontend files through Python's HTTP server, receives text input
from the frontend, generates a language-model response, retrieves matching sign
motions, and returns a Three.js-compatible animation payload as JSON.

The module also manages conversation history, application shutdown, model startup,
and automatic browser launch. It uses http.server and socketserver directly.
"""

import http.server
import errno
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
from types import FrameType
from typing import Any, Dict, List, Optional

from switch_anim import TestAnimLoader
from language_utils import *
from motion_retrieval import (
    build_threejs_motion_from_tokens,
    flatten_traced_tokens,
    retrieve_json_paths,
)


PORT: int = 8000  # preferred local server port
# current server instance; does not exist before startup
httpd: Optional[socketserver.TCPServer] = None
 
latest_llm_answer: str = ""

# multi-turn conversation history lookup 
conversation_history: List[Dict[str, str]] = []
MAX_HISTORY_MESSAGES: int = 15
TRIM_TO_MESSAGES: int = 6
history_store = ChatHistoryStore("chat_history.jsonl")

# loading and normalizing local animation JSON files for frontend
loader = TestAnimLoader(
    default_json="tmp/default_motion.json"
)


def process_answer_text(answer_text: str) -> Dict[str, Any]:
    """
    Process an assistant answer through the language and motion-token pipeline.

    The answer is divided into sentences, analyzed for entities, normalized, 
    tokenized, and traced to tokens that can subsequently be matched to motion 
    JSON files.

    answer_text  (str)  :  assistant answer to process

    Return:
    (Dict[str, Any])    :  processed llm answer for motion concatenation
    """
    raw_sentences = break_into_sentences(answer_text)

    # detect entities from answer sentences
    entities = []
    for sentence in raw_sentences:
        entities.extend(detect_entities(sentence))
    
    # normalize sentence: remove special symbols, lower case...
    normalized_sentences = [
        normalize_sentence_for_match(sentence)
        for sentence in raw_sentences
    ]

    # tokenize text while preserving detected entities as single chunks
    tokenized_sentences = [
        tokenize_with_entities(sentence, entities)
        for sentence in normalized_sentences
    ]

    # find matching chunks(tokens) in vocabs
    traced_tokens = []
    for tokens in tokenized_sentences:
        traced, tags = trace_tokens(tokens, entities)

        traced_tokens.append(traced)

    return {
        "sentences": raw_sentences,
        "entities": entities,
        "normalized_sentences": normalized_sentences,
        "tokenized_sentences": tokenized_sentences,
        "traced_tokens": traced_tokens,
    }


def print_pipeline_debug(debug_data: Dict[str, Any]) -> None:
    """
    Debug helper to print intermediate language-processing results for 
    debugging.

    debug_data  (Dict[str, Any])  :  processed llm answer
    """
    print("\n--- PIPELINE DEBUG ---")
    print("[SENTENCES]")
    for i, sentence in enumerate(debug_data["sentences"], 1):
        print(f"{i}. {sentence}")

    print("\n[ENTITIES]")
    print(debug_data["entities"])

    print("\n[NORMALIZED]")
    for item in debug_data["normalized_sentences"]:
        print(item)

    print("\n[TOKENIZED WITH ENTITIES]")
    for item in debug_data["tokenized_sentences"]:
        print(item)

    print("\n[TRACED TOKENS]")
    for item in debug_data["traced_tokens"]:
        print(item)

    print("----------------------\n")


def print_json_paths(debug_data: Dict[str, Any]) -> None:
    """
    Debug helper to print the motion JSON path found for each traced token.

    Missing motion matches are marked explicitly in the debug output.

    debug_data  (Dict[str, Any])  :  processed llm answer
    """
    print("\n[JSON PATHS]")

    for sentence_idx, traced_tokens in enumerate(debug_data["traced_tokens"], 1):
        json_paths = retrieve_json_paths(traced_tokens)

        print(f"\nSentence {sentence_idx}:")
        for token, json_path in zip(traced_tokens, json_paths):
            if json_path is None:
                print(f"[MISSING] {token}")
            else:
                print(f"[FOUND]   {token} -> {json_path}")

    print()



class ReusableTCPServer(socketserver.TCPServer):
    """
    Provide a TCP server whose local address can be reused after shutdown.

    TCP server is a program that listens for network connections on a specific 
    host and port.

    The communication flow is:
    Browser → HTTP request → TCP connection → Python server
    Browser ← HTTP response ← TCP connection ← Python server
    """

    allow_reuse_address = True


def create_server(
    handler_factory: Any,
    preferred_port: int = PORT,
    host: str = "127.0.0.1",
    max_attempts: int = 20,
) -> ReusableTCPServer:
    """
    Create a local HTTP server using the first available port in a range.

    handler_factory  (Any)  :  HTTP request-handler class or compatible 
                               callable
    preferred_port   (int)  :  first local port to try
    host             (str)  :  host interface on which the server listens
    max_attempts     (int)  :  maximum number of consecutive ports to try

    Return:
    (ReusableTCPServer)     :  configured local server bound to an available 
                               port
    """
    for port in range(preferred_port, preferred_port + max_attempts):
        try:
            # bind the HTTP handler to the current host and candidate port
            return ReusableTCPServer((host, port), handler_factory)
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            print(f"Port {port} is already in use, trying {port + 1}...")

    raise OSError(
        errno.EADDRINUSE,
        f"No free port found from {preferred_port} to {preferred_port+max_attempts-1}",
    )


class FrontendHandler(http.server.SimpleHTTPRequestHandler):
    """
    Serve frontend files and implement the application's backend HTTP API.

    GET requests outside the API are delegated to SimpleHTTPRequestHandler for 
    static-file serving. 
    Application endpoints exchange UTF-8 JSON payloads with the browser frontend.
    """

    def __init__(
        self, *args: Any, directory: Optional[str] = None, **kwargs: Any
    ) -> None:
        """
        Initialize the request handler with the frontend file directory.

        args       (Any)            :  positional arguments supplied by TCP server
        directory  (Optional[str])  :  root directory used to serve frontend files
        kwargs     (Any)            :  keyword arguments supplied by the TCP server
        """
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """
        Write HTTP access-log message to standard output.

        format  (str)  :  request log format supplied by the HTTP handler
        args    (Any)  :  values interpolated into the request log format
        """
        # prefix server request logs so HTTP traffic is easy to identify
        print("[HTTP]", format % args)

    def end_headers(self) -> None:
        """Prevent browsers from retaining replaceable avatar model files."""
        if urlparse(self.path).path.startswith("/models/"):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        """
        Send one dictionary to the frontend as an HTTP JSON response.

        payload  (Dict[str, Any])  :  response data to serialize as JSON
        status   (int)             :  HTTP response status code
        """
        # serialize the backend payload to the UTF-8 response body expected 
        # by the frontend
        body = json.dumps(payload).encode("utf-8")
        # send HTTP status line before response headers and body
        self.send_response(status)
        # tell the frontend that the response body contains UTF-8 JSON
        self.send_header("Content-Type", "application/json; charset=utf-8")
        # provide the exact byte length required by the HTTP client
        self.send_header("Content-Length", str(len(body)))
        # finish the HTTP header section
        self.end_headers()
        # write the serialized backend response to the frontend connection
        self.wfile.write(body)

    def do_GET(self) -> None:
        """
        Handle API and frontend static-file HTTP GET requests.
        """
        # parse frontend request target separately
        parsed = urlparse(self.path)

        # handle the frontend request for the initial/default avatar animation
        if parsed.path == "/api/start":
            try:
                # default animation payload returned during frontend startup
                payload = loader.get_default_payload()
                # default animation to the frontend as a successful JSON response
                self._send_json(payload, status=200)
            except Exception as e:
                # report backend loading failures to frontend as HTTP server error
                self._send_json({"error": str(e)}, status=500)
            return

        # serve all non-API GET requests as frontend files from configured 
        # directory
        return super().do_GET()

    def do_POST(self) -> None:
        """
        Handle animation, text-input, and shutdown HTTP POST requests.
        """
        # parse frontend request target separately from any query parameters
        parsed = urlparse(self.path)

        # Handle the frontend request for the ending animation.
        if parsed.path == "/api/end":
            try:
                # load animation and label its requested frontend camera mode
                payload = loader.load_payload(
                    "Headbutt_mixamo_com_frames.json",
                    animation_name="headbutt",
                    camera_state="end",
                )
                # return the ending animation to the frontend as JSON
                self._send_json(payload, status=200)
            except Exception as e:
                # report animation-loading failures to the frontend
                self._send_json({"error": str(e)}, status=500)
            return

        # handle text submitted by the frontend text input
        if parsed.path == "/api/text":
            try:
                # read the request-body byte count supplied by frontend HTTP client
                content_length = int(self.headers.get("Content-Length", 0))
                # read JSON request body from the frontend connection
                raw_body = self.rfile.read(content_length)
                # decode the UTF-8 request body and deserialize its JSON object
                data = json.loads(raw_body.decode("utf-8"))

                # extract and normalize the text field sent by text_input.js
                user_text = data.get("text", "").strip()
                if not user_text:
                    # reject empty frontend submissions as an HTTP client error
                    self._send_json({"error": "Empty text input"}, status=400)
                    return

                print("[TEXT INPUT]", user_text)

                # llm feedback and processing 
                global latest_llm_answer, conversation_history
                # classify user input on whether needs conversation history
                # to generate answer
                context_type = classify_context_need(user_text)
                print("[CONTEXT TYPE]", context_type)

                if context_type == "SELF_CONTAINED":
                    # no history needed
                    latest_llm_answer = get_response(user_text)

                elif context_type == "RECENT_CONTEXT":
                    # use the latest in-memory stored history
                    latest_llm_answer = get_response(
                        user_text,
                        conversation_history=conversation_history[-10:]
                    )

                elif context_type == "ARCHIVE_CONTEXT":
                    # search in-memory stored history
                    archived_messages = history_store.search_messages(
                        user_text,
                        limit=6
                    )

                    combined_history = (
                        archived_messages + conversation_history[-10:]
                    )

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
                
                print(f"\nAssistant: {latest_llm_answer}")

                # process generated llm answer and print out for visibility (debug)
                debug_data = process_answer_text(latest_llm_answer)
                print_pipeline_debug(debug_data)
                print_json_paths(debug_data)
                traced_tokens = flatten_traced_tokens(debug_data["traced_tokens"])
                
                # build associated motion from the answer processing result
                motion_result = build_threejs_motion_from_tokens(
                    traced_tokens,
                    output_path="tmp/answer_motion.json",
                    interpolation_frames=5,
                    animation_name="answer_motion",
                    source_text=latest_llm_answer,
                )

                # package the generated motion file into the shared frontend 
                # animation format
                animation_payload = loader.load_payload(
                    "tmp/answer_motion.json",
                    animation_name="answer_motion",
                    camera_state="start",
                )

                # chat history management
                # add new conversation to history
                # and remove early conversation if the logged history is too long
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

                # return answer, diagnostics, and playable animation to text_input.js
                self._send_json(
                    {
                        "ok": True,
                        "received_text": user_text,
                        "answer_text": latest_llm_answer,
                        "traced_tokens": traced_tokens,
                        "missing_motion_tokens": motion_result["missing_tokens"],
                        "motion_source": str(animation_payload["source"]),
                        "animation": animation_payload["animation"],
                        "camera": animation_payload["camera"],
                        "frames": animation_payload["frames"],
                    },
                    status=200,
                )

            except Exception as e:
                # return unexpected text-pipeline failures to the frontend
                self._send_json({"error": str(e)}, status=500)
            return

        # handle browser's request to stop the local application server
        if parsed.path == "/__shutdown__":
            # shutdown request before stopping its server connection
            self.send_response(200)
            # finish header-only portion of the plain-text response
            self.end_headers()
            # confirm shutdown to the requesting frontend code
            self.wfile.write(b"shutting down")
            # shut down outside the active request thread to avoid a 
            # server deadlock
            threading.Thread(target=shutdown_app, daemon=True).start()
            return

        # reject POST requests for routes not implemented by backend
        self.send_response(404)
        # finish the response headers for the unknown route
        self.end_headers()
        # return small plain-text error body to the HTTP client
        self.wfile.write(b"not found")


def cleanup_app_resources() -> None:
    """
    Delete temporary application resources created during the current run.
    """
    try:
        history_store.delete_file()
        print("Deleted chat history JSONL.")
    except Exception as e:
        print("Failed to delete chat history JSONL:", e)

def handle_exit_signal(signum: int, frame: Optional[FrameType]) -> None:
    """
    Clean up resources and stop the server after an operating-system signal.

    signum  (int)                  :  received operating-system signal number
    frame   (Optional[FrameType])  :  active stack frame supplied by the signal 
                                      handler
    """
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

def shutdown_app() -> None:
    """
    Clean up resources and stop the server after a frontend shutdown request.
    """
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


def main() -> None:
    """
    Initialize application resources and run the local frontend/backend server.

    The server is started on a background thread, after which the frontend page 
    is opened in the default browser. 
    The main thread remains active until shutdown or interruption.
    """
    global httpd

    # locate the directory served by the HTTP handler and its frontend entry page
    base_dir = Path(__file__).resolve().parent
    index_file = base_dir / "templates" / "index.html"

    if not index_file.exists():
        print(f"Error: {index_file} not found.")
        return

    class HandlerFactory(FrontendHandler):
        """
        Bind each HTTP request handler to the application frontend directory.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """
            Initialize one frontend request handler.

            args    (Any)  :  positional arguments supplied by the TCP server
            kwargs  (Any)  :  keyword arguments supplied by the TCP server
            """
            # Make base_dir the document root for frontend static-file requests
            super().__init__(*args, directory=str(base_dir), **kwargs)

    # bind the backend HTTP server to an available local port
    with create_server(HandlerFactory) as server:
        # retain server globally so signal and frontend shutdown handlers can 
        # stop it
        httpd = server
        # read actual selected port in case preferred port was unavailable
        selected_port = server.server_address[1]
        # build the local frontend URL opened in the user's browser
        url = f"http://127.0.0.1:{selected_port}/templates/index.html"

        load_model()
        load_text_analyzer()
        load_vocab_tree(
            "./vocabs/all_vocabs.json",
            "./vocabs/all_vocabs_metadata.jsonl"
        )

        atexit.register(cleanup_app_resources)
        signal.signal(signal.SIGINT, handle_exit_signal)   # Ctrl+C
        signal.signal(signal.SIGTERM, handle_exit_signal)

        print(f"Serving frontend from: {base_dir}")
        print(f"Open in browser: {url}")

        # process HTTP requests on a daemon thread while the main thread 
        # monitors shutdown
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        # allow the local HTTP server to begin listening before launching 
        # the frontend
        time.sleep(0.5)
        # open the frontend entry page using the system's default browser
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
