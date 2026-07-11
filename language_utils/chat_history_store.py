"""
Chat history logging and retrieving mechanism for multi-turn conversation.
The history will be overwritten everytime when starting the main program.

The current prototype uses a lightweight chat-history mechanism to support multi-turn conversation. 
Previous user and assistant messages are stored and added to the prompt when the system determines 
that contextual information is required.

For advance future usage, it is recommended to use conversation summarisation, semantic memory 
retrieval, structured user memory, or other agent-based memory technical stacks.
"""


import json
import time
from pathlib import Path
from typing import List, Dict


class ChatHistoryStore:
    def __init__(self, file_path: str = "chat_history.jsonl") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.touch(exist_ok=True)

    def append_message(
        self, role: str, content: str, timestamp: float = None) -> Dict:
        """
        Append one message to the chat history file. 
        For a message, it contains:
            role      (str)   : "user" or "assistant" (llm model)
            content   (str)   : either user input or llm answer
            timestamp (float) : timestamp when the conversation info is documented
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp if timestamp is not None else time.time(),
        }

        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")
        return message

    def append_messages(self, messages: List[Dict]) -> None:
        """
        Append input messages to the chat history file. 
        For each message, it contains:
            role      (str)   : "user" or "assistant" (llm model)
            content   (str)   : either user input or llm answer
            timestamp (float) : timestamp when the conversation info is documented
        """
        if not messages:
            return

        with self.file_path.open("a", encoding="utf-8") as f:
            for message in messages:
                record = {
                    "role": message["role"],
                    "content": message["content"],
                    "timestamp": message.get("timestamp", time.time()),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_all_messages(self) -> List[Dict]:
        """
        Readin all stored chat history info from file.
        """
        messages = []

        with self.file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return messages

    def load_recent_messages(self, limit: int = 20) -> List[Dict]:
        """
        From all readin stored chat history, we keep only last n dialogue info.
        limit (int) : last n dialogue info to be used for contextual memory.
        """
        messages = self.load_all_messages()
        if limit <= 0:
            return []
        return messages[-limit:]

    def search_messages(self, query: str, limit: int = 6) -> List[Dict]:
        """
        Search contextual information from last n dialogue info.
        We query contextual chat by the new user inout, to retrieve relative info.
        query  (str)  : user input
        limit  (int)  : last n dialogue info to be used for contextual memory.
        """
        query = (query or "").strip().lower()
        if not query:
            return []

        query_terms = [term for term in query.split() if len(term) > 2]
        if not query_terms:
            return []

        scored_messages = []

        with self.file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = str(message.get("content", "")).lower()
                if not content:
                    continue

                score = sum(1 for term in query_terms if term in content)
                if score > 0:
                    scored_messages.append((score, message))

        scored_messages.sort(
            key=lambda item: (
                item[0],
                item[1].get("timestamp", 0),
            ),
            reverse=True,
        )

        results = []
        seen = set()

        for _, message in scored_messages:
            key = (
                message.get("role", ""),
                message.get("content", ""),
                message.get("timestamp", 0),
            )
            if key in seen:
                continue
            seen.add(key)
            results.append(message)

            if len(results) >= limit:
                break

        results.sort(key=lambda msg: msg.get("timestamp", 0))
        return results

    def delete_file(self) -> None:
        """
        Delete the chat history file when the program quits.
        For now, we only support per-process one-time conversation history log and query.
        """
        if self.file_path.exists():
            self.file_path.unlink()