import json
import time
from pathlib import Path
from typing import List, Dict


class ChatHistoryStore:
    def __init__(self, file_path: str = "chat_history.jsonl"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.touch(exist_ok=True)

    def append_message(self, role: str, content: str, timestamp: float = None) -> Dict:
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp if timestamp is not None else time.time(),
        }

        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

        return message

    def append_messages(self, messages: List[Dict]) -> None:
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
        messages = self.load_all_messages()
        if limit <= 0:
            return []
        return messages[-limit:]

    def search_messages(self, query: str, limit: int = 6) -> List[Dict]:
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
        if self.file_path.exists():
            self.file_path.unlink()