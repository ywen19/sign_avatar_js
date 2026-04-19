import re
from typing import List

from language_utils.gliner_service import load_gliner_model, predict_entities


GLINER_LABELS = [
    "person",
    "city",
    "region",
    "country",
    "restaurant",
    "gallery",
    "museum",
    "address",
]

GLINER_THRESHOLD = 0.65


class TextAnalyzer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextAnalyzer, cls).__new__(cls)
            cls._instance.model_loaded = False
        return cls._instance

    def load_model(self) -> None:
        if self.model_loaded:
            print("Text analyzer already loaded.")
            return

        print("Loading text analyzer...")
        load_gliner_model()
        self.model_loaded = True
        print("Text analyzer loaded.")

    def break_into_sentences(self, text: str) -> List[str]:
        if not text:
            return []

        text = text.strip()
        if not text:
            return []

        parts = re.split(r"(?<=[.!?])\s+", text)
        return [part.strip() for part in parts if part.strip()]

    def normalize_sentence_for_match(self, text: str) -> str:
        text = text.lower()
        text = text.replace("&", " and ")
        text = text.replace("'", "")

        # remove punctuation except letters, digits, spaces, and hyphens
        text = re.sub(r"[^a-z0-9\s-]", " ", text)

        # collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize_plain(self, text: str) -> List[str]:
        text = self.normalize_sentence_for_match(text)
        return text.split() if text else []

    def tokenize_with_entities(self, text: str, entities: List[str]) -> List[str]:
        if not text:
            return []

        sentence_tokens = self.tokenize_plain(text)
        if not entities:
            return sentence_tokens

        entity_index = {}
        entity_lookup = {}

        for ent in entities:
            ent_tokens = self.tokenize_plain(ent)
            if not ent_tokens:
                continue

            first = ent_tokens[0]
            entity_index.setdefault(first, []).append(ent_tokens)
            entity_lookup[tuple(ent_tokens)] = ent

        for first in entity_index:
            entity_index[first].sort(key=len, reverse=True)

        results = []
        i = 0

        while i < len(sentence_tokens):
            current = sentence_tokens[i]
            matched = None

            for ent_tokens in entity_index.get(current, []):
                length = len(ent_tokens)
                if sentence_tokens[i:i + length] == ent_tokens:
                    matched = ent_tokens
                    break

            if matched:
                results.append(entity_lookup[tuple(matched)])
                i += len(matched)
            else:
                results.append(sentence_tokens[i])
                i += 1

        return results

    def _strip_address_prefix(self, text: str) -> str:
        text = text.strip()

        prefixes = ["room ", "gate ", "flat ", "apartment "]
        lowered = text.lower()

        for prefix in prefixes:
            if lowered.startswith(prefix):
                return text[len(prefix):].strip()

        return text

    def detect_entities(self, text: str) -> List[str]:
        if not self.model_loaded:
            raise RuntimeError("Text analyzer not loaded. Call load_model() first.")

        raw_entities = predict_entities(
            text,
            GLINER_LABELS,
            threshold=GLINER_THRESHOLD,
        )

        cleaned = []
        seen = set()

        for ent in raw_entities:
            chunk_text = ent.get("text", "").strip()
            chunk_label = ent.get("label", "").strip().lower()

            if not chunk_text:
                continue

            if chunk_label == "address":
                chunk_text = self._strip_address_prefix(chunk_text)
                if not chunk_text:
                    continue

            normalized = self.normalize_sentence_for_match(chunk_text)
            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            cleaned.append(normalized)

        return cleaned


text_analyzer = TextAnalyzer()


def load_text_analyzer() -> None:
    text_analyzer.load_model()


def break_into_sentences(text: str) -> List[str]:
    return text_analyzer.break_into_sentences(text)


def tokenize_plain(text: str) -> List[str]:
    return text_analyzer.tokenize_plain(text)


def tokenize_with_entities(text: str, entities: List[str]) -> List[str]:
    return text_analyzer.tokenize_with_entities(text, entities)


def normalize_sentence_for_match(text: str) -> str:
    return text_analyzer.normalize_sentence_for_match(text)


def detect_entities(text: str) -> List[str]:
    return text_analyzer.detect_entities(text)