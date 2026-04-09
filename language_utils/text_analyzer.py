import importlib.util
import re
import subprocess
import sys
from typing import List

import contractions
import spacy


class TextAnalyzer:
    _instance = None
    MODEL_NAME = "en_core_web_sm"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextAnalyzer, cls).__new__(cls)
            cls._instance.nlp = None
            cls._instance.keep_labels = {"PERSON", "GPE", "LOC", "FAC", "ORG"}
        return cls._instance

    def _ensure_model_installed(self) -> None:
        if importlib.util.find_spec(self.MODEL_NAME) is None:
            print(f"{self.MODEL_NAME} not found. Downloading...")
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", self.MODEL_NAME]
            )
        else:
            print(f"{self.MODEL_NAME} already installed.")

    def load_model(self) -> None:
        if self.nlp is not None:
            print("Text analyzer already loaded.")
            return

        print("Loading text analyzer...")
        self._ensure_model_installed()
        self.nlp = spacy.load(self.MODEL_NAME)
        print("Text analyzer loaded.")

    def break_into_sentences(self, text: str) -> List[str]:
        if self.nlp is None:
            raise RuntimeError("Text analyzer not loaded. Call load_model() first.")

        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

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
                L = len(ent_tokens)
                if sentence_tokens[i:i + L] == ent_tokens:
                    matched = ent_tokens
                    break

            if matched:
                results.append(entity_lookup[tuple(matched)])
                i += len(matched)
            else:
                results.append(sentence_tokens[i])
                i += 1

        return results

    def detect_entities(self, text: str) -> List[str]:
        keep_labels = {"PERSON", "GPE", "LOC", "FAC", "ORG"}
        doc = self.nlp(text)
        results = []

        for ent in doc.ents:
            if ent.label_ in keep_labels:
                results.append(self.normalize_sentence_for_match(ent.text))

        return results


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