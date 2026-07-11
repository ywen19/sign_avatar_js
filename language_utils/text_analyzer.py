"""
Defines the text processing pipeline for SmolLM responses before they are 
converted into sign language motions.

It handles sentence splitting, text normalization, number normalization,
tokenization, and GLiNER-based entity detection. Named entities such as people,
places, and addresses are preserved as single chunks for downstream matching.

TextAnalyzer is implemented as a singleton so the analyzer is loaded once and 
reused across the same process.
"""


import re
from typing import List

from language_utils.gliner_service import load_gliner_model, predict_entities
from language_utils.number_normalization import normalize_numbers_in_sentence


# entity labels passed to GLiNER for detecting chunks in SmolLM responses
GLINER_LABELS: List[str] = [
    "person",
    "city",
    "region",
    "country",
    "restaurant",
    "gallery",
    "museum",
    "address",
]
# minimum confidence score for accepting GLiNER entity predictions
GLINER_THRESHOLD: float = 0.65


class TextAnalyzer:
    _instance = None  # for singleton service

    def __new__(cls):
        """
        For singleton service.
        Only create instance if no instance of this class exists;
        if exists, return the living instance.
        """
        if cls._instance is None:
            cls._instance = super(TextAnalyzer, cls).__new__(cls)
            cls._instance.model_loaded = False
        return cls._instance

    def load_model(self) -> None:
        """
        Load the GLiNER-based text analyzer.

        Initializes the entity detection model used by the text analysis
        pipeline. If the analyzer is already loaded, it reuses the 
        existing model.
        """
        if self.model_loaded:
            print("Text analyzer already loaded.")
            return

        print("Loading text analyzer...")
        load_gliner_model()
        self.model_loaded = True
        print("Text analyzer loaded.")

    def break_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        """
        if not text:
            return []

        text = text.strip()
        if not text:
            return []

        parts = re.split(r"(?<=[.!?])\s+", text)
        return [part.strip() for part in parts if part.strip()]

    def normalize_sentence_for_match(self, text: str) -> str:
        """
        Normalize text into a clean format for matching.

        Including: 
        lowercases the text, normalizes common symbols, removes most punctuation, 
        and collapses extra spaces. It keeps letters, digits, spaces,
        and hyphens so the result can be used for token matching.
        """
        text = text.lower()
        text = text.replace("&", " and ")
        text = text.replace("'", "")

        # remove punctuation except letters, digits, spaces, and hyphens
        text = re.sub(r"[^a-z0-9\s-]", " ", text)

        # collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def canonicalize_for_entity_match(self, text: str) -> str:
        """
        Currently this uses the same normalization as general sentence matching,
        but it is kept as a separate wrapper so entity matching rules can be changed
        independently later.
        """
        return self.normalize_sentence_for_match(text)

    def tokenize_plain(self, text: str) -> List[str]:
        """
        Normalize text and split it into plain tokens.

        This is used as the base tokenization step before entity-aware token merging.
        """
        text = self.canonicalize_for_entity_match(text)
        return text.split() if text else []

    def tokenize_with_entities(self, text: str, entities: List[str]) -> List[str]:
        """
        Tokenize text while preserving detected entities as single chunks.

        This method first tokenizes the sentence normally, then checks whether any
        sequence of tokens matches a detected entity. If a match is found, the full
        entity phrase is kept as one token instead of being split into separate words.

        text      (str)        :  input sentence to tokenize
        entities  (List[str])  :  detected entity strings to preserve as single chunks

        Return:
        results   (List[str])  :  token list with matching entities kept as single chunks
        """
        if not text:
            return []

        sentence_tokens = self.tokenize_plain(text)
        if not entities:
            return sentence_tokens

        entity_index = {}
        entity_lookup = {}

        # entity split preparation for fast lookup
        # entity_index:  helps quickly find possible entity matches 
        #                while scanning sentence tokens
        # entity_lookup: maps matched token sequences back to the 
        #                entity string to preserve it
        # ent example: "British Museum"
        # entity_index = {"british": [["british", "museum"]]}
        # entity_lookup = {("british", "museum"): "British Museum"}
        for ent in entities:
            # normalize and split the entity text into tokens
            ent_tokens = self.tokenize_plain(ent)
            if not ent_tokens:
                continue
            # use the first token as the lookup key for faster matching
            first = ent_tokens[0]
            # store this entity token sequence under its first token
            entity_index.setdefault(first, []).append(ent_tokens)
            # keep a mapping from the normalized token sequence back to 
            # the original entity string
            entity_lookup[tuple(ent_tokens)] = ent

        # sort entity candidates so longer entities are checked first
        for first in entity_index:
            # for the same starting token, prefer longer matches first
            entity_index[first].sort(key=len, reverse=True)

        results = []  # store the final tokenization result
        i = 0  # current position while scanning sentence_tokens

        # scan through the sentence tokens from left to right
        while i < len(sentence_tokens):
            # get the current token
            current = sentence_tokens[i]
            # store the matched entity token sequence, if any
            matched = None

            # check possible entity sequences that start with the current token
            for ent_tokens in entity_index.get(current, []):
                length = len(ent_tokens)
                # compare the sentence slice with the entity token sequence
                # slice starts from the current token to the length of the entity
                if sentence_tokens[i:i + length] == ent_tokens:
                    # save the matched entity sequence
                    matched = ent_tokens
                    break  # stop checking after the first match

            if matched:
                # if an entity sequence was matched, keep it as one chunk
                # skip over all tokens that belonged to this matched entity
                results.append(entity_lookup[tuple(matched)])
                i += len(matched)
            else:
                # if no entity matched, keep the current token as normal
                # and add the current plain token to the result
                results.append(sentence_tokens[i])
                # move to the next token
                i += 1

        return results

    def _strip_address_prefix(self, text: str) -> str:
        """
        Remove generic address prefixes from detected address entities.

        GLiNER may include words such as "room", "gate", "flat", or "apartment"
        as part of an address entity. This helper removes those prefixes so the
        remaining address/code can be matched more directly downstream.

        text  (str)  :  detected address entity text

        Returns:
        (str)  :  address text with known prefixes removed.
        """
        text = text.strip()

        prefixes = [
            "room ", "gate ", "flat ", "apartment ",
            "room", "gate", "flat", "apartment"
        ]
        lowered = text.lower()

        for prefix in prefixes:
            if lowered.startswith(prefix):
                return text[len(prefix):].strip()

        return text

    def detect_entities(self, text: str) -> List[str]:
        """
        Detect and clean named entities from input text using GLiNER service 
        using the configured entity labels.

        It then cleans the detected entities by removing empty results, stripping
        generic address prefixes, normalizing entity text for matching, and removing
        duplicate entities.

        Since entities may often use capital letters, we detect the entities first
        from the smollm decoded answer, and then apply normalization over the 
        answer as well as the detected entities.

        text  (str)  :  input text to analyze

        Return:
        cleaned  (List[str])  :  normalized entity strings detected from the text
        """
        if not self.model_loaded:
            raise RuntimeError("Text analyzer not loaded. Call load_model() first.")

        raw_entities = predict_entities(
            text,
            GLINER_LABELS,
            threshold=GLINER_THRESHOLD,
        )

        cleaned = []
        seen = set()  # avoid duplicants on detected entities

        for ent in raw_entities:
            chunk_text = ent.get("text", "").strip()
            chunk_label = ent.get("label", "").strip().lower()

            if not chunk_text:
                continue

            if chunk_label == "address":
                chunk_text = self._strip_address_prefix(chunk_text)
                if not chunk_text:
                    continue

            normalized = self.canonicalize_for_entity_match(chunk_text)
            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            cleaned.append(normalized)

        return cleaned

    def normalize_for_runtime_match(self, text: str) -> str:
        """
        Normalize text for runtime vocabulary or motion matching.

        First converts numbers into word form, then applies the standard
        sentence normalization used for matching.

        text  (str)  :  input text

        Return:
        text  (str)  :  normalized text ready for runtime matching
        """
        text = normalize_numbers_in_sentence(text)
        text = self.normalize_sentence_for_match(text)
        return text


# singleton instance initialization
text_analyzer = TextAnalyzer()


# module-level wrappers provide a simple public API without exposing 
# the service instance
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

def normalize_for_runtime_match(text: str) -> str:
    return text_analyzer.normalize_for_runtime_match(text)