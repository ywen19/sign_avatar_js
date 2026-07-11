"""
Provides a lookup service for identifying the linguistic identity of text chunks,
such as time expressions, negations, WH-question words, or normal chunks.

The identity metadata is loaded from a JSONL file, where each line contains information about
a vocabulary chunk and its corresponding identity label. These identity labels are mainly used
to support sign language motion generation, where certain chunk types may require different
word ordering or special handling, such as reordering. 

This singleton service ensures that the identity metadata is loaded only once and shared
within the same process. It prevents repeated file loading, reduces unnecessary memory usage,
and provides a central place to load identity metadata and query chunk-level identity tags.
"""


import json
import os

from typing import List, Dict, Tuple


class IdentityLookup:
    _instance = None  # for singleton service

    # integer tags used to mark whether a chunk needs special reordering
    REORDER_NORMAL = 0
    REORDER_TIME = 1
    REORDER_NEGATION = 2
    REORDER_WH = 3

    def __new__(cls, metadata_path="all_vocabs_metadata.jsonl"):
        """
        For singleton service.
        Only create instance if no instance of this class exists;
        if exists, return the living instance.
        """
        if cls._instance is None:
            cls._instance = super(IdentityLookup, cls).__new__(cls)
            cls._instance.loaded = False
        return cls._instance

    def load(self, metadata_path="all_vocabs_metadata.jsonl") -> None:
        """
        Initialize the identity lookup service from a metadata JSONL file.
        This method sets up the metadata path and lookup dictionary, then loads the
        metadata only once for reuse across the process.

        metadata_path  (str)  : path to the metadata file containing chunk identity labels
        """
        if self.loaded:
            print("Identity lookup already loaded.")
            return

        self.metadata_path = metadata_path
        self.chunk_to_identity = {}
        self._load_metadata()
        self.loaded = True
        print("Identity lookup loaded.")

    def _load_metadata(self) -> None:
        """
        Actual helper function to read metadata JSONL file and populate the chunk-to-identity 
        lookup table.
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Identity metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} in {self.metadata_path}"
                    ) from e

                chunk = obj.get("chunk")
                identity = obj.get("identity", "normal")

                if not isinstance(chunk, str) or not chunk:
                    continue

                if not isinstance(identity, str):
                    identity = "normal"

                self.chunk_to_identity[chunk] = identity

    def get_identity(self, chunk: str) -> str:
        """
        Get the identity label for a given text chunk.

        chunk  (str)  :  text chunk to look up

        Return:
        identity  (str)  :  Identity label of the chunk, such as "normal", "time", 
                            "negation", or "wh". Returns "normal" if the chunk is 
                            not found in the lookup table
        """
        if not self.loaded:
            raise RuntimeError("Identity lookup not loaded. Call load_identity_lookup() first.")
        return self.chunk_to_identity.get(chunk, "normal")

    def get_reorder_tag(self, chunk: str) -> int:
        """
        Looks up the chunk identity, then converts it into a numeric reorder tag used by the 
        sign language motion generation pipeline.

        chunk  (str)  :  text chunk to look up

        Return:
        (int)  :  int tag based on the chunk identity tag
        """
        identity = self.get_identity(chunk)

        if identity == "time":
            return self.REORDER_TIME
        if identity == "negation":
            return self.REORDER_NEGATION
        if identity == "wh":
            return self.REORDER_WH
        return self.REORDER_NORMAL

    def is_time(self, chunk: str) -> bool:
        return self.get_identity(chunk) == "time"

    def is_negation(self, chunk: str) -> bool:
        return self.get_identity(chunk) == "negation"

    def is_wh(self, chunk: str) -> bool:
        return self.get_identity(chunk) == "wh"


# singleton instance initialization
identity_lookup = IdentityLookup()


# module-level wrappers functions that expose simple load, identity lookup, reorder tag,
# and identity checking operations without requiring callers to interact with
# the IdentityLookup instance directly.
def load_identity_lookup(metadata_path: str) -> None:
    identity_lookup.load(metadata_path)


def get_identity(chunk: str) -> str:
    return identity_lookup.get_identity(chunk)


def get_reorder_tag(chunk: str) -> int:
    return identity_lookup.get_reorder_tag(chunk)


def is_time(chunk: str) -> bool:
    return identity_lookup.is_time(chunk)


def is_negation(chunk: str) -> bool:
    return identity_lookup.is_negation(chunk)


def is_wh(chunk: str) -> bool:
    return identity_lookup.is_wh(chunk)