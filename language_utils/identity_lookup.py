import json
import os


class IdentityLookup:
    _instance = None

    REORDER_NORMAL = 0
    REORDER_TIME = 1
    REORDER_NEGATION = 2
    REORDER_WH = 3

    def __new__(cls, metadata_path="all_vocabs_metadata.jsonl"):
        if cls._instance is None:
            cls._instance = super(IdentityLookup, cls).__new__(cls)
            cls._instance.loaded = False
        return cls._instance

    def load(self, metadata_path="all_vocabs_metadata.jsonl"):
        if self.loaded:
            print("Identity lookup already loaded.")
            return

        self.metadata_path = metadata_path
        self.chunk_to_identity = {}
        self._load_metadata()
        self.loaded = True
        print("Identity lookup loaded.")

    def _load_metadata(self):
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

    def get_identity(self, chunk):
        if not self.loaded:
            raise RuntimeError("Identity lookup not loaded. Call load_identity_lookup() first.")
        return self.chunk_to_identity.get(chunk, "normal")

    def get_reorder_tag(self, chunk):
        identity = self.get_identity(chunk)

        if identity == "time":
            return self.REORDER_TIME
        if identity == "negation":
            return self.REORDER_NEGATION
        if identity == "wh":
            return self.REORDER_WH
        return self.REORDER_NORMAL

    def is_time(self, chunk):
        return self.get_identity(chunk) == "time"

    def is_negation(self, chunk):
        return self.get_identity(chunk) == "negation"

    def is_wh(self, chunk):
        return self.get_identity(chunk) == "wh"


identity_lookup = IdentityLookup()


def load_identity_lookup(metadata_path):
    identity_lookup.load(metadata_path)


def get_identity(chunk):
    return identity_lookup.get_identity(chunk)


def get_reorder_tag(chunk):
    return identity_lookup.get_reorder_tag(chunk)


def is_time(chunk):
    return identity_lookup.is_time(chunk)


def is_negation(chunk):
    return identity_lookup.is_negation(chunk)


def is_wh(chunk):
    return identity_lookup.is_wh(chunk)