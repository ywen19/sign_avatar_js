import json
from typing import List


END = "__end__"


class VocabTree:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VocabTree, cls).__new__(cls)
            cls._instance.root = {}
            cls._instance.vocab_list = []
        return cls._instance

    def load_vocab_json(self, json_path: str) -> None:
        print(f"Loading vocab json from: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            self.vocab_list = json.load(f)

        print(f"Loaded {len(self.vocab_list)} vocab items.")

    def build_tree(self) -> None:
        if not self.vocab_list:
            print("No vocab loaded. Tree build skipped.")
            return

        print("Building vocab tree...")
        self.root = {}

        for vocab_key in self.vocab_list:
            tokens = [token for token in vocab_key.lower().split("_") if token]
            if not tokens:
                continue

            node = self.root

            for token in tokens:
                if token not in node:
                    node[token] = {}
                node = node[token]

            node[END] = vocab_key

        print("Vocab tree built.")

    def _print_subtree(self, node, indent=0) -> None:
        for key, child in node.items():
            if key == END:
                print("  " * indent + f"[END] {child}")
            else:
                print("  " * indent + key)
                self._print_subtree(child, indent + 1)

    def print_subtree(self, prefix=None) -> None:
        if not self.root:
            print("Tree is empty.")
            return

        if not prefix:
            print("Printing full tree...")
            self._print_subtree(self.root)
            return

        tokens = [token for token in prefix.lower().split() if token]
        node = self.root

        for token in tokens:
            if token not in node:
                print(f"Prefix not found: {prefix}")
                return
            node = node[token]

        print(f"Printing subtree for prefix: {prefix}")
        self._print_subtree(node)

    def match_from(self, tokens: List[str], start_idx: int, entity_set=None):
        if not self.root:
            return None

        if entity_set is None:
            entity_set = set()

        node = self.root
        j = start_idx
        last_match = None

        while j < len(tokens):
            current = tokens[j]

            # entity chunks are hard boundaries unless they are the first chunk
            if j != start_idx and current in entity_set:
                break

            current_norm = current.lower()
            if current_norm not in node:
                break

            node = node[current_norm]
            j += 1

            if END in node:
                last_match = {
                    "vocab_key": node[END],
                    "end_idx": j,
                }

        return last_match

    def trace(self, tokens: List[str], entities: List[str]) -> List[str]:
        results = []
        entity_set = set(entities)
        i = 0

        while i < len(tokens):
            current = tokens[i]

            if current in entity_set:
                results.append(current)
                i += 1
                continue

            match = self.match_from(tokens, i, entity_set=entity_set)

            if match is not None:
                results.append(match["vocab_key"])
                i = match["end_idx"]
            else:
                i += 1

        return results


vocab_tree = VocabTree()


def load_vocab_tree(json_path: str) -> None:
    vocab_tree.load_vocab_json(json_path)
    vocab_tree.build_tree()


def print_vocab_subtree(prefix=None) -> None:
    vocab_tree.print_subtree(prefix)


def trace_tokens(tokens: List[str], entities: List[str]) -> List[str]:
    return vocab_tree.trace(tokens, entities)

def load_vocab_json(json_path: str) -> None:
    return vocab_tree.load_vocab_json(json_path)


if __name__ == "__main__":
    vocab_tree = VocabTree()
    vocab_tree.load_vocab_json("../vocabs/all_vocabs.json")
    vocab_tree.build_tree()

    print("\n================ SUBTREE TEST ================\n")
    vocab_tree.print_subtree("british")

    print("\n================ MATCH TESTS ================\n")

    test_cases = [
        {
            "name": "single word match",
            "tokens": ["bournemouth"],
            "entities": [],
        },
        {
            "name": "multi word match",
            "tokens": ["a", "lot"],
            "entities": [],
        },
        {
            "name": "skip unknown token",
            "tokens": ["xyzabc", "bournemouth"],
            "entities": [],
        },
        {
            "name": "preserve entity",
            "tokens": ["bournemouth", "has", "museum", "called", "bournemouth art museum"],
            "entities": ["bournemouth art museum"],
        },
        {
            "name": "entity and vocab mixed",
            "tokens": ["bournemouth", "has", "a", "museum", "called", "bournemouth art museum"],
            "entities": ["bournemouth art museum"],
        },
        {
            "name": "longest phrase match",
            "tokens": ["a", "long", "time", "ago"],
            "entities": [],
        },
        {
            "name": "stop at entity boundary",
            "tokens": ["a", "lot", "bournemouth art museum"],
            "entities": ["bournemouth art museum"],
        },
        {
            "name": "checking if contain sentences",
            "tokens": ["what", "would", "you", "like", "to", "eat", "today"],
            "entities": [],
        },
    ]

    for case in test_cases:
        print(f"[TEST] {case['name']}")
        print("tokens   :", case["tokens"])
        print("entities :", case["entities"])

        traced = vocab_tree.trace(case["tokens"], case["entities"])
        print("traced   :", traced)
        print()