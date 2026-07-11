"""
This module builds a Trie-based longest-match lookup from the motion vocabulary.

Each vocab phrase is stored token by token in a prefix tree, allowing the matcher
to find the longest valid vocab chunk starting from each token position.
"""


import json
from typing import List, Tuple, Set

from language_utils.number_normalization import normalize_numbers_in_sentence
from language_utils.identity_lookup import (
    load_identity_lookup,
    get_reorder_tag
)


# trie terminal marker. When this key exists in a node, the path up to 
# this node represents a complete vocab item.
END: str = "__end__"


class VocabTree:
    _instance = None  # for singleton service

    def __new__(cls):
        """
        For singleton service.
        Only create instance if no instance of this class exists;
        if exists, return the living instance.
        """
        if cls._instance is None:
            cls._instance = super(VocabTree, cls).__new__(cls)
            cls._instance.root = {}
            cls._instance.vocab_list = []
        return cls._instance

    def load_vocab_json(self, json_path: str) -> None:
        """
        Load the motion vocabulary list from a JSON file.

        json_path (str): Path to the vocab JSON file.
        """
        print(f"Loading vocab json from: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            self.vocab_list = json.load(f)

        print(f"Loaded {len(self.vocab_list)} vocab items.")

    def build_tree(self) -> None:
        """
        Build a Trie-based vocab tree from the loaded vocab list.

        Each vocab key is split by underscores into tokens. The 
        tokens are inserted into the tree one by one, and the 
        END marker is used to mark a complete vocab phrase.

        Example of trie tree structure:
        {
            "a": {
                "long": {
                    "time": {
                        "ago": {
                            "__end__": "a_long_time_ago"
                        }
                    }
                },
                "lot": {
                    "__end__": "a_lot"
                }
            }
        }
        """
        if not self.vocab_list:
            print("No vocab loaded. Tree build skipped.")
            return

        print("Building vocab tree...")
        self.root = {}  # reset the root node before building

        # insert each vocab item into the Trie tree
        for vocab_key in self.vocab_list:
            # split vocab keys like "a_long_time_ago" into 
            # ["a", "long", "time", "ago"]
            tokens = [token for token in vocab_key.lower().split("_") if token]
            if not tokens:
                continue

            # start inserting from the root node
            # node here is not a copy of self.root
            # but pointing to the same address
            # so changing node is to change self.root
            node = self.root
            # insert each token as one level in the Trie
            for token in tokens:
                if token not in node:
                    node[token] = {}
                node = node[token]
            # mark this node as the end of a complete vocab phrase
            node[END] = vocab_key

        print("Vocab tree built.")

    def _print_subtree(self, node, indent=0) -> None:
        """
        Recursively print a subtree from the given Trie node.

        This helper is used for debugging and inspecting the vocab tree 
        structure. Each nested level is printed with extra indentation.
        """
        for key, child in node.items():
            if key == END:
                print("  " * indent + f"[END] {child}")
            else:
                print("  " * indent + key)
                self._print_subtree(child, indent + 1)

    def print_subtree(self, prefix=None) -> None:
        """
        Print the full Trie or a subtree under a given prefix.

        If no prefix is given, the whole vocab tree is printed. 
        If a prefix is given, this method first walks through the Trie to 
        find that prefix, then prints only the subtree under it.
        """
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
        """
        Find the longest vocab match starting from a given token index.

        The function walks through the Trie from tokens[start_idx] and 
        keeps track of the latest complete vocab phrase it has seen. 
        If a longer valid phrase is found, it replaces the previous match.

        Entity chunks are treated as hard boundaries unless the entity is 
        the first chunk being matched. This prevents normal vocab matching 
        from crossing into a detected entity phrase.

        tokens     (List[str])  :  tokenized input sentence
        start_idx  (int)        :  index to start matching from
        entity_set              :  optional set of detected entity chunks

        Return:
        dict | None: The longest match found, containing:
            "vocab_key": matched vocab key
            "end_idx": index after the matched span
            None if no vocab match is found.
        """
        if not self.root:
            return None

        # use an empty entity set if no entities are provided
        if entity_set is None:
            entity_set = set()

        # start matching from the root of the Trie
        node = self.root
        j = start_idx  # scans forward through the input tokens from j
        # store the latest complete vocab match found so far
        last_match = None

        # keep scanning while there are tokens left
        while j < len(tokens):
            current = tokens[j]

            # stop if we hit an entity boundary after the first token
            # this prevents matching across detected entity chunks
            if j != start_idx and current in entity_set:
                break

            current_norm = current.lower()

            # stop if the current token does not continue any vocab path
            if current_norm not in node:
                break

            # move down one level in the Trie
            node = node[current_norm]
            j += 1

            # if this Trie node marks a complete vocab phrase, 
            # save it as a match
            if END in node:
                last_match = {
                    "vocab_key": node[END],
                    "end_idx": j,
                }

        return last_match

    def _lookup_exact_phrase(self, text: str):
        """
        Look up whether the whole text exactly matches a vocab phrase.

        This helper splits the input text into tokens and walks through 
        the Trie.
        It only returns a vocab key if the full token sequence reaches 
        an END marker. Partial matches are not accepted.

        Mainly used for entity chunks.

        text (str): Text phrase to look up.

        Return:
        str | None: The matched vocab key if the full phrase exists in 
        the Trie, otherwise None.
        """
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            return None

        # start searching from the root of the Trie
        node = self.root

        # walk through the Trie token by token
        for token in tokens:
            if token not in node:
                # when prase not in the vocab
                return None
            # move down to the next Trie node
            node = node[token]

        if END in node:
            return node[END]
        return None

    def _split_mixed_entity_chunk(self, text: str) -> List[str]:
        """
        Split a mixed entity chunk into separate letter and digit parts.

        Spaces, hyphens, and other punctuation are treated as boundaries. 
        Consecutive letters are grouped together, and consecutive digits 
        are grouped together.

        For example:
            "A12B"        ->  ["A", "12", "B"]
            "Room-204A"   ->  ["Room", "204", "A"]
            "AB12 CD34"   ->  ["AB", "12", "CD", "34"]

        text (str)   :  Entity chunk to split.

        Return:
        (List[str])  :  Split parts, such as letters and number groups.
        """
        parts = []  # final split parts
        current = []  # characters for the current part being built
        # whether the current part is made of letters or digits
        current_kind = None 

        def flush():
            # allow inner function to update from the outer function
            nonlocal current, current_kind
            # if there is a current part, save it into the result list
            if current:
                parts.append("".join(current))
                # reset the current part and its kind
                current = []
                current_kind = None

        for ch in text:
            # treat spaces and hyphens as separators.
            # save the current part, skip the separator, 
            # and continue reading
            # for example "Room-204A" should be ["Room", "204", "A"]
            if ch.isspace() or ch == "-":
                flush()
                continue

            if ch.isdigit():
                kind = "digit"
            elif ch.isalpha():
                kind = "alpha"
            else:
                flush()
                continue
            
            # start a new part if nothing is being built yet
            if current_kind is None:
                current.append(ch)
                current_kind = kind
            # keep adding characters if they are the same kind 
            # as the current part
            elif current_kind == kind:
                current.append(ch)
            # if the kind changes, 
            # save the old part and start a new one
            else:
                flush()
                current.append(ch)
                current_kind = kind
        # save the final part after the loop ends
        flush()
        return parts

    def _number_piece_to_tokens(self, piece: str) -> List[str]:
        """
        Convert a numeric text piece into word tokens.

        This is mainly used for number parts inside mixed entity 
        chunks, such as room numbers, codes, or postcodes.

        piece (str)  :  numeric text piece to convert

        Return:
        (List[str])  :  word tokens converted from the number piece
        """
        if not piece:
            return []

        converted = normalize_numbers_in_sentence(piece).lower().strip()
        if not converted:
            return []

        return [token for token in converted.split() if token]

    def _trace_plain_tokens(
        self, tokens: List[str], allow_alpha_fallback: bool
    ) -> Tuple[List[str], List[int]]:
        """
        Trace plain (normal) tokens into vocab keys and reorder tags.

        Scans the token list from left to right. It first tries to 
        match the longest vocab phrase from the current position. 
        
        If no vocab match is found, it can convert digit tokens into 
        number words and trace them again. When allowed, unknown 
        alphabetic tokens are spelled out letter by letter.

        tokens                (List[str])  :  plain input tokens to trace
        allow_alpha_fallback  (bool)       :  whether unknown alphabetic 
                                              tokens can be split into 
                                              single letters

        Returns:
        (Tuple[List[str], List[int]])      :  matched vocab keys and 
                                              their reorder tags.
        """
        results = []
        reorder_tags = []  # reorder tag for each result item
        i = 0  # current scanning index

        # scan tokens from left to right
        while i < len(tokens):
            # try find the longest vocab match from the current token
            match = self.match_from(tokens, i, entity_set=set())

            # if a vocab match is found, 
            # save it and jump to the token after the match
            if match is not None:
                vocab_key = match["vocab_key"]
                results.append(vocab_key)
                reorder_tags.append(get_reorder_tag(vocab_key))
                i = match["end_idx"]
                continue

            # get the current token if no vocab match was found
            current = tokens[i]

            # if the token is a number, 
            # convert it into number words and trace again
            if current.isdigit():
                number_tokens = self._number_piece_to_tokens(current)
                if number_tokens:
                    sub_results, sub_tags = self._trace_plain_tokens(
                        number_tokens,
                        allow_alpha_fallback=False,
                    )
                    results.extend(sub_results)
                    reorder_tags.extend(sub_tags)
                i += 1  # move to the next token
                continue

            # if allowed, 
            # spell unknown alphabetic tokens letter by letter
            if allow_alpha_fallback and current.isalpha():
                for ch in current.lower():
                    results.append(ch)
                    reorder_tags.append(get_reorder_tag(ch))

            i += 1  # move to the next token

        return results, reorder_tags

    def _trace_entity_chunk(self, chunk: str) -> Tuple[List[str], List[int]]:
        """
        Checks whether the whole entity chunk exactly exists in the vocab tree. 
        If it does, the whole chunk is kept as one matched vocab item.

        If there is no whole-phrase match, the entity chunk is split into smaller
        letter and digit parts, then traced again. 
        Unknown alphabetic parts are allowed to fall back to spelling letter by 
        letter.

        chunk (str)                    :  entity chunk to trace

        Return:
        (Tuple[List[str], List[int]])  :  traced vocab keys and their reorder tags
        """
        # try to match the whole entity chunk as one complete vocab phrase
        whole_match = self._lookup_exact_phrase(chunk)
        # if the whole entity exists in vocab, keep it as one result
        if whole_match is not None:
            return [whole_match], [get_reorder_tag(whole_match)]
        # if the whole chunk cannot be matched, 
        # split it into smaller letter/digit parts
        sub_tokens = self._split_mixed_entity_chunk(chunk)
         # if nothing useful is left after splitting, return empty results
        if not sub_tokens:
            return [], []

        # trace the split parts again
        # allow_alpha_fallback=True means 
        # unknown words can be spelled letter by letter
        return self._trace_plain_tokens(sub_tokens, allow_alpha_fallback=True)

    def trace(self, tokens: List[str], entities: List[str]
    ) -> Tuple[List[str], List[int]]:
        """
        Main tracing method (wrapper )of VocabTree. It scans the input tokens 
        from left to right, handles detected entity chunks first, then uses 
        the vocab Trie to find the longest matching vocab phrase from each 
        position.

        If a token cannot be matched directly but is numeric, it is converted 
        into number-word tokens and traced again.

        tokens    (List[str])  :  tokenized sentence chunks
        entities  (List[str])  :  detected entity chunks that should be handled 
                                  as protected chunks.

        Return:
        (Tuple[List[str], List[int]])  :  matched vocab keys and reorder tags
        """
        results = []
        reorder_tags = []
        entity_set = set(entities)
        i = 0

        while i < len(tokens):
            current = tokens[i]

            if current in entity_set:
                sub_results, sub_tags = self._trace_entity_chunk(current)
                results.extend(sub_results)
                reorder_tags.extend(sub_tags)
                i += 1
                continue

            match = self.match_from(tokens, i, entity_set=entity_set)

            if match is not None:
                vocab_key = match["vocab_key"]
                results.append(vocab_key)
                reorder_tags.append(get_reorder_tag(vocab_key))
                i = match["end_idx"]
                continue

            if current.isdigit():
                number_tokens = self._number_piece_to_tokens(current)
                if number_tokens:
                    sub_results, sub_tags = self._trace_plain_tokens(
                        number_tokens,
                        allow_alpha_fallback=False,
                    )
                    results.extend(sub_results)
                    reorder_tags.extend(sub_tags)

            i += 1

        return results, reorder_tags


# singleton instance initialization
vocab_tree = VocabTree()


# module-level wrappers provide a simple public API without exposing 
# the service instance
def load_vocab_tree(json_path: str, identity_metadata_path: str) -> None:
    # load the identity metadata in 
    load_identity_lookup(identity_metadata_path)
    # build the vocab tree
    vocab_tree.load_vocab_json(json_path)
    vocab_tree.build_tree()


def print_vocab_subtree(prefix=None) -> None:
    vocab_tree.print_subtree(prefix)


def trace_tokens(tokens: List[str], entities: List[str]
) -> Tuple[List[str], List[int]]:
    return vocab_tree.trace(tokens, entities)


def load_vocab_json(json_path: str) -> None:
    return vocab_tree.load_vocab_json(json_path)


# mainly for local run test
if __name__ == "__main__":
    vocab_tree = VocabTree()
    load_identity_lookup("../vocabs/all_vocabs_metadata.jsonl")
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
            "name": "postcode-like entity fallback",
            "tokens": ["the fishmonger", "bh1 1jq"],
            "entities": ["the fishmonger", "bh1 1jq"],
        },
    ]

    for case in test_cases:
        print(f"[TEST] {case['name']}")
        print("tokens   :", case["tokens"])
        print("entities :", case["entities"])

        traced, reorder_tags = vocab_tree.trace(case["tokens"], case["entities"])
        print("traced   :", traced)
        print("tags     :", reorder_tags)
