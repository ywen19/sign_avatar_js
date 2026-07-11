"""
We annotate the identity of chunks inside the vocab.
This is mainly for reordering contexts in BSL syntax.

Manually check and fix is acquired after this automation.
"""

import argparse
import json
from pathlib import Path
import sys

from typing import Final, Dict, Set, List


# add project root for module import
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from language_utils import load_model, cleanup, get_response


# only some identities will actively affect reordering for now
IDENTITY_RULES: Dict[str, str] = {
    "entity": "protect",
    "question_phrase": "protect",
    "time": "start",
    "wh": "end",
    "negation": "end",
    # "pronoun": "unchanged",
    # "location": "unchanged",
    "normal": "unchanged",
}

wh_words: Set[str] = {"what", "when", "where", "who", "why", "how",
    "whats", "whens", "wheres", "whos", "why", "hows", "which"}

yes_no_words: Set[str] = {
    "is", "are", "am",
    "was", "were",
    "can", "could",
    "do", "does", "did",
    "have", "has", "had",
    "will", "would",
    "shall", "should",
    "may", "might",
    "isnt", "arent", "wasnt", "werent",
    "cant", "couldnt",
    "dont", "doesnt", "didnt",
    "havent", "hasnt", "hadnt",
    "wont", "wouldnt",
    "shouldnt", "shallnt",
    "maynt", "mightnt",
}

negation_words: Set[str] = {
    "not", "no", "never",
    "cannot", "cant",
    "dont", "doesnt", "didnt",
    "isnt", "arent", "aint",
    "wasnt", "werent",
    "wont", "wouldnt",
    "shouldnt", "couldnt", "mustnt",
    "havent", "hasnt", "hadnt",
    "without",
}


def load_vocab_json(json_path: str) -> List[str]:
    """
    Read in all vocabs to be labeled from a json file.

    json_path  (str)  :  path to the json file where all vocabs are stored

    Return:
    (List[str])       :  all vocabs stored in a list
    """
    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(f"Vocab file not found: {json_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in: {json_path}")

    vocab_list = []
    for item in data:
        if not isinstance(item, str):
            continue

        vocab_key = item.strip()
        if vocab_key:
            vocab_list.append(vocab_key)

    return vocab_list


def is_wh_word_or_wh_phrase(words: List[str]) -> str:
    """
    Detect if input is question phrase starts with wh word or wh word.

    For now we decide length more than 1 will be phrase.
    This is not ideal if we extend the vocab, but for now, 
    such logic works because the vocab contains only either wh word
    or question sentence.

    words  (List[str])  :  vocab chunk (split into list by word)

    Return:
    (str)               :  identity label
    """
    if words[0] in wh_words:
        return "wh" if len(words) == 1 else "question_phrase"
    return None


def is_yes_no_question_phrase(words: List[str]) -> str:
    """
    Detect if input is question phrase not starts with wh word.

    words  (List[str])  :  vocab chunk (split into list by word)

    Return:
    (str)               :  identity label
    """
    if words[0] in yes_no_words and len(words) > 1:
        return "question_phrase"
    return None


def is_negation(words: List[str]) -> str:
    """
    Detect if input is negation word.

    words  (List[str])  :  vocab chunk (split into list by word)

    Return:
    (str)               :  identity label
    """
    if len(words) == 1 and words[0] in negation_words:
        return "negation"
    return None


def is_time_by_smollm(formatted_chunk: str)  -> str:
    """
    Detect time identity rule by pretrained llm as classfier.

    formatted_chunk  (str)  :  normalized vocab chunk (replace _ by space)

    Return:
    (str)                   :  identity label
    """
    prompt = (
        f'Is "{formatted_chunk}" a time expression used to indicate time, date, '
        f'duration, frequency, or temporal reference? '
        f'Answer only "yes" or "no".'
    )
    result = get_response(prompt).strip().lower()
    if result.startswith("yes"):
        return "time"
    return None 


def find_chunk_indentity(vocab_chunk: str) -> str:
    """
    Wrapper function to detect vocab chunk identity.

    vocab_chunk  (str)  :  vocab chunk

    Return:
    (str)               :  identity label
    """
    formatted_chunk = vocab_chunk.replace("_", " ").strip()
    words = formatted_chunk.split()
    wh_state = is_wh_word_or_wh_phrase(words)
    if wh_state:
        return wh_state
    yes_no_state = is_yes_no_question_phrase(words)
    if yes_no_state:
        return yes_no_state
    negation_state = is_negation(words)
    if negation_state:
        return negation_state
    time_state = is_time_by_smollm(formatted_chunk)
    if time_state:
        return time_state
    return "normal"


def append_jsonl_record(output_path: str, chunk: str, identity: str) -> None:
    """
    Append the vocab chunk and its detected identity to metadata file (JSONL).

    output_path  (str)  :  metadata file path
    chunk        (str)  :  vocab chunk
    identity     (str)  :  detected chunk identity
    """
    record = {
        "chunk": chunk,
        "identity": identity,
    }
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vocab_file",
        type=str,
        default="../vocabs/all_vocabs.json",
    )

    parser.add_argument(
        "--vocab_output",
        type=str,
        default="../vocabs/all_vocabs_metadata.jsonl",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    vocab_list = load_vocab_json(args.vocab_file)

    print(f"Loaded {len(vocab_list)} vocab items from: {args.vocab_file}")
    print(f"Output JSONL will be written to: {args.vocab_output}")

    # load smollm model 
    load_model()

    try:
        for vocab_chunk in vocab_list:
            chunk_state = find_chunk_indentity(vocab_chunk)
            append_jsonl_record(args.vocab_output, vocab_chunk, chunk_state)
    finally:
        cleanup()

if __name__ == "__main__":
    main()