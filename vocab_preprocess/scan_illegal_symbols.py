import os
import re
import json
import argparse
from collections import Counter


def normalize_name(name):
    name = name.lower()
    name = name.replace(".", "")
    name = name.replace(" ", "_")
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name


def display_char(ch):
    if ch == " ":
        return "space"
    if ch == "\t":
        return "\\t"
    if ch == "\n":
        return "\\n"
    return ch


def collect_illegal_symbols(root_path):
    symbol_counter = Counter()
    folders_with_illegal_symbols = []
    total_folders_scanned = 0

    for current_path, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            total_folders_scanned += 1
            illegal_chars = [ch for ch in dirname if not (ch.isalnum() or ch == "_")]

            if illegal_chars:
                full_path = os.path.join(current_path, dirname)
                symbol_counter.update(illegal_chars)
                folders_with_illegal_symbols.append({
                    "path": full_path,
                    "folder_name": dirname,
                    "illegal_chars": [display_char(ch) for ch in illegal_chars],
                    "normalized_name": normalize_name(dirname),
                })

    report = {
        "root_path": root_path,
        "total_folders_scanned": total_folders_scanned,
        "folders_with_illegal_symbols": len(folders_with_illegal_symbols),
        "illegal_symbol_counts": {
            display_char(symbol): count
            for symbol, count in sorted(symbol_counter.items(), key=lambda x: x[0])
        },
        "affected_folders": folders_with_illegal_symbols,
    }

    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan folder names and output illegal symbol report as JSON."
    )
    parser.add_argument(
        "root_path",
        help="Path to the parent folder"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_path = args.root_path

    if not os.path.exists(root_path) or not os.path.isdir(root_path):
        print("Invalid folder:", root_path)
        return

    report = collect_illegal_symbols(root_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()