import os
import json
import argparse


def collect_folder_names(root_path):
    folder_names = []

    for name in os.listdir(root_path):
        full_path = os.path.join(root_path, name)
        if os.path.isdir(full_path):
            folder_names.append(name)

    folder_names.sort()
    return folder_names


def write_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write all immediate subfolder names into a JSON file."
    )
    parser.add_argument(
        "root_path",
        help="Path to the parent folder"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_path = args.root_path

    if not os.path.exists(root_path) or not os.path.isdir(root_path):
        write_json({
            "error": "Invalid folder",
            "root_path": root_path
        }, args.output)
        return

    folder_names = collect_folder_names(root_path)

    report = {
        "root_path": root_path,
        "total_folders": len(folder_names),
        "folder_names": folder_names
    }

    write_json(report, args.output)


if __name__ == "__main__":
    main()