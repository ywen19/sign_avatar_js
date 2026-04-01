import json
import argparse


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def lowercase_duplicate_groups(duplicate_groups):
    merged = {}
    collisions = {}

    for key, values in duplicate_groups.items():
        lower_key = key.lower()
        lower_values = [value.lower() for value in values]

        if lower_key not in merged:
            merged[lower_key] = []
        else:
            collisions.setdefault(lower_key, []).append(key)

        for value in lower_values:
            if value not in merged[lower_key]:
                merged[lower_key].append(value)

    return merged, collisions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lowercase duplicate_groups JSON safely, merging any key collisions."
    )
    parser.add_argument(
        "input_json",
        help="Path to duplicants.json"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path"
    )
    parser.add_argument(
        "--report-output",
        help="Optional path to write collision report JSON"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data = load_json(args.input_json)
    duplicate_groups = data.get("duplicate_groups", {})

    lowered_groups, collisions = lowercase_duplicate_groups(duplicate_groups)

    output_data = dict(data)
    output_data["duplicate_groups"] = lowered_groups

    write_json(args.output, output_data)

    report = {
        "input_json": args.input_json,
        "output_json": args.output,
        "original_group_count": len(duplicate_groups),
        "lowercased_group_count": len(lowered_groups),
        "collisions": collisions
    }

    if args.report_output:
        write_json(args.report_output, report)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()