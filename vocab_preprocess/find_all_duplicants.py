import json
import re
import argparse


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def get_folder_names(all_folders_json):
    folder_names = all_folders_json.get("folder_names", [])
    return set(folder_names)


def get_duplicate_groups(duplicants_json):
    groups = duplicants_json.get("duplicate_groups", {})
    if not isinstance(groups, dict):
        return {}
    return groups


def collect_duplicate_candidates(folder_names):
    """
    Build groups like:
      user -> [user, user_1, user_2]
    only when the base folder itself exists.
    """
    grouped = {}

    pattern = re.compile(r"^(.*)_(\d+)$")

    for name in sorted(folder_names):
        match = pattern.match(name)
        if not match:
            continue

        base_name = match.group(1)
        if base_name not in folder_names:
            continue

        if base_name not in grouped:
            grouped[base_name] = [base_name]

        grouped[base_name].append(name)

    # keep each group sorted as base, _1, _2, ...
    for base_name, values in grouped.items():
        suffix_names = [v for v in values if v != base_name]
        suffix_names.sort(key=lambda x: int(x[len(base_name) + 1:]))
        grouped[base_name] = [base_name] + suffix_names

    return grouped


def find_missing_duplicate_groups(folder_names, existing_groups):
    """
    Returns:
      - missing_groups: brand new groups absent from duplicants.json
      - incomplete_groups: existing groups missing some members
    """
    candidate_groups = collect_duplicate_candidates(folder_names)

    missing_groups = {}
    incomplete_groups = {}

    for base_name, candidate_values in candidate_groups.items():
        if base_name not in existing_groups:
            missing_groups[base_name] = candidate_values
            continue

        existing_values = set(existing_groups[base_name])
        missing_values = [v for v in candidate_values if v not in existing_values]

        if missing_values:
            incomplete_groups[base_name] = {
                "existing": list(existing_groups[base_name]),
                "missing": missing_values,
                "final": candidate_values
            }

    return missing_groups, incomplete_groups


def append_missing_groups(existing_groups, missing_groups, incomplete_groups):
    appended_group_count = 0
    appended_value_count = 0

    for key, values in missing_groups.items():
        if key not in existing_groups:
            existing_groups[key] = list(values)
            appended_group_count += 1
            appended_value_count += len(values)

    for key, details in incomplete_groups.items():
        if key not in existing_groups:
            existing_groups[key] = list(details["final"])
            appended_group_count += 1
            appended_value_count += len(details["final"])
        else:
            for value in details["missing"]:
                if value not in existing_groups[key]:
                    existing_groups[key].append(value)
                    appended_value_count += 1

            # keep base first, then numeric suffix order
            base_name = key
            suffix_names = [v for v in existing_groups[key] if v != base_name]
            suffix_names.sort(key=lambda x: int(x[len(base_name) + 1:]))
            existing_groups[key] = [base_name] + suffix_names

    return appended_group_count, appended_value_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare all folder names against duplicants.json and append any missing duplicate members/groups."
    )
    parser.add_argument(
        "all_folders_json",
        help="Path to all_folder_names.json"
    )
    parser.add_argument(
        "duplicants_json",
        help="Path to duplicants.json"
    )
    parser.add_argument(
        "--report-output",
        help="Optional path to write a comparison report JSON"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    all_folders_data = load_json(args.all_folders_json)
    duplicants_data = load_json(args.duplicants_json)

    folder_names = get_folder_names(all_folders_data)
    existing_groups = get_duplicate_groups(duplicants_data)

    missing_groups, incomplete_groups = find_missing_duplicate_groups(folder_names, existing_groups)

    appended_group_count, appended_value_count = append_missing_groups(
        existing_groups,
        missing_groups,
        incomplete_groups
    )

    duplicants_data["duplicate_groups"] = existing_groups
    save_json(args.duplicants_json, duplicants_data)

    report = {
        "all_folders_json": args.all_folders_json,
        "duplicants_json": args.duplicants_json,
        "missing_group_count": len(missing_groups),
        "incomplete_group_count": len(incomplete_groups),
        "appended_group_count": appended_group_count,
        "appended_value_count": appended_value_count,
        "missing_groups_found": missing_groups,
        "incomplete_groups_found": incomplete_groups
    }

    if args.report_output:
        save_json(args.report_output, report)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()