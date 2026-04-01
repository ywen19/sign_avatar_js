import os
import json
import argparse


def remove_apostrophe(name):
    return name.replace("'", "")


def get_current_folder_names(root_path):
    return [
        name for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]


def split_file_suffix(file_name, old_folder_name):
    base_name, ext = os.path.splitext(file_name)

    if base_name == old_folder_name:
        return "", ext

    prefix = old_folder_name + "_"
    if base_name.startswith(prefix):
        suffix = base_name[len(old_folder_name):]   # keeps the leading "_"
        return suffix, ext

    return None, ext


def build_unique_name(target_name, used_names, duplicate_groups):
    if target_name not in used_names:
        used_names.add(target_name)
        return target_name

    if target_name not in duplicate_groups:
        duplicate_groups[target_name] = [target_name]

    index = 1
    while True:
        candidate = "{}_{}".format(target_name, index)
        if candidate not in used_names:
            used_names.add(candidate)
            duplicate_groups[target_name].append(candidate)
            return candidate
        index += 1


def plan_operations(root_path):
    folder_names = get_current_folder_names(root_path)
    used_names = set(folder_names)
    duplicate_groups = {}
    operations = []

    for old_name in folder_names:
        if "'" not in old_name:
            continue

        used_names.remove(old_name)

        base_target = remove_apostrophe(old_name)
        final_target = build_unique_name(base_target, used_names, duplicate_groups)

        operations.append({
            "source_folder": old_name,
            "target_folder": final_target
        })

    return operations, duplicate_groups


def plan_file_operations(folder_path, old_folder_name, new_folder_name):
    file_operations = []

    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(full_path):
            continue

        suffix, ext = split_file_suffix(file_name, old_folder_name)
        if suffix is None:
            continue

        new_file_name = new_folder_name + suffix + ext
        file_operations.append({
            "old_name": file_name,
            "new_name": new_file_name,
            "status": "planned"
        })

    return file_operations


def execute_operations(root_path, operations, dry_run=False):
    report_operations = []

    folder_renamed_count = 0
    folder_skipped_count = 0
    file_renamed_count = 0
    file_skipped_count = 0

    for operation in operations:
        source_folder = operation["source_folder"]
        target_folder = operation["target_folder"]

        source_path = os.path.join(root_path, source_folder)
        target_path = os.path.join(root_path, target_folder)

        op_report = {
            "source_folder": source_folder,
            "targets": [
                {
                    "folder_name": target_folder,
                    "mode": "rename",
                    "file_operations": []
                }
            ],
            "status": "planned" if dry_run else "done"
        }

        if not os.path.exists(source_path):
            op_report["status"] = "missing_source"
            folder_skipped_count += 1
            report_operations.append(op_report)
            continue

        file_operations = plan_file_operations(source_path, source_folder, target_folder)

        if dry_run:
            op_report["targets"][0]["file_operations"] = file_operations
            folder_renamed_count += 1
            file_renamed_count += len(file_operations)
            report_operations.append(op_report)
            continue

        if os.path.exists(target_path):
            op_report["status"] = "target_exists"
            folder_skipped_count += 1
            report_operations.append(op_report)
            continue

        os.rename(source_path, target_path)
        folder_renamed_count += 1

        executed_file_operations = []

        for file_op in file_operations:
            old_file_path = os.path.join(target_path, file_op["old_name"])
            new_file_path = os.path.join(target_path, file_op["new_name"])

            file_record = {
                "old_name": file_op["old_name"],
                "new_name": file_op["new_name"],
                "status": None
            }

            if os.path.exists(old_file_path) and not os.path.exists(new_file_path):
                os.rename(old_file_path, new_file_path)
                file_record["status"] = "renamed"
                file_renamed_count += 1
            else:
                file_record["status"] = "skipped"
                file_skipped_count += 1

            executed_file_operations.append(file_record)

        op_report["targets"][0]["file_operations"] = executed_file_operations
        report_operations.append(op_report)

    report = {
        "root_path": root_path,
        "mode": "dry_run" if dry_run else "live",
        "total_operations": len(operations),
        "folder_renamed_count": folder_renamed_count,
        "folder_skipped_count": folder_skipped_count,
        "file_renamed_count": file_renamed_count,
        "file_skipped_count": file_skipped_count,
        "operations": report_operations
    }

    return report


def load_existing_duplicate_groups(path):
    if not path:
        return {}

    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "duplicate_groups" in data and isinstance(data["duplicate_groups"], dict):
        return data["duplicate_groups"]

    if isinstance(data, dict):
        return data

    return {}


def merge_duplicate_groups(existing_groups, new_groups):
    merged = {}

    for key, values in existing_groups.items():
        merged[key] = list(values)

    for key, values in new_groups.items():
        if key not in merged:
            merged[key] = list(values)
        else:
            for value in values:
                if value not in merged[key]:
                    merged[key].append(value)

    return merged


def write_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apostrophe cleanup for folder names and matching files."
    )
    parser.add_argument(
        "root_path",
        help="Path to the parent folder"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path"
    )
    parser.add_argument(
        "--duplicate-json",
        help="Existing duplicate-groups JSON to merge into during live run"
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

    operations, duplicate_groups = plan_operations(root_path)

    if args.dry_run:
        dryrun_report = execute_operations(root_path, operations, dry_run=True)
        write_json(dryrun_report, args.output)
        return

    execute_operations(root_path, operations, dry_run=False)

    existing_groups = load_existing_duplicate_groups(args.duplicate_json)
    merged_groups = merge_duplicate_groups(existing_groups, duplicate_groups)

    duplicate_report = {
        "root_path": root_path,
        "mode": "live",
        "duplicate_groups": merged_groups
    }
    write_json(duplicate_report, args.output)


if __name__ == "__main__":
    main()