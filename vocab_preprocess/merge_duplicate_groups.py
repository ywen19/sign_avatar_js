import os
import re
import json
import shutil
import uuid
import argparse


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def sort_group_members(group_values, base_name):
    def sort_key(name):
        if name == base_name:
            return (0, -1, name)

        m = re.match(r"^(.*)_(\d+)$", name)
        if m and m.group(1) == base_name:
            return (1, int(m.group(2)), name)

        return (2, 10**9, name)

    return sorted(group_values, key=sort_key)


def list_files(folder_path):
    return sorted(
        [
            name for name in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, name))
        ]
    )


def make_temp_name(file_name):
    return "__tmp__{}__{}".format(uuid.uuid4().hex, file_name)


def build_base_rename_plan(base_folder, base_files):
    plan = []
    for index, file_name in enumerate(sorted(base_files)):
        _, ext = os.path.splitext(file_name)
        target_file = "{}_{}{}".format(base_folder, index, ext)
        plan.append({
            "old_name": file_name,
            "new_name": target_file
        })
    return plan


def build_suffix_rename_plan(base_folder, suffix_files, start_index):
    plan = []
    next_index = start_index

    for file_name in sorted(suffix_files):
        _, ext = os.path.splitext(file_name)
        target_file = "{}_{}{}".format(base_folder, next_index, ext)
        plan.append({
            "old_name": file_name,
            "new_name": target_file,
            "assigned_index": next_index
        })
        next_index += 1

    return plan, next_index - 1


def collect_group_plan(root_path, base_name, group_values):
    ordered_members = sort_group_members(group_values, base_name)

    existing_folders = []
    missing_folders = []

    for folder_name in ordered_members:
        folder_path = os.path.join(root_path, folder_name)
        if os.path.isdir(folder_path):
            existing_folders.append(folder_name)
        else:
            missing_folders.append(folder_name)

    if base_name not in existing_folders:
        return {
            "base_folder": base_name,
            "group_members": ordered_members,
            "existing_folders": existing_folders,
            "missing_folders": missing_folders,
            "status": "base_missing"
        }

    base_path = os.path.join(root_path, base_name)
    base_files = list_files(base_path)
    base_rename_plan = build_base_rename_plan(base_name, base_files)
    largest_index = len(base_rename_plan) - 1

    suffix_operations = []
    total_suffix_files = 0

    for folder_name in ordered_members:
        if folder_name == base_name:
            continue
        if folder_name not in existing_folders:
            continue

        folder_path = os.path.join(root_path, folder_name)
        suffix_files = list_files(folder_path)

        if not suffix_files:
            suffix_operations.append({
                "folder_name": folder_name,
                "files_before": [],
                "rename_plan": [],
                "move_plan": [],
                "delete_folder": True
            })
            continue

        rename_plan, largest_index = build_suffix_rename_plan(
            base_name,
            suffix_files,
            largest_index + 1
        )

        move_plan = [
            {
                "source_folder": folder_name,
                "source_file_after_rename": item["new_name"],
                "target_folder": base_name,
                "target_file": item["new_name"]
            }
            for item in rename_plan
        ]

        suffix_operations.append({
            "folder_name": folder_name,
            "files_before": suffix_files,
            "rename_plan": rename_plan,
            "move_plan": move_plan,
            "delete_folder": True
        })

        total_suffix_files += len(suffix_files)

    if total_suffix_files == 0:
        status = "no_suffix_files"
    else:
        status = "planned"

    return {
        "base_folder": base_name,
        "group_members": ordered_members,
        "existing_folders": existing_folders,
        "missing_folders": missing_folders,
        "base_files_before": base_files,
        "base_rename_plan": base_rename_plan,
        "largest_index_after_base": len(base_rename_plan) - 1,
        "suffix_operations": suffix_operations,
        "status": status
    }


def build_plan(root_path, duplicate_groups, only_group=None):
    operations = []

    keys = sorted(duplicate_groups.keys())
    if only_group:
        keys = [key for key in keys if key == only_group]

    for base_name in keys:
        group_values = duplicate_groups[base_name]
        if not isinstance(group_values, list) or len(group_values) < 2:
            continue

        operation = collect_group_plan(root_path, base_name, group_values)
        operations.append(operation)

    return operations


def rename_files_with_temp(folder_path, rename_plan):
    results = []
    temp_records = []
    errors = []

    # First pass: rename old -> temp
    for item in rename_plan:
        old_path = os.path.join(folder_path, item["old_name"])
        if not os.path.exists(old_path):
            errors.append("Missing file: {}".format(old_path))
            continue

        temp_name = make_temp_name(item["old_name"])
        temp_path = os.path.join(folder_path, temp_name)

        while os.path.exists(temp_path):
            temp_name = make_temp_name(item["old_name"])
            temp_path = os.path.join(folder_path, temp_name)

        try:
            os.rename(old_path, temp_path)
            temp_records.append({
                "temp_name": temp_name,
                "new_name": item["new_name"],
                "old_name": item["old_name"]
            })
        except Exception as e:
            errors.append(
                "Failed renaming {} -> {}: {}".format(old_path, temp_path, str(e))
            )

    # Second pass: rename temp -> final
    for record in temp_records:
        temp_path = os.path.join(folder_path, record["temp_name"])
        final_path = os.path.join(folder_path, record["new_name"])

        if os.path.exists(final_path):
            errors.append("Final target already exists: {}".format(final_path))
            continue

        try:
            os.rename(temp_path, final_path)
            results.append({
                "old_name": record["old_name"],
                "new_name": record["new_name"],
                "status": "renamed"
            })
        except Exception as e:
            errors.append(
                "Failed renaming {} -> {}: {}".format(temp_path, final_path, str(e))
            )

    return results, errors


def execute_group(root_path, operation):
    base_folder = operation["base_folder"]
    base_path = os.path.join(root_path, base_folder)

    result = {
        "base_folder": base_folder,
        "status": operation["status"],
        "existing_folders": operation.get("existing_folders", []),
        "missing_folders": operation.get("missing_folders", []),
        "base_files_before_count": len(operation.get("base_files_before", [])),
        "suffix_file_count_before": sum(
            len(item.get("files_before", []))
            for item in operation.get("suffix_operations", [])
        ),
        "base_renamed_files": [],
        "suffix_renamed_files": [],
        "moved_files": [],
        "deleted_folders": [],
        "errors": []
    }

    if operation["status"] == "base_missing":
        result["errors"].append("Base folder missing.")
        return result

    if operation["status"] == "no_suffix_files":
        result["status"] = "no_action_needed"
        return result

    # 1. Normalize base folder files to base_0, base_1, ...
    base_results, base_errors = rename_files_with_temp(base_path, operation["base_rename_plan"])
    result["base_renamed_files"] = base_results
    result["errors"].extend(base_errors)

    # 2. Rename files in suffix folders with continuing index, then move them
    for suffix_op in operation["suffix_operations"]:
        folder_name = suffix_op["folder_name"]
        folder_path = os.path.join(root_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # rename inside suffix folder first
        suffix_results, suffix_errors = rename_files_with_temp(folder_path, suffix_op["rename_plan"])
        result["suffix_renamed_files"].append({
            "folder_name": folder_name,
            "files": suffix_results
        })
        result["errors"].extend(suffix_errors)

        # move renamed files into base folder
        for item in suffix_op["rename_plan"]:
            renamed_file_name = item["new_name"]
            source_path = os.path.join(folder_path, renamed_file_name)
            target_path = os.path.join(base_path, renamed_file_name)

            if not os.path.exists(source_path):
                result["errors"].append("Missing renamed suffix file: {}".format(source_path))
                continue

            if os.path.exists(target_path):
                result["errors"].append("Target file already exists in base: {}".format(target_path))
                continue

            try:
                shutil.move(source_path, target_path)
                result["moved_files"].append({
                    "source_folder": folder_name,
                    "file_name": renamed_file_name,
                    "target_folder": base_folder
                })
            except Exception as e:
                result["errors"].append(
                    "Failed moving {} -> {}: {}".format(source_path, target_path, str(e))
                )

        # remove suffix folder if empty
        try:
            remaining = os.listdir(folder_path)
            if remaining:
                result["errors"].append(
                    "Folder not empty after move: {} | remaining={}".format(folder_name, remaining)
                )
            else:
                os.rmdir(folder_path)
                result["deleted_folders"].append(folder_name)
        except Exception as e:
            result["errors"].append(
                "Failed deleting folder {}: {}".format(folder_name, str(e))
            )

    if result["errors"]:
        result["status"] = "completed_with_errors"
    else:
        result["status"] = "done"

    result["base_files_after_count"] = len(list_files(base_path))
    return result


def execute_plan(root_path, operations, dry_run=False):
    if dry_run:
        return {
            "root_path": root_path,
            "mode": "dry_run",
            "total_groups": len(operations),
            "operations": operations
        }

    execution_results = []
    groups_done = 0
    groups_with_errors = 0
    groups_no_action = 0

    for operation in operations:
        result = execute_group(root_path, operation)
        execution_results.append(result)

        if result["status"] == "done":
            groups_done += 1
        elif result["status"] == "no_action_needed":
            groups_no_action += 1
        else:
            groups_with_errors += 1

    return {
        "root_path": root_path,
        "mode": "live",
        "total_groups": len(operations),
        "groups_done": groups_done,
        "groups_no_action": groups_no_action,
        "groups_with_errors": groups_with_errors,
        "operations": execution_results
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge duplicate-group folders into the base folder and reindex files safely."
    )
    parser.add_argument(
        "root_path",
        help="Path to the parent folder containing vocab folders"
    )
    parser.add_argument(
        "duplicants_json",
        help="Path to duplicants.json"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only; do not modify files"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the JSON report"
    )
    parser.add_argument(
        "--only-group",
        help="Process only one duplicate group key for safe testing"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.root_path):
        write_json(args.output, {
            "error": "Invalid root folder",
            "root_path": args.root_path
        })
        return

    duplicants_data = load_json(args.duplicants_json)
    duplicate_groups = duplicants_data.get("duplicate_groups", {})

    if not isinstance(duplicate_groups, dict):
        write_json(args.output, {
            "error": "Invalid duplicate_groups structure",
            "duplicants_json": args.duplicants_json
        })
        return

    if args.only_group and args.only_group not in duplicate_groups:
        write_json(args.output, {
            "error": "Requested group not found in duplicate_groups",
            "only_group": args.only_group
        })
        return

    operations = build_plan(
        args.root_path,
        duplicate_groups,
        only_group=args.only_group
    )

    report = execute_plan(args.root_path, operations, dry_run=args.dry_run)
    write_json(args.output, report)


if __name__ == "__main__":
    main()