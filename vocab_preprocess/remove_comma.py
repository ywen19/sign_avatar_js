import os
import re
import json
import shutil
import argparse


RULES = {
    "Hello,_my_name_is": {
        "action": "remove_comma"
    },
    "Hello,_how_are_you_": {
        "action": "remove_comma"
    },
    "Tap_You,_Hello": {
        "action": "remove_comma"
    },
    "Again_please,_slowly": {
        "action": "remove_comma"
    },
    "Are_you_happy,_not_happy_": {
        "action": "remove_comma"
    },
    "I_Don't_Understand,_Again_Please": {
        "action": "remove_comma"
    },
    "What_About_You,_Your_Response": {
        "action": "remove_comma"
    },
    "Difficult,_Problem": {
        "action": "remove_comma"
    },
    "angry,_very": {
        "action": "rewrite",
        "target": "very_angry"
    },
    "Nothing,_Nobody": {
        "action": "split_alias",
        "targets": ["Nothing", "Nobody"]
    },
    "Two_Of_Us,_We_Two": {
        "action": "split_alias",
        "targets": ["Two_Of_Us", "We_Two"]
    },
    "both,_we": {
        "action": "split_alias",
        "targets": ["both", "we"]
    },
    "Always_Helping,_Help_Continuously": {
        "action": "split_alias",
        "targets": ["Always_Helping", "Help_Continuously"]
    },
}


def remove_comma_name(name):
    name = name.replace(",", "")
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name


def split_file_suffix(file_name, old_folder_name):
    base_name, ext = os.path.splitext(file_name)

    if base_name == old_folder_name:
        return "", ext

    prefix = old_folder_name + "_"
    if base_name.startswith(prefix):
        suffix = base_name[len(prefix):]
        return "_" + suffix, ext

    return None, ext


def get_current_folder_names(root_path):
    return [
        name for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]


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


def plan_file_ops_for_single(folder_path, old_folder_name, new_folder_name):
    file_ops = []

    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(full_path):
            continue

        suffix, ext = split_file_suffix(file_name, old_folder_name)
        if suffix is None:
            continue

        new_file_name = new_folder_name + suffix + ext
        file_ops.append({
            "old_name": file_name,
            "new_name": new_file_name,
            "status": "planned"
        })

    return file_ops


def plan_operations(root_path):
    folder_names = get_current_folder_names(root_path)
    used_names = set(folder_names)
    duplicate_groups = {}
    operations = []

    for old_name in folder_names:
        if old_name not in RULES:
            continue

        rule = RULES[old_name]
        action = rule["action"]

        used_names.remove(old_name)

        if action == "remove_comma":
            base_target = remove_comma_name(old_name)
            final_target = build_unique_name(base_target, used_names, duplicate_groups)

            operations.append({
                "action": action,
                "source_folder": old_name,
                "targets": [final_target]
            })

        elif action == "rewrite":
            base_target = rule["target"]
            final_target = build_unique_name(base_target, used_names, duplicate_groups)

            operations.append({
                "action": action,
                "source_folder": old_name,
                "targets": [final_target]
            })

        elif action == "split_alias":
            final_targets = []
            for base_target in rule["targets"]:
                final_target = build_unique_name(base_target, used_names, duplicate_groups)
                final_targets.append(final_target)

            operations.append({
                "action": action,
                "source_folder": old_name,
                "targets": final_targets
            })

    return operations, duplicate_groups


def execute_operations(root_path, operations, dry_run=False):
    report_ops = []

    folder_renamed_count = 0
    folder_created_count = 0
    folder_skipped_count = 0
    file_renamed_count = 0
    file_skipped_count = 0

    for op in operations:
        source_folder = op["source_folder"]
        source_path = os.path.join(root_path, source_folder)
        action = op["action"]
        targets = op["targets"]

        op_report = {
            "action": action,
            "source_folder": source_folder,
            "targets": [],
            "status": "planned" if dry_run else "done"
        }

        if not os.path.exists(source_path):
            op_report["status"] = "missing_source"
            folder_skipped_count += 1
            report_ops.append(op_report)
            continue

        if action in ("remove_comma", "rewrite"):
            target_folder = targets[0]
            target_path = os.path.join(root_path, target_folder)

            file_ops = plan_file_ops_for_single(source_path, source_folder, target_folder)

            if dry_run:
                op_report["targets"].append({
                    "folder_name": target_folder,
                    "mode": "rename",
                    "file_operations": file_ops
                })
                folder_renamed_count += 1
                file_renamed_count += len(file_ops)
                report_ops.append(op_report)
                continue

            if os.path.exists(target_path):
                op_report["status"] = "target_exists"
                folder_skipped_count += 1
                report_ops.append(op_report)
                continue

            os.rename(source_path, target_path)
            folder_renamed_count += 1

            executed_file_ops = []
            for file_op in file_ops:
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

                executed_file_ops.append(file_record)

            op_report["targets"].append({
                "folder_name": target_folder,
                "mode": "rename",
                "file_operations": executed_file_ops
            })

        elif action == "split_alias":
            original_file_ops = {}
            for target_folder in targets:
                original_file_ops[target_folder] = plan_file_ops_for_single(
                    source_path, source_folder, target_folder
                )

            if dry_run:
                first = True
                for target_folder in targets:
                    op_report["targets"].append({
                        "folder_name": target_folder,
                        "mode": "rename" if first else "duplicate",
                        "file_operations": original_file_ops[target_folder]
                    })
                    if first:
                        folder_renamed_count += 1
                        first = False
                    else:
                        folder_created_count += 1
                    file_renamed_count += len(original_file_ops[target_folder])

                report_ops.append(op_report)
                continue

            conflicting_targets = [
                target for target in targets
                if os.path.exists(os.path.join(root_path, target))
            ]

            if conflicting_targets:
                op_report["status"] = "target_exists"
                op_report["conflicting_targets"] = conflicting_targets
                folder_skipped_count += 1
                report_ops.append(op_report)
                continue

            first_target_folder = targets[0]
            first_target_path = os.path.join(root_path, first_target_folder)

            os.rename(source_path, first_target_path)
            folder_renamed_count += 1

            for file_name in os.listdir(first_target_path):
                full_path = os.path.join(first_target_path, file_name)
                if not os.path.isfile(full_path):
                    continue

                suffix, ext = split_file_suffix(file_name, source_folder)
                if suffix is None:
                    continue

                new_file_name = first_target_folder + suffix + ext
                new_file_path = os.path.join(first_target_path, new_file_name)

                if not os.path.exists(new_file_path):
                    os.rename(full_path, new_file_path)

            for target_folder in targets[1:]:
                target_path = os.path.join(root_path, target_folder)
                shutil.copytree(first_target_path, target_path)
                folder_created_count += 1

            for target_folder in targets:
                target_path = os.path.join(root_path, target_folder)
                executed_file_ops = []

                for file_name in os.listdir(target_path):
                    full_path = os.path.join(target_path, file_name)
                    if not os.path.isfile(full_path):
                        continue

                    base_name, ext = os.path.splitext(file_name)
                    prefix = first_target_folder

                    if base_name == prefix:
                        suffix = ""
                    elif base_name.startswith(prefix + "_"):
                        suffix = base_name[len(prefix):]
                    else:
                        continue

                    new_file_name = target_folder + suffix + ext
                    new_file_path = os.path.join(target_path, new_file_name)

                    if file_name == new_file_name:
                        executed_file_ops.append({
                            "old_name": file_name,
                            "new_name": new_file_name,
                            "status": "unchanged"
                        })
                        continue

                    if not os.path.exists(new_file_path):
                        os.rename(full_path, new_file_path)
                        executed_file_ops.append({
                            "old_name": file_name,
                            "new_name": new_file_name,
                            "status": "renamed"
                        })
                        file_renamed_count += 1
                    else:
                        executed_file_ops.append({
                            "old_name": file_name,
                            "new_name": new_file_name,
                            "status": "skipped"
                        })
                        file_skipped_count += 1

                op_report["targets"].append({
                    "folder_name": target_folder,
                    "mode": "rename" if target_folder == first_target_folder else "duplicate",
                    "file_operations": executed_file_ops
                })

        report_ops.append(op_report)

    report = {
        "root_path": root_path,
        "mode": "dry_run" if dry_run else "live",
        "total_operations": len(operations),
        "folder_renamed_count": folder_renamed_count,
        "folder_created_count": folder_created_count,
        "folder_skipped_count": folder_skipped_count,
        "file_renamed_count": file_renamed_count,
        "file_skipped_count": file_skipped_count,
        "operations": report_ops
    }

    return report


def write_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comma cleanup for manually reviewed folder names."
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
        report = execute_operations(root_path, operations, dry_run=True)
        write_json(report, args.output)
    else:
        execute_operations(root_path, operations, dry_run=False)
        write_json({
            "root_path": root_path,
            "mode": "live",
            "duplicate_groups": duplicate_groups
        }, args.output)


if __name__ == "__main__":
    main()