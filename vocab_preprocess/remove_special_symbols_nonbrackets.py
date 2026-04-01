import os
import argparse
import json


def normalize_name(name):
    name = name.replace(".", "")
    name = name.replace("-", "")
    name = name.replace(" ", "_")
    return name


def build_unique_name(target_name, used_names, original_name):
    if target_name == original_name:
        return target_name

    if target_name not in used_names:
        return target_name

    index = 1
    while True:
        candidate = "{}_{}".format(target_name, index)
        if candidate not in used_names:
            return candidate
        index += 1


def plan_renames(root_path):
    entries = os.listdir(root_path)
    folder_names = []

    for name in entries:
        full_path = os.path.join(root_path, name)
        if os.path.isdir(full_path):
            folder_names.append(name)

    used_names = set(folder_names)
    rename_plan = []

    for old_name in folder_names:
        normalized_name = normalize_name(old_name)
        final_name = build_unique_name(normalized_name, used_names, old_name)

        if final_name != old_name:
            used_names.remove(old_name)
            used_names.add(final_name)

            rename_plan.append({
                "old_name": old_name,
                "normalized_name": normalized_name,
                "final_name": final_name,
            })

    return rename_plan


def build_new_file_name(old_file_name, old_folder_name, new_folder_name):
    base_name, ext = os.path.splitext(old_file_name)

    if base_name == old_folder_name:
        new_base_name = new_folder_name
    elif base_name.startswith(old_folder_name + "_"):
        suffix = base_name[len(old_folder_name):]
        new_base_name = new_folder_name + suffix
    else:
        return None

    return new_base_name + ext


def plan_file_renames(folder_path, old_folder_name, new_folder_name):
    file_plans = []

    for file_name in os.listdir(folder_path):
        old_file_path = os.path.join(folder_path, file_name)

        if not os.path.isfile(old_file_path):
            continue

        new_file_name = build_new_file_name(file_name, old_folder_name, new_folder_name)

        if new_file_name is None or new_file_name == file_name:
            continue

        file_plans.append({
            "old_name": file_name,
            "new_name": new_file_name,
            "status": "planned"
        })

    return file_plans


def apply_renames(root_path, rename_plan, dry_run=False):
    operations = []

    renamed_folders = 0
    renamed_files = 0
    skipped_folders = 0
    skipped_files = 0

    for item in rename_plan:
        old_name = item["old_name"]
        final_name = item["final_name"]

        old_path = os.path.join(root_path, old_name)
        new_path = os.path.join(root_path, final_name)

        folder_record = {
            "old_folder_name": old_name,
            "new_folder_name": final_name,
            "folder_status": None,
            "file_operations": []
        }

        if not os.path.exists(old_path):
            folder_record["folder_status"] = "missing_source"
            operations.append(folder_record)
            skipped_folders += 1
            continue

        if not dry_run and os.path.exists(new_path):
            folder_record["folder_status"] = "target_exists"
            operations.append(folder_record)
            skipped_folders += 1
            continue

        if dry_run:
            folder_record["folder_status"] = "planned"
            folder_record["file_operations"] = plan_file_renames(
                old_path,
                old_name,
                final_name
            )
            renamed_folders += 1
            renamed_files += len(folder_record["file_operations"])
            operations.append(folder_record)
            continue

        os.rename(old_path, new_path)
        folder_record["folder_status"] = "renamed"
        renamed_folders += 1

        for file_name in os.listdir(new_path):
            current_file_path = os.path.join(new_path, file_name)

            if not os.path.isfile(current_file_path):
                continue

            new_file_name = build_new_file_name(file_name, old_name, final_name)

            if new_file_name is None or new_file_name == file_name:
                continue

            new_file_path = os.path.join(new_path, new_file_name)

            file_record = {
                "old_name": file_name,
                "new_name": new_file_name,
                "status": None
            }

            if os.path.exists(new_file_path):
                file_record["status"] = "target_exists"
                skipped_files += 1
            else:
                os.rename(current_file_path, new_file_path)
                file_record["status"] = "renamed"
                renamed_files += 1

            folder_record["file_operations"].append(file_record)

        operations.append(folder_record)

    report = {
        "root_path": root_path,
        "mode": "dry_run" if dry_run else "live",
        "planned_folder_renames": len(rename_plan),
        "renamed_folders": renamed_folders,
        "skipped_folders": skipped_folders,
        "renamed_files": renamed_files,
        "skipped_files": skipped_files,
        "operations": operations
    }

    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename sibling subfolders by removing '.' and '-', replacing spaces with '_', resolving conflicts with numeric suffixes, and renaming matching files inside each folder."
    )
    parser.add_argument(
        "root_path",
        help="Path to the parent folder containing subfolders to rename"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview rename actions without changing anything"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_path = args.root_path
    dry_run = args.dry_run

    if not os.path.exists(root_path) or not os.path.isdir(root_path):
        print(json.dumps({
            "root_path": root_path,
            "error": "Invalid folder"
        }, indent=2, ensure_ascii=False))
        return

    rename_plan = plan_renames(root_path)
    report = apply_renames(root_path, rename_plan, dry_run=dry_run)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()