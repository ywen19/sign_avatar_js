import os
import argparse

def cleanup_empty_folders(root_path, dry_run=False):
    if not os.path.exists(root_path) or not os.path.isdir(root_path):
        print("Invalid folder:", root_path)
        return

    count = 0

    for current_path, dirnames, filenames in os.walk(root_path, topdown=False):
        try:
            if not os.listdir(current_path):
                if dry_run:
                    print("[DRY RUN] Would remove:", current_path)
                else:
                    os.rmdir(current_path)
                    print("Removed:", current_path)
                count += 1
        except Exception as e:
            print("Could not process {}: {}".format(current_path, e))

    action = "Would remove" if dry_run else "Removed"
    print("\n{} {} empty folders.".format(action, count))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove empty folders recursively."
    )
    parser.add_argument(
        "root_path",
        help="Path to the parent folder"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only; do not delete anything"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cleanup_empty_folders(args.root_path, dry_run=args.dry_run)