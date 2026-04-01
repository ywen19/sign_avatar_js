import argparse
from pathlib import Path
import json
import cv2
import subprocess
import shutil
import time


def get_all_subfolder_names(vocab_root):
    vocab_root = Path(vocab_root)
    return sorted([p.name for p in vocab_root.iterdir() if p.is_dir()])


def get_all_files_in_subfolder(vocab_root, subfolder_name):
    subfolder_path = Path(vocab_root) / subfolder_name
    return sorted([p.name for p in subfolder_path.iterdir() if p.is_file()])


def get_video_fps(video_path, default_fps=30.0):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return default_fps

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps is None or fps <= 1e-6:
        return default_fps

    return fps


def convert_video_to_frames(
    vocab_root, subfolder_name, file_name, frame_output_dir, fps=30.0
):
    if fps is None or fps <= 0:
        raise RuntimeError(f"Invalid FPS for {file_name}: {fps}")

    full_file_path = Path(vocab_root) / subfolder_name / file_name
    frame_output_dir = Path(frame_output_dir)

    reset_frame_output_dir(frame_output_dir)
    output_pattern = frame_output_dir / "%06d.jpg"

    cmd = [
        "ffmpeg",
        "-i", str(full_file_path),
        "-f", "image2",
        "-vf", f"fps={fps}/1",
        "-qscale", "0",
        str(output_pattern),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed for {full_file_path}: {e}")

    extracted_frames = sorted(frame_output_dir.glob("*.jpg"))
    if len(extracted_frames) == 0:
        raise RuntimeError(f"No frames extracted for {full_file_path}")


def reset_frame_output_dir(frame_output_dir):
    frame_output_dir = Path(frame_output_dir)
    if frame_output_dir.exists():
        shutil.rmtree(frame_output_dir)
    frame_output_dir.mkdir(parents=True, exist_ok=True)


def process_file(vocab_root, subfolder_name, file_name, frame_output_dir):
    full_file_path = Path(vocab_root) / subfolder_name / file_name
    print(f"  Processing file: {full_file_path}")

    if not full_file_path.exists():
        raise RuntimeError(f"Missing file: {full_file_path}")

    fps = get_video_fps(full_file_path, default_fps=30.0)
    print(f"    FPS: {fps}")

    try:
        convert_video_to_frames(
            vocab_root=vocab_root,
            subfolder_name=subfolder_name,
            file_name=file_name,
            frame_output_dir=frame_output_dir,
            fps=fps,
        )
        print(f"    Frames written to: {frame_output_dir}")
    except Exception:
        reset_frame_output_dir(frame_output_dir)
        raise

    time.sleep(0.2)
    reset_frame_output_dir(frame_output_dir)


def append_progress(progress_file, subfolder_name, processed_files):
    with open(progress_file, "a", encoding="utf-8") as f:
        json.dump({subfolder_name: processed_files}, f)
        f.write("\n")
        f.flush()


def append_missing(missing_file, subfolder_name, missing_files):
    with open(missing_file, "a", encoding="utf-8") as f:
        json.dump({subfolder_name: missing_files}, f)
        f.write("\n")
        f.flush()


def get_resume_info(progress_file):
    progress_path = Path(progress_file)
    if not progress_path.exists():
        return set(), None, None

    records = []
    with open(progress_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        return set(), None, None

    completed_subfolders = set()
    for record in records[:-1]:
        key = next(iter(record))
        completed_subfolders.add(key)

    last_record = records[-1]
    last_subfolder = next(iter(last_record))
    last_processed_files = last_record[last_subfolder]

    return completed_subfolders, last_subfolder, last_processed_files


def process_subfolder(vocab_root, progress_file, missing_file, subfolder_name, frame_output_dir):
    print(f"Processing: {subfolder_name}")
    all_files = get_all_files_in_subfolder(vocab_root, subfolder_name)
    processed_files = []
    missing_files = []

    for file_name in all_files:
        try:
            process_file(vocab_root, subfolder_name, file_name, frame_output_dir)
            processed_files.append(file_name)
        except Exception:
            missing_files.append(file_name)

    append_progress(progress_file, subfolder_name, processed_files)

    if missing_files:
        append_missing(missing_file, subfolder_name, missing_files)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vocab_root", type=str, default="../demo/vocabs"
    )
    parser.add_argument(
        "--progress_file", type=str, default="./progress.jsonl"
    )
    parser.add_argument(
        "--missing_file", type=str, default="./missing.jsonl"
    )
    parser.add_argument(
    "--frame_output_dir", type=str, default="/media/ywen/DATA800/tmp_frames"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_root = Path(args.frame_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_vocabs = get_all_subfolder_names(args.vocab_root)
    print(f"We have in total {len(all_vocabs)} vocab folders to be prcossed \n")
    print("printing first 20 as examples: \n")
    print(all_vocabs[:20])

    completed_subfolders, last_subfolder, last_processed_files = get_resume_info(args.progress_file)

    for subfolder_name in all_vocabs[:3]:
        if subfolder_name in completed_subfolders:
            print(f"Skipping completed subfolder: {subfolder_name}")
            continue

        if subfolder_name == last_subfolder:
            all_files = get_all_files_in_subfolder(args.vocab_root, subfolder_name)
            if all_files == last_processed_files:
                print(f"Skipping completed last subfolder: {subfolder_name}")
                continue

        process_subfolder(
            args.vocab_root,
            args.progress_file,
            args.missing_file,
            subfolder_name,
            output_root,
        )


if __name__ == "__main__":
    main()