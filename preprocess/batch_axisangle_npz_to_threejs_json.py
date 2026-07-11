"""
This module provides batch converting SMPL-X axis-angle rotations in NPZ files to
Three.js-compatible quaternion animation JSON files using a Mixamo bone structure.

Each NPZ file is first converted to SMPL-X quaternion motion, retargeted to Mixamo
bones using stored rotation offsets, and combined with rest-pose rotations loaded
from a GLB model. The final animation is stored using the frontend structure:
{"name": str, "fps": int, "bones": {bone_name: [{"f": int, "rot": List[float]}]}}.

Vocabulary folders can be processed from a metadata list. Processing results, 
missing folders, conversion errors, and resume information are recorded in JSONL 
logs.

Refer to convert_smplx_motion_to_mixamo_body.py for the single-file conversion.
"""


import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Set, Tuple
import numpy as np


# add the repository root when this module is executed directly as a script
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess.convert_smplx_motion_to_mixamo_body import (
    bake_frontend_motion,
    convert_motion,
    load_json,
    load_model_rest_rotations,
)
from preprocess.batch_axisangle_npz_to_quaternion_json import (
    convert_npz_to_output_json,
    load_relaxed_hand_refs,
    load_vocab_list,
    write_jsonl_line,
)


# file paths acquired for batch processing
DEFAULT_INPUT_DIR: Path = Path("/home/ywen/Desktop/smpl_data")
DEFAULT_OUTPUT_DIR: Path = Path("/home/ywen/Desktop/smpl_data_threejs")
DEFAULT_VOCABS_JSON: Path = REPO_ROOT / "vocabs" / "all_vocabs.json"
DEFAULT_HANDPOSES_NPZ: Path = (
    REPO_ROOT / "smplestx_npz_extract" / "smplx_handposes.npz"
)
DEFAULT_RETARGET_OFFSETS: Path = (
    REPO_ROOT / "vocabs" / "smplx_to_mixamo_body_rotation_offsets.json"
)
DEFAULT_MODEL_GLB: Path = REPO_ROOT / "models" / "woman.glb"


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Write a dictionary to a formatted JSON file.

    path  (Path)            :  output JSON file path
    data  (Dict[str, Any])  :  data to serialize
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def load_completed_vocab_statuses(processed_jsonl: Path) -> Dict[str, Any]:
    """
    Load processing records from a JSONL log.

    A record is considered completed when it contains a vocabulary value,
    has a status of "done", "empty", or "missing", and contains no errors.

    processed_jsonl  (Path)  :  JSON log file path

    Return:
    (Dict[str, Any])         :  completed processing record
    """
    if not processed_jsonl.exists():
        return {}

    completed_statuses = {"done", "empty", "missing"}
    completed = {}

    with open(processed_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            vocab = record.get("vocab")
            status = record.get("status")
            errors = record.get("errors") or []

            if vocab and status in completed_statuses and not errors:
                completed[str(vocab)] = record

    return completed


def convert_npz_to_threejs_json(
    npz_path: Path,
    hand_refs: Tuple[np.ndarray, np.ndarray],
    offsets_data: Dict[str, Any],
    rest_rotations: Dict[str, Tuple[float, ...]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convert one SMPL-X axis-angle NPZ file to frontend animation data.

    The source rotations are converted to SMPL-X quaternions, retargeted 
    to Mixamo bones, and baked with model rest-pose rotations for direct 
    Three.js playback.

    npz_path        (Path)                           :  SMPL-X NPZ file path
    hand_refs       (Tuple[np.ndarray, np.ndarray])  :  relaxed-hand poses
    offsets_data    (Dict[str, Any])                 :  SMPL-X-to-Mixamo offsets
    rest_rotations  (Dict[str, Tuple[float, ...]])   :  model rest rotations

    Return:
    (Tuple[Dict[str, Any], Dict[str, Any]])          :  frontend animation and 
                                                        conversion statistics
    """
    # convert NPZ axis-angle rotations to SMPL-X XYZW quaternion keyframes
    smplx_motion = convert_npz_to_output_json(npz_path, hand_refs)
    # retarget SMPL-X quaternion keyframes to their corresponding Mixamo bones
    (
        mixamo_motion,
        skipped_bones,
        skipped_entries,
        converted_entries,
        converted_by_group,
    ) = convert_motion(
        smplx_motion, offsets_data
    )
    # combine Mixamo motion with the GLB rest pose for frontend playback
    frontend_motion = bake_frontend_motion(
        mixamo_motion,
        rest_rotations,
        animation_name=npz_path.stem,
    )

    return frontend_motion, {
        "skipped_bones": skipped_bones,
        "skipped_entries": skipped_entries,
        "converted_entries": converted_entries,
        "converted_by_group": converted_by_group,
    }


def process_vocab(
    vocab: str,
    input_dir: Path,
    output_dir: Path,
    hand_refs: Tuple[np.ndarray, np.ndarray],
    offsets_data: Dict[str, Any],
    rest_rotations: Dict[str, Tuple[float, ...]],
) -> Dict[str, Any]:
    """
    Convert every NPZ file contained in one vocabulary folder.

    Missing and empty input folders are returned as completed status records. 
    Individual file failures are collected.

    vocab           (str)                            :  vocabulary folder name
    input_dir       (Path)                           :  root directory for NPZ 
    output_dir      (Path)                           :  root directory for output
    hand_refs       (Tuple[np.ndarray, np.ndarray])  :  relaxed-hand poses
    offsets_data    (Dict[str, Any])                 :  SMPL-X-to-Mixamo offsets
    rest_rotations  (Dict[str, Tuple[float, ...]])   :  model rest rotations

    Return:
    (Dict[str, Any])                                 :  conversion status 
    """
    # resolve the matching vocabulary folders under the input and output roots
    input_vocab_dir = input_dir / vocab
    output_vocab_dir = output_dir / vocab
    output_vocab_dir.mkdir(parents=True, exist_ok=True)

    if not input_vocab_dir.is_dir():
        return {
            "vocab": vocab,
            "status": "missing",
            "input_dir": str(input_vocab_dir),
            "npz_count": 0,
            "converted_count": 0,
            "errors": [],
        }

    npz_paths = sorted(input_vocab_dir.glob("*.npz"))
    if not npz_paths:
        return {
            "vocab": vocab,
            "status": "empty",
            "input_dir": str(input_vocab_dir),
            "output_dir": str(output_vocab_dir),
            "npz_count": 0,
            "converted_count": 0,
            "errors": [],
        }

    errors = []
    converted_count = 0
    converted_entries = 0
    skipped_entries = 0
    skipped_bones: Set[str] = set()

    # convert each NPZ file independently
    for npz_path in npz_paths:
        output_json_path = output_vocab_dir / f"{npz_path.stem}.json"
        try:
            frontend_motion, stats = convert_npz_to_threejs_json(
                npz_path,
                hand_refs,
                offsets_data,
                rest_rotations,
            )
            write_json(output_json_path, frontend_motion)
            converted_count += 1
            converted_entries += stats["converted_entries"]
            skipped_entries += stats["skipped_entries"]
            skipped_bones.update(stats["skipped_bones"])
        except Exception as exc:
            errors.append(
                {
                    "npz": str(npz_path),
                    "output": str(output_json_path),
                    "error": str(exc),
                }
            )

    return {
        "vocab": vocab,
        "status": "done" if not errors else "done_with_errors",
        "input_dir": str(input_vocab_dir),
        "output_dir": str(output_vocab_dir),
        "npz_count": len(npz_paths),
        "converted_count": converted_count,
        "converted_entries": converted_entries,
        "skipped_entries": skipped_entries,
        "skipped_bones": sorted(skipped_bones),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-convert SMPL-X axis-angle NPZ files to "
        "Three.js frontend motion JSON."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR
        )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
        )
    parser.add_argument(
        "--vocabs-json", type=Path, default=DEFAULT_VOCABS_JSON
        )
    parser.add_argument(
        "--handposes-npz", type=Path, default=DEFAULT_HANDPOSES_NPZ
        )
    parser.add_argument(
        "--retarget-offsets", type=Path, default=DEFAULT_RETARGET_OFFSETS
        )
    parser.add_argument(
        "--model-glb", type=Path, default=DEFAULT_MODEL_GLB
        )
    parser.add_argument(
        "--processed-jsonl", type=Path, default=None
        )
    parser.add_argument(
        "--missing-jsonl", type=Path, default=None
        )
    parser.add_argument(
        "--append-logs",
        action="store_true",
        help="Append to existing JSONL logs instead of "
        "replacing them at startup.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Read processed_vocabs.jsonl and skip vocabs already "
        "marked done, empty, or missing.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run batch NPZ conversion for every vocabulary in the metadata list.

    Shared hand references, retarget offsets, and GLB rest rotations are 
    loaded once. 
    """
    args = parse_args()

    # resolve all input and model resource paths before processing
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    vocabs_json = args.vocabs_json.resolve()
    handposes_npz = args.handposes_npz.resolve()
    retarget_offsets = args.retarget_offsets.resolve()
    model_glb = args.model_glb.resolve()

    # use output-directory log paths unless custom paths were provided
    processed_jsonl = (
        args.processed_jsonl.resolve()
        if args.processed_jsonl
        else output_dir / "processed_vocabs.jsonl"
    )
    missing_jsonl = (
        args.missing_jsonl.resolve()
        if args.missing_jsonl
        else output_dir / "missing_vocabs.jsonl"
    )

    # create output and log directories before opening any files
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_jsonl.parent.mkdir(parents=True, exist_ok=True)
    missing_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # load metadata and conversion resources shared by every NPZ file
    vocabs = load_vocab_list(vocabs_json)
    hand_refs = load_relaxed_hand_refs(handposes_npz)
    offsets_data = load_json(retarget_offsets)
    rest_rotations = load_model_rest_rotations(model_glb)
    # load completed records only when resume mode is enabled
    completed_vocab_statuses = (
        load_completed_vocab_statuses(processed_jsonl)
        if args.resume
        else {}
    )

    # preserve existing logs for append or resume mode and replace them otherwise
    log_mode = "a" if args.append_logs or args.resume else "w"
    total_npz = 0
    total_converted = 0
    total_missing_or_empty = 0
    total_skipped = 0

    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Vocab count: {len(vocabs)}")
    print(f"Retarget offsets: {retarget_offsets}")
    print(f"Model GLB rest pose: {model_glb}")
    print(f"Processed log: {processed_jsonl}")
    print(f"Missing log: {missing_jsonl}")
    if args.resume:
        print(
            f"Resume mode: skipping {len(completed_vocab_statuses)} "
            "previously completed vocabs"
        )

    # keep both JSONL logs open while vocabulary results are written incrementally
    with open(processed_jsonl, log_mode, encoding="utf-8") as processed_log, open(
        missing_jsonl,
        log_mode,
        encoding="utf-8",
    ) as missing_log:
        for index, vocab in enumerate(vocabs, start=1):
            # skip successfully completed vocabularies when resuming an earlier run
            if vocab in completed_vocab_statuses:
                total_skipped += 1
                if total_skipped == 1 or total_skipped % 500 == 0:
                    print(
                        f"[{index}/{len(vocabs)}] {vocab}: skipped from processed log"
                    )
                continue

            # convert the current vocabulary folder and collect its status record
            result = process_vocab(
                vocab,
                input_dir,
                output_dir,
                hand_refs,
                offsets_data,
                rest_rotations,
            )

            total_npz += result["npz_count"]
            total_converted += result["converted_count"]

            # record every processed vocabulary immediately for progress recovery
            write_jsonl_line(
                processed_log,
                {
                    "index": index,
                    "total": len(vocabs),
                    **result,
                },
            )

            if result["status"] in {"missing", "empty"}:
                total_missing_or_empty += 1
                # duplicate missing and empty records into the dedicated lookup log
                write_jsonl_line(
                    missing_log,
                    {
                        "index": index,
                        "total": len(vocabs),
                        **result,
                    },
                )

            print(
                f"[{index}/{len(vocabs)}] {vocab}: "
                f"{result['status']}, "
                "converted {result['converted_count']}/{result['npz_count']}"
            )

    print("Done.")
    print(f"NPZ files seen: {total_npz}")
    print(f"Frontend JSON files converted: {total_converted}")
    print(f"Missing or empty vocab folders: {total_missing_or_empty}")
    print(f"Skipped from processed log: {total_skipped}")


if __name__ == "__main__":
    main()
