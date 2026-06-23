import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from convert_smplx_motion_to_mixamo_body import (  # noqa: E402
    bake_frontend_motion,
    convert_motion,
    load_json,
    load_model_rest_rotations,
)
from vocab_preprocess.batch_axisangle_npz_to_quaternion_json import (  # noqa: E402
    convert_npz_to_output_json,
    load_relaxed_hand_refs,
    load_vocab_list,
    write_jsonl_line,
)


DEFAULT_INPUT_DIR = Path("/home/ywen/Desktop/smpl_data")
DEFAULT_OUTPUT_DIR = Path("/home/ywen/Desktop/smpl_data_threejs")
DEFAULT_VOCABS_JSON = REPO_ROOT / "vocabs" / "all_vocabs.json"
DEFAULT_HANDPOSES_NPZ = REPO_ROOT / "smplestx_npz_extract" / "smplx_handposes.npz"
DEFAULT_RETARGET_OFFSETS = REPO_ROOT / "vocabs" / "smplx_to_mixamo_body_rotation_offsets.json"
DEFAULT_MODEL_GLB = REPO_ROOT / "models" / "woman.glb"


def write_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def load_completed_vocab_statuses(processed_jsonl: Path) -> dict[str, dict]:
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


def convert_npz_to_threejs_json(npz_path: Path, hand_refs, offsets_data, rest_rotations) -> dict:
    smplx_motion = convert_npz_to_output_json(npz_path, hand_refs)
    mixamo_motion, skipped_bones, skipped_entries, converted_entries, converted_by_group = (
        convert_motion(smplx_motion, offsets_data)
    )
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


def process_vocab(vocab: str, input_dir: Path, output_dir: Path, hand_refs, offsets_data, rest_rotations):
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
    skipped_bones = set()

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-convert SMPL-X axis-angle NPZ files to Three.js frontend motion JSON."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocabs-json", type=Path, default=DEFAULT_VOCABS_JSON)
    parser.add_argument("--handposes-npz", type=Path, default=DEFAULT_HANDPOSES_NPZ)
    parser.add_argument("--retarget-offsets", type=Path, default=DEFAULT_RETARGET_OFFSETS)
    parser.add_argument("--model-glb", type=Path, default=DEFAULT_MODEL_GLB)
    parser.add_argument("--processed-jsonl", type=Path, default=None)
    parser.add_argument("--missing-jsonl", type=Path, default=None)
    parser.add_argument(
        "--append-logs",
        action="store_true",
        help="Append to existing JSONL logs instead of replacing them at startup.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Read processed_vocabs.jsonl and skip vocabs already marked done, empty, or missing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    vocabs_json = args.vocabs_json.resolve()
    handposes_npz = args.handposes_npz.resolve()
    retarget_offsets = args.retarget_offsets.resolve()
    model_glb = args.model_glb.resolve()

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

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_jsonl.parent.mkdir(parents=True, exist_ok=True)
    missing_jsonl.parent.mkdir(parents=True, exist_ok=True)

    vocabs = load_vocab_list(vocabs_json)
    hand_refs = load_relaxed_hand_refs(handposes_npz)
    offsets_data = load_json(retarget_offsets)
    rest_rotations = load_model_rest_rotations(model_glb)
    completed_vocab_statuses = (
        load_completed_vocab_statuses(processed_jsonl)
        if args.resume
        else {}
    )

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
        print(f"Resume mode: skipping {len(completed_vocab_statuses)} previously completed vocabs")

    with open(processed_jsonl, log_mode, encoding="utf-8") as processed_log, open(
        missing_jsonl,
        log_mode,
        encoding="utf-8",
    ) as missing_log:
        for index, vocab in enumerate(vocabs, start=1):
            if vocab in completed_vocab_statuses:
                total_skipped += 1
                if total_skipped == 1 or total_skipped % 500 == 0:
                    print(f"[{index}/{len(vocabs)}] {vocab}: skipped from processed log")
                continue

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
                f"{result['status']}, converted {result['converted_count']}/{result['npz_count']}"
            )

    print("Done.")
    print(f"NPZ files seen: {total_npz}")
    print(f"Frontend JSON files converted: {total_converted}")
    print(f"Missing or empty vocab folders: {total_missing_or_empty}")
    print(f"Skipped from processed log: {total_skipped}")


if __name__ == "__main__":
    main()
