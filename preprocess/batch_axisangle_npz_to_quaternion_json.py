"""
This module provides batch converting smplx axis-angle rotation in .npz 
to quaternion rotation that is compatible with smplx blender official plugin.

The converted rotations are stored by:
{bone_name: {frame: int, rotation: List[float]}}.

We use Mixamo-based rig in the frontend,
to convert quaternion result from this module to mixamo- and frontend- compatible
quaternion rotation,
please refer to convert_smplx_motion_to_mixamo_body.py for single file conversion,
or batch_axisangle_npz_to_threejs_json.py for batch conversion.


Axis-angle rotation to smpl-blender quaternion follows the math formulation 
described in: https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon
"""


import argparse
import json
import math
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Any, TextIO


# file paths acquired for batch processing
DEFAULT_INPUT_DIR: Path = Path("/home/ywen/Desktop/smpl_data")
DEFAULT_OUTPUT_DIR: Path = Path("/home/ywen/Desktop/smpl_data_quaternion")
DEFAULT_VOCABS_JSON: Path = Path(
    "/home/ywen/Desktop/sign_avatar_js/vocabs/all_vocabs.json"
)
DEFAULT_HANDPOSES_NPZ: Path = Path(
    "/home/ywen/Desktop/sign_avatar_js/smplestx_npz_extract/smplx_handposes.npz"
)

# smpl joint name
BODY_JOINT_NAMES_BLENDER: List[str] = [
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

LEFT_HAND_JOINT_NAMES_BLENDER: List[str] = [
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
]

RIGHT_HAND_JOINT_NAMES_BLENDER: List[str] = [
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]


def rodrigues_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    """
    Convert inferred rodrigues axis-angle to quaternion for bone rotations.

    rotvec  (np.ndarray)  :  rodrigues axis-angle to be converted

    Return:
    (np.ndarray)          :  quaternion rotation 
    """
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    axis = rotvec / theta  # unit rotation axis
    half = theta * 0.5
    s = np.sin(half)
    w = np.cos(half)

    return np.array(
        [axis[0] * s, axis[1] * s, axis[2] * s, w],
        dtype=np.float32,
    )


def quat_xyzw_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Combine two quaternion rotations using the Hamilton product.

    Here, it is mainly used to apply the fixed pelvis-coordinate correction
    before the original SMPL-X pelvis rotation.

    q1  (np.ndarray)  :  first quaternion, with shape (4,) and XYZW component ordering.
                         In the pelvis conversion, this is the fixed correction rotation
    q2  (np.ndarray)  :  second quaternion, with shape (4,) and XYZW component ordering.
                         In the pelvis conversion, this is the original SMPL-X rotation

    Return:
    (np.ndarray)  :  combined quaternion with shape (4,), using XYZW component ordering
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    """ 
    Normalize a quaternion to unit length while preserving XYZW ordering. 

    If the quaternion magnitude is too close to zero, return the identity quaternion to avoid 
    division by zero and invalid rotation values. 
    
    q (np.ndarray)  :  quaternion with shape (4,) and XYZW component ordering 
    
    Return: 
    (np.ndarray)    :  normalized quaternion with shape (4,), using XYZW component ordering 
    """
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def axis_angle_to_quat_xyzw(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Convert a separately defined rotation axis and angle to a quaternion. 
    
    Here, it is used to create one fixed quaternion representing the 180-degree X-axis correction 
    applied to every pelvis rotation. The returned quaternion describes how the pelvis should 
    be corrected.

    axis       (np.ndarray)  :  rotation axis with shape (3,) 
    angle_rad  (float)       :  rotation angle in radians 
    
    Return: 
    (np.ndarray)             : quaternion with shape (4,), using XYZW component ordering
    """
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    half = angle_rad * 0.5
    s = np.sin(half)
    w = np.cos(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, w], dtype=np.float32)


def convert_pose_block_to_quat(rotvec_block: np.ndarray) -> np.ndarray:
    """ 
    Convert a block of rodrigues axis-angle rotations to XYZW quaternions. 
    
    The input may contain all joint rotations for one frame or for multiple frames. 
    Each joint rotation is converted independently using rodrigues_to_quat_xyzw. 
    
    rotvec_block  (np.ndarray)  :  axis-angle rotations with shape (J, 3) for one frame, 
                                   or (F, J, 3) for multiple frames 
    
    Return: 
    (np.ndarray)                :  quaternion rotations with shape (J, 4) for one frame, 
                                   or (F, J, 4) for multiple frames, using XYZW component 
                                   ordering
    """
    rotvec_block = np.asarray(rotvec_block, dtype=np.float32)

    if rotvec_block.ndim == 2:
        return np.stack([rodrigues_to_quat_xyzw(rv) for rv in rotvec_block], axis=0)

    if rotvec_block.ndim == 3:
        return np.stack(
            [
                np.stack([rodrigues_to_quat_xyzw(rv) for rv in frame], axis=0)
                for frame in rotvec_block
            ],
            axis=0,
        )

    raise ValueError(f"Expected rotvec_block ndim 2 or 3, got shape {rotvec_block.shape}")


def convert_hand_pose_with_reference(
    hand_pose: np.ndarray, hand_ref_flat: np.ndarray
) -> np.ndarray:
    """
    Axis-angle to quaternion conversion for hands.

    SMPL-X is adding the reference rodrigues rotation to the relaxed hand rodrigues rotation, 
    so we have to do the same here.
    
    This means that pose values for relaxed hand model cannot be interpreted as rotations in the local 
    joint coordinate system of the relaxed hand.
    Reference:
    https://github.com/vchoutas/smplx/blob/f4206853a4746139f61bdcf58571f2cea0cbebad/smplx/body_models.py
    full_pose += self.pose_mean

    hand_pose      (np.ndarray)  :  hand pose axis-angle values with shape (45,), (15, 3), (F, 45), or
                                    (F, 15, 3) 
    hand_ref_flat  (np.ndarray)  :  relaxed-hand reference axis-angle values with shape (45,), 
                                    loaded from ./smpl_handposes.npz 
    
    Return: 
    (np.ndarray)                 :  hand joint quaternions with shape (15, 4) for one frame, or 
                                    (F, 15, 4) for multiple frames, using XYZW ordering
    """
    hand_pose = np.asarray(hand_pose, dtype=np.float32)
    hand_ref = np.asarray(hand_ref_flat, dtype=np.float32).reshape(15, 3)

    if hand_pose.ndim == 1:
        hand_pose = hand_pose.reshape(15, 3)
        hand_pose_final = hand_pose + hand_ref
        return convert_pose_block_to_quat(hand_pose_final)

    if hand_pose.ndim == 2:
        if hand_pose.shape[-1] == 45:
            hand_pose = hand_pose.reshape(hand_pose.shape[0], 15, 3)
        elif hand_pose.shape == (15, 3):
            hand_pose_final = hand_pose + hand_ref
            return convert_pose_block_to_quat(hand_pose_final)
        else:
            raise ValueError(f"Unexpected 2D hand pose shape: {hand_pose.shape}")

        hand_pose_final = hand_pose + hand_ref[None, :, :]
        return convert_pose_block_to_quat(hand_pose_final)

    if hand_pose.ndim == 3:
        hand_pose_final = hand_pose + hand_ref[None, :, :]
        return convert_pose_block_to_quat(hand_pose_final)

    raise ValueError(f"Unexpected hand pose shape: {hand_pose.shape}")


def ensure_frames_joints3(arr: np.ndarray, num_joints: int, name: str) -> np.ndarray:
    """
    Normalize a pose array to shape (F, J, 3). 
    
    This allows the rest of the module to process single-frame and multiple-frame pose data using 
    one consistent array shape. 
    
    Supported input shapes: 
        (J * 3,) -> (1, J, 3)
        (J, 3) -> (1, J, 3)
        (F, J * 3) -> (F, J, 3)
        (F, J, 3) -> (F, J, 3) 
        
    arr         (np.ndarray)  :  pose array containing axis-angle rotations 
    num_joints  (int)         :  expected number of joints in the pose array 
    name        (str)         :  array name used in validation error messages 
    
    Return: 
    (np.ndarray)              :  pose array with shape (F, J, 3)
    """
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 1:
        if arr.shape[0] != num_joints * 3:
            raise ValueError(f"{name}: expected {(num_joints * 3,)} got {arr.shape}")
        return arr.reshape(1, num_joints, 3)

    if arr.ndim == 2:
        if arr.shape == (num_joints, 3):
            return arr.reshape(1, num_joints, 3)
        if arr.shape[1] == num_joints * 3:
            return arr.reshape(arr.shape[0], num_joints, 3)

    if arr.ndim == 3 and arr.shape[1:] == (num_joints, 3):
        return arr

    raise ValueError(f"{name}: unsupported shape {arr.shape}")


def load_relaxed_hand_refs(handposes_npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read in relaxed-hand reference pose constants from local npz file.

    handposes_npz_path  (Path)     :  path to the relaxed-hand reference pose npz file

    Return:
    Tuple[np.ndarray, np.ndarray]  :  left hand relaxed reference pose and right hand 
                                      relaxed reference pose
    """
    handpose_data = np.load(handposes_npz_path, allow_pickle=True)
    hand_poses = handpose_data["hand_poses"].item()
    left_hand_relaxed_ref, right_hand_relaxed_ref = hand_poses["relaxed"]
    return left_hand_relaxed_ref.astype(np.float32), right_hand_relaxed_ref.astype(np.float32)


def convert_npz_to_output_json(
    npz_path: Path, hand_refs: Tuple[np.ndarray, np.ndarray]
) -> Dict[str, List[Any]]:
    """
    Convert a single npz file's axis-angle rotation to blender smplx compatible quaternion
    rotation.

    npz_path   (Path)                           :  npz file path
    hand_refs  (Tuple[np.ndarray, np.ndarray])  :  hand relaxed pose reference, (left, right)

    Return:
    (Dict[str, List[Any]])                      :  converted quaternion rotation stored as:  
                                                   {
                                                        bone_name: [
                                                            "frame": int,
                                                            "rotation: List[float]
                                                        ]
                                                   }
    """
    # decouple hand relaxed poses for left and right hand
    left_hand_relaxed_ref, right_hand_relaxed_ref = hand_refs
    # load source npz axis-angle rotation to be converted
    pose_data = np.load(npz_path)


    # check validity of loaded axis-angle rotations
    global_orient = ensure_frames_joints3(pose_data["global_orient"], 1, "global_orient")
    body_pose = ensure_frames_joints3(pose_data["body_pose"], 21, "body_pose")
    jaw_pose = ensure_frames_joints3(pose_data["jaw_pose"], 1, "jaw_pose")

    left_hand_pose = pose_data["left_hand_pose"].astype(np.float32)
    right_hand_pose = pose_data["right_hand_pose"].astype(np.float32)

    num_frames = global_orient.shape[0]

    if body_pose.shape[0] != num_frames:
        raise ValueError(f"body_pose frame count mismatch: {body_pose.shape[0]} vs {num_frames}")
    if jaw_pose.shape[0] != num_frames:
        raise ValueError(f"jaw_pose frame count mismatch: {jaw_pose.shape[0]} vs {num_frames}")


    # convert hand axis-angle to quaternion with relaxed hand poses
    left_hand_quat = convert_hand_pose_with_reference(left_hand_pose, left_hand_relaxed_ref)
    right_hand_quat = convert_hand_pose_with_reference(right_hand_pose, right_hand_relaxed_ref)

    # add a frame dimension for single-frame hand poses so all hand quaternion
    # arrays consistently use shape (F, 15, 4)
    if left_hand_quat.ndim == 2:
        left_hand_quat = left_hand_quat[None, :, :]
    if right_hand_quat.ndim == 2:
        right_hand_quat = right_hand_quat[None, :, :]

    if left_hand_quat.shape[0] != num_frames:
        raise ValueError(
            f"left_hand_pose frame count mismatch: {left_hand_quat.shape[0]} vs {num_frames}"
        )
    if right_hand_quat.shape[0] != num_frames:
        raise ValueError(
            f"right_hand_pose frame count mismatch: {right_hand_quat.shape[0]} vs {num_frames}"
        )


    # convert pelvis axis-angle to quaternion with 180-degree on x as correction
    pelvis_quat = convert_pose_block_to_quat(global_orient)[:, 0, :]  # (F, 4)
    jaw_quat = convert_pose_block_to_quat(jaw_pose)[:, 0, :]          # (F, 4)
    body_quat = convert_pose_block_to_quat(body_pose)                 # (F, 21, 4)

    pelvis_correction = axis_angle_to_quat_xyzw(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        math.pi,
    )

    pelvis_quat_corrected = np.stack(
        [normalize_quat_xyzw(quat_xyzw_multiply(pelvis_correction, q)) for q in pelvis_quat],
        axis=0,
    )


    # all joint conversion
    frame_ids = (
        pose_data["frame_ids"].tolist()
        if "frame_ids" in pose_data.files
        else list(range(num_frames))
    )

    all_joint_rotations = {"pelvis": pelvis_quat_corrected}

    for name, joint_quats in zip(
        BODY_JOINT_NAMES_BLENDER,
        np.transpose(body_quat, (1, 0, 2)),
    ):
        all_joint_rotations[name] = joint_quats

    all_joint_rotations["jaw"] = jaw_quat

    for name, joint_quats in zip(
        LEFT_HAND_JOINT_NAMES_BLENDER,
        np.transpose(left_hand_quat, (1, 0, 2)),
    ):
        all_joint_rotations[name] = joint_quats

    for name, joint_quats in zip(
        RIGHT_HAND_JOINT_NAMES_BLENDER,
        np.transpose(right_hand_quat, (1, 0, 2)),
    ):
        all_joint_rotations[name] = joint_quats

    return {
        joint_name: [
            {
                "frame": int(frame_ids[i]),
                "rotation": joint_quats[i].tolist(),
            }
            for i in range(num_frames)
        ]
        for joint_name, joint_quats in all_joint_rotations.items()
    }


def load_vocab_list(vocabs_json_path: Path) -> List[str]:
    """
    Load all vocabs as a list from metadata file, for easier track when batch run.

    vocabs_json_path  (Path)  :  JSON file path where all vacabs are stored

    Return:
    (List[str])               :  all vocabs in a list
    """
    with open(vocabs_json_path, "r", encoding="utf-8") as f:
        vocabs = json.load(f)

    if not isinstance(vocabs, list):
        raise TypeError(f"Expected list in {vocabs_json_path}, got {type(vocabs).__name__}")

    return [str(vocab) for vocab in vocabs]


def write_jsonl_line(handle: TextIO, payload: dict) -> None:
    """
    Write a dictionary as a single JSON line to a text stream. 
    
    The payload is serialized as JSON, followed by a newline character. The stream is flushed 
    immediately to ensure that the written data is available. 
    
    handle: writable text stream or file-like object
    payload: The dictionary to serialize and write
    """
    handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    handle.flush()


def process_vocab(
    vocab: str, input_dir: Path, output_dir: Path, hand_refs: Tuple[np.ndarray, np.ndarray]
) -> Dict[str, Any]:
    """
    Wrapper function to convert all nps files under a vocab folder one by one.

    vocab       (str)                            :  vocab name
    input_dir   (Path)                           :  npz data root folder path
    output_dir  (Path)                           :  output data root folder path
    hand_refs   (Tuple[np.ndarray, np.ndarray])  :  hand relaxed pose reference, (left, right)

    Return:
    (Dict[Any])                                  :  process results and logs
    """
    # find the vocab folder path under input root
    # and create vocab folder under output root
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

    # process each npz file under vocab folder
    for npz_path in npz_paths:
        output_json_path = output_vocab_dir / f"{npz_path.stem}.json"
        try:
            output_json = convert_npz_to_output_json(npz_path, hand_refs)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(output_json, f, indent=2)
            converted_count += 1
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
        "errors": errors,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-convert SMPL-X axis-angle NPZ files to quaternion JSON files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocabs-json", type=Path, default=DEFAULT_VOCABS_JSON)
    parser.add_argument("--handposes-npz", type=Path, default=DEFAULT_HANDPOSES_NPZ)
    parser.add_argument("--processed-jsonl", type=Path, default=None)
    parser.add_argument("--missing-jsonl", type=Path, default=None)
    parser.add_argument(
        "--append-logs",
        action="store_true",
        help="Append to existing JSONL logs instead of replacing them at startup.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    vocabs_json = args.vocabs_json.resolve()
    handposes_npz = args.handposes_npz.resolve()

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

    log_mode = "a" if args.append_logs else "w"
    total_converted = 0
    total_npz = 0
    total_missing_or_empty = 0

    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Vocab count: {len(vocabs)}")
    print(f"Processed log: {processed_jsonl}")
    print(f"Missing log: {missing_jsonl}")

    with open(processed_jsonl, log_mode, encoding="utf-8") as processed_log, open(
        missing_jsonl,
        log_mode,
        encoding="utf-8",
    ) as missing_log:
        for index, vocab in enumerate(vocabs, start=1):
            result = process_vocab(vocab, input_dir, output_dir, hand_refs)

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
    print(f"JSON files converted: {total_converted}")
    print(f"Missing or empty vocab folders: {total_missing_or_empty}")


if __name__ == "__main__":
    main()
