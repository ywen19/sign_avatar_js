import json
import math
import numpy as np


def rodrigues_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    axis = rotvec / theta
    half = theta * 0.5
    s = np.sin(half)
    w = np.cos(half)

    return np.array([
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
        w
    ], dtype=np.float32)


def quat_xyzw_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float32)


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def axis_angle_to_quat_xyzw(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    half = angle_rad * 0.5
    s = np.sin(half)
    w = np.cos(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, w], dtype=np.float32)


def convert_pose_block_to_quat(rotvec_block: np.ndarray) -> np.ndarray:
    """
    rotvec_block:
      - (J, 3) for one frame
      - (F, J, 3) for multiple frames

    returns:
      - (J, 4) or (F, J, 4)
    """
    rotvec_block = np.asarray(rotvec_block, dtype=np.float32)

    if rotvec_block.ndim == 2:
        return np.stack([rodrigues_to_quat_xyzw(rv) for rv in rotvec_block], axis=0)

    if rotvec_block.ndim == 3:
        return np.stack(
            [np.stack([rodrigues_to_quat_xyzw(rv) for rv in frame], axis=0) for frame in rotvec_block],
            axis=0
        )

    raise ValueError(f"Expected rotvec_block ndim 2 or 3, got shape {rotvec_block.shape}")


def convert_hand_pose_with_reference(hand_pose: np.ndarray, hand_ref_flat: np.ndarray) -> np.ndarray:
    """
    hand_pose:
      - old format: (45,) or (15, 3)
      - new format: (F, 45) or (F, 15, 3)

    hand_ref_flat:
      - (45,) relaxed hand reference
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
    Normalize pose arrays into shape (F, J, 3).

    Accepts:
      - (J*3,)          -> (1, J, 3)
      - (J, 3)          -> (1, J, 3)
      - (F, J*3)        -> (F, J, 3)
      - (F, J, 3)       -> (F, J, 3)
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


POSE_NPZ_PATH = "./demo/output_smplx/queue/queue_person00_smplx_pose.npz"
HANDPOSES_NPZ_PATH = "./smplx_handposes.npz"
OUTPUT_JSON_PATH = "./queue_person00_smplx_quat_blender_names.json"

body_joint_names_blender = [
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

left_hand_joint_names_blender = [
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

right_hand_joint_names_blender = [
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

pose_data = np.load(POSE_NPZ_PATH)

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

handpose_data = np.load(HANDPOSES_NPZ_PATH, allow_pickle=True)
hand_poses = handpose_data["hand_poses"].item()
left_hand_relaxed_ref, right_hand_relaxed_ref = hand_poses["relaxed"]

left_hand_relaxed_ref = left_hand_relaxed_ref.astype(np.float32)
right_hand_relaxed_ref = right_hand_relaxed_ref.astype(np.float32)

left_hand_quat = convert_hand_pose_with_reference(left_hand_pose, left_hand_relaxed_ref)
right_hand_quat = convert_hand_pose_with_reference(right_hand_pose, right_hand_relaxed_ref)

if left_hand_quat.ndim == 2:
    left_hand_quat = left_hand_quat[None, :, :]
if right_hand_quat.ndim == 2:
    right_hand_quat = right_hand_quat[None, :, :]

if left_hand_quat.shape[0] != num_frames:
    raise ValueError(f"left_hand_pose frame count mismatch: {left_hand_quat.shape[0]} vs {num_frames}")
if right_hand_quat.shape[0] != num_frames:
    raise ValueError(f"right_hand_pose frame count mismatch: {right_hand_quat.shape[0]} vs {num_frames}")

pelvis_quat = convert_pose_block_to_quat(global_orient)[:, 0, :]   # (F, 4)
jaw_quat = convert_pose_block_to_quat(jaw_pose)[:, 0, :]           # (F, 4)
body_quat = convert_pose_block_to_quat(body_pose)                  # (F, 21, 4)

pelvis_correction = axis_angle_to_quat_xyzw(
    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    math.pi
)

pelvis_quat_corrected = np.stack([
    normalize_quat_xyzw(quat_xyzw_multiply(pelvis_correction, q))
    for q in pelvis_quat
], axis=0)

frame_ids = pose_data["frame_ids"].tolist() if "frame_ids" in pose_data.files else list(range(num_frames))

all_joint_rotations = {}

all_joint_rotations["pelvis"] = pelvis_quat_corrected
for name, joint_quats in zip(body_joint_names_blender, np.transpose(body_quat, (1, 0, 2))):
    all_joint_rotations[name] = joint_quats

all_joint_rotations["jaw"] = jaw_quat
for name, joint_quats in zip(left_hand_joint_names_blender, np.transpose(left_hand_quat, (1, 0, 2))):
    all_joint_rotations[name] = joint_quats
for name, joint_quats in zip(right_hand_joint_names_blender, np.transpose(right_hand_quat, (1, 0, 2))):
    all_joint_rotations[name] = joint_quats

output_json = {
    joint_name: [
        {
            "frame": int(frame_ids[i]),
            "rotation": joint_quats[i].tolist()
        }
        for i in range(num_frames)
    ]
    for joint_name, joint_quats in all_joint_rotations.items()
}

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(output_json, f, indent=2)

print(json.dumps(output_json, indent=2))
print(f"\nSaved to: {OUTPUT_JSON_PATH}")