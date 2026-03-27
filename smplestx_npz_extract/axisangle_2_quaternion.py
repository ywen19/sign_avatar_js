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
    return np.stack([rodrigues_to_quat_xyzw(rv) for rv in rotvec_block], axis=0)


def convert_hand_pose_with_reference(hand_pose_flat: np.ndarray, hand_ref_flat: np.ndarray) -> np.ndarray:
    hand_pose = hand_pose_flat.reshape(15, 3).astype(np.float32)
    hand_ref = hand_ref_flat.reshape(15, 3).astype(np.float32)
    hand_pose_final = hand_pose + hand_ref
    return convert_pose_block_to_quat(hand_pose_final)


POSE_NPZ_PATH = "./demo/output_smplx/queue/000037_person00_smplx_pose.npz"
HANDPOSES_NPZ_PATH = "./smplx_handposes.npz"
OUTPUT_JSON_PATH = "./000037_person00_smplx_quat_blender_names.json"

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

global_orient = pose_data["global_orient"].reshape(1, 3).astype(np.float32)
body_pose = pose_data["body_pose"].reshape(21, 3).astype(np.float32)
left_hand_pose = pose_data["left_hand_pose"].astype(np.float32)
right_hand_pose = pose_data["right_hand_pose"].astype(np.float32)
jaw_pose = pose_data["jaw_pose"].reshape(1, 3).astype(np.float32)

handpose_data = np.load(HANDPOSES_NPZ_PATH, allow_pickle=True)
hand_poses = handpose_data["hand_poses"].item()
left_hand_relaxed_ref, right_hand_relaxed_ref = hand_poses["relaxed"]

left_hand_relaxed_ref = left_hand_relaxed_ref.astype(np.float32)
right_hand_relaxed_ref = right_hand_relaxed_ref.astype(np.float32)

# Convert source SMPL-X pose to quaternions
pelvis_quat = rodrigues_to_quat_xyzw(global_orient[0])
jaw_quat = rodrigues_to_quat_xyzw(jaw_pose[0])

body_quat = convert_pose_block_to_quat(body_pose)
left_hand_quat = convert_hand_pose_with_reference(left_hand_pose, left_hand_relaxed_ref)
right_hand_quat = convert_hand_pose_with_reference(right_hand_pose, right_hand_relaxed_ref)

# Apply one global correction ONLY to pelvis
# First try: 180 degrees around X
pelvis_correction = axis_angle_to_quat_xyzw(np.array([1.0, 0.0, 0.0], dtype=np.float32), math.pi)

# corrected pelvis = correction * source
pelvis_quat = normalize_quat_xyzw(quat_xyzw_multiply(pelvis_correction, pelvis_quat))

output_json = {
    "pelvis": pelvis_quat.tolist(),
    **{name: quat.tolist() for name, quat in zip(body_joint_names_blender, body_quat)},
    "jaw": jaw_quat.tolist(),
    **{name: quat.tolist() for name, quat in zip(left_hand_joint_names_blender, left_hand_quat)},
    **{name: quat.tolist() for name, quat in zip(right_hand_joint_names_blender, right_hand_quat)},
}

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(output_json, f, indent=2)

print(json.dumps(output_json, indent=2))
print(f"\nSaved to: {OUTPUT_JSON_PATH}")