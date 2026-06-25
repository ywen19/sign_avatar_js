from pathlib import Path
from typing import Iterable, List, Optional
import json


# Current raw SMPL-X NPZ motion data root
SMPL_DATA_ROOT = Path("/home/ywen/Desktop/smpl_data_threejs/")

# Keep the avatar root orientation stable across concatenated vocab clips.
# The converted motion clips carry a skeleton-coordinate offset on the hips; using
# mathematical identity ([0, 0, 0, 1]) makes the model lie/fold incorrectly.
FIXED_HIP_ROTATION = [
    0.7193659472722247,
    0.00006376511902153256,
    -0.012867924020065152,
    -0.6945120923141355,
]

# temp json file paths following the order of a sentence example
# this is used for scripting motion concatenation
# remove this later once we finished the axis 2 angle conversion
temp_json_paths = [
    "../vocabs/bournemouth_0.json",
    "../vocabs/art_gallery_0.json",
    "../vocabs/one_0.json",
    "../vocabs/three_0.json",
    "../vocabs/high_0.json",
    "../vocabs/street_0.json",
    "../vocabs/bournemouth_0.json",
    "../vocabs/b_0.json",
    "../vocabs/h_0.json",
    "../vocabs/one_0.json",
    "../vocabs/one_0.json",
    "../vocabs/j_0.json",
    "../vocabs/q_0.json",
    "../vocabs/england_0.json",
]


def retrieve_json_paths(tokens: Iterable[str]) -> List[Optional[Path]]:
    json_paths: List[Optional[Path]] = []

    for token in tokens:
        vocab_name = token.strip().lower().replace(" ", "_")
        json_path = SMPL_DATA_ROOT / vocab_name / f"{vocab_name}_0.json"

        json_paths.append(json_path if json_path.exists() else None)

    return json_paths


def retrieve_npz_paths(tokens: Iterable[str]) -> List[Optional[Path]]:
    # Backward-compatible alias for older debug code/imports.
    # The frontend motion pipeline now consumes converted Three.js JSON files.
    return retrieve_json_paths(tokens)


def normalize_rotation(rot):
    norm = sum(v * v for v in rot) ** 0.5

    if norm < 1e-8:
        return [0.0, 0.0, 0.0, 1.0]

    return [v / norm for v in rot]


def interpolate_rotation(rot_a, rot_b, t):
    rot = [
        (1.0 - t) * rot_a[0] + t * rot_b[0],
        (1.0 - t) * rot_a[1] + t * rot_b[1],
        (1.0 - t) * rot_a[2] + t * rot_b[2],
        (1.0 - t) * rot_a[3] + t * rot_b[3],
    ]

    return normalize_rotation(rot)


def normalize_bone_name(bone_name):
    return str(bone_name).strip().lower().replace("mixamorig:", "")


def is_hip_bone(bone_name):
    return normalize_bone_name(bone_name) == "hips"


def get_output_rotation(bone_name, source_rotation):
    if is_hip_bone(bone_name):
        return list(FIXED_HIP_ROTATION)

    return source_rotation


def load_motion_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def validate_motion_json(motion_json, json_path=None):
    bone_names = list(motion_json.keys())

    if not bone_names:
        raise ValueError(f"Empty motion JSON: {json_path}")

    ref_bone = bone_names[0]
    ref_frame_count = len(motion_json[ref_bone])

    for bone_name in bone_names:
        frames = motion_json[bone_name]

        if len(frames) != ref_frame_count:
            raise ValueError(
                f"Frame count mismatch in {json_path}: "
                f"{bone_name} has {len(frames)} frames, "
                f"expected {ref_frame_count}"
            )

        for item in frames:
            if "frame" not in item or "rotation" not in item:
                raise ValueError(
                    f"Invalid frame item in {json_path}, bone {bone_name}: {item}"
                )

            if len(item["rotation"]) != 4:
                raise ValueError(
                    f"Invalid rotation length in {json_path}, bone {bone_name}: "
                    f"{item['rotation']}"
                )

    return bone_names, ref_frame_count


def append_clip_to_output(output_json, motion_json, bone_names, frame_counter):
    """
    Append all frames from one vocab motion into output_json.

    Returns:
        Updated frame_counter.
    """
    num_frames = len(motion_json[bone_names[0]])

    for local_frame_idx in range(num_frames):
        for bone_name in bone_names:
            source_rotation = motion_json[bone_name][local_frame_idx]["rotation"]
            rotation = get_output_rotation(bone_name, source_rotation)

            output_json[bone_name].append(
                {
                    "frame": frame_counter,
                    "rotation": rotation,
                }
            )

        frame_counter += 1

    return frame_counter


def append_interpolation_to_output(
    output_json,
    prev_motion,
    next_motion,
    bone_names,
    frame_counter,
    interpolation_frames,
):
    """
    Append interpolation frames between prev_motion's last frame
    and next_motion's first frame.

    Returns:
        Updated frame_counter.
    """
    if interpolation_frames <= 0:
        return frame_counter

    for interp_idx in range(1, interpolation_frames + 1):
        t = interp_idx / (interpolation_frames + 1)

        for bone_name in bone_names:
            if is_hip_bone(bone_name):
                interp_rotation = list(FIXED_HIP_ROTATION)
            else:
                prev_rotation = prev_motion[bone_name][-1]["rotation"]
                next_rotation = next_motion[bone_name][0]["rotation"]

                interp_rotation = interpolate_rotation(
                    prev_rotation,
                    next_rotation,
                    t,
                )

            output_json[bone_name].append(
                {
                    "frame": frame_counter,
                    "rotation": interp_rotation,
                }
            )

        frame_counter += 1

    return frame_counter


def build_concatenated_motion_json(
    json_paths,
    interpolation_frames=5,
    output_path="motion.json",
):
    """
    Concatenate multiple vocab motion JSON files into one motion JSON,
    inserting linear interpolation frames between neighboring vocab motions.
    """
    json_paths = list(json_paths)

    if not json_paths:
        raise ValueError("json_paths is empty.")

    motion_cache = {}

    def get_motion(path):
        path = str(path)

        if path not in motion_cache:
            motion = load_motion_json(path)
            validate_motion_json(motion, path)
            motion_cache[path] = motion

        return motion_cache[path]

    first_motion = get_motion(json_paths[0])
    bone_names, _ = validate_motion_json(first_motion, json_paths[0])

    output_json = {bone_name: [] for bone_name in bone_names}
    frame_counter = 1

    previous_motion = None

    for json_path in json_paths:
        current_motion = get_motion(json_path)

        current_bone_names, _ = validate_motion_json(current_motion, json_path)

        if set(current_bone_names) != set(bone_names):
            raise ValueError(
                f"Bone mismatch in {json_path}. "
                f"Expected same bones as first motion."
            )

        if previous_motion is not None:
            frame_counter = append_interpolation_to_output(
                output_json=output_json,
                prev_motion=previous_motion,
                next_motion=current_motion,
                bone_names=bone_names,
                frame_counter=frame_counter,
                interpolation_frames=interpolation_frames,
            )

        frame_counter = append_clip_to_output(
            output_json=output_json,
            motion_json=current_motion,
            bone_names=bone_names,
            frame_counter=frame_counter,
        )

        previous_motion = current_motion

    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=2)

    print(f"Saved concatenated motion to: {output_path}")
    print(f"Total frames: {frame_counter - 1}")
    print(f"Total bones: {len(bone_names)}")

    return output_json


if __name__ == "__main__":
    build_concatenated_motion_json(
        SMPL_DATA_ROOT,
        interpolation_frames=1,
        output_path="motion.json",
    )