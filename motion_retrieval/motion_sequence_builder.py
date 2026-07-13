"""
This module builds final avatar motion JSON files from traced vocab tokens.

It resolves each vocab token to its corresponding motion JSON file, loads and
validates the motion data, then concatenates multiple vocab motion clips into
one continuous animation.

Interpolation frames are inserted between neighboring clips to make transitions
smoother. Only bones used by the frontend signing animation are included in the
concatenated output.
"""


from pathlib import Path
from typing import List, Optional, Dict, Any, Final, Tuple
import json


# current motion data root
SMPL_DATA_ROOT: Final[Path] = Path("/home/ywen/Desktop/smpl_data_threejs/")

# Keep this ordered list aligned with ALLOWED_ANIMATION_BONE_NAMES in
# static/js/avatar_animation.js. Source clips use prefixed Mixamo bone names.
ALLOWED_ANIMATION_BONES: Final[Tuple[str, ...]] = (
    "mixamorig:Spine",
    "mixamorig:Spine1",
    "mixamorig:Spine2",
    "mixamorig:Neck",
    "mixamorig:Head",
    "mixamorig:LeftShoulder",
    "mixamorig:LeftArm",
    "mixamorig:LeftForeArm",
    "mixamorig:LeftHand",
    "mixamorig:RightShoulder",
    "mixamorig:RightArm",
    "mixamorig:RightForeArm",
    "mixamorig:RightHand",
    "mixamorig:LeftHandThumb1",
    "mixamorig:LeftHandThumb2",
    "mixamorig:LeftHandThumb3",
    "mixamorig:LeftHandIndex1",
    "mixamorig:LeftHandIndex2",
    "mixamorig:LeftHandIndex3",
    "mixamorig:LeftHandMiddle1",
    "mixamorig:LeftHandMiddle2",
    "mixamorig:LeftHandMiddle3",
    "mixamorig:LeftHandRing1",
    "mixamorig:LeftHandRing2",
    "mixamorig:LeftHandRing3",
    "mixamorig:LeftHandPinky1",
    "mixamorig:LeftHandPinky2",
    "mixamorig:LeftHandPinky3",
    "mixamorig:RightHandThumb1",
    "mixamorig:RightHandThumb2",
    "mixamorig:RightHandThumb3",
    "mixamorig:RightHandIndex1",
    "mixamorig:RightHandIndex2",
    "mixamorig:RightHandIndex3",
    "mixamorig:RightHandMiddle1",
    "mixamorig:RightHandMiddle2",
    "mixamorig:RightHandMiddle3",
    "mixamorig:RightHandRing1",
    "mixamorig:RightHandRing2",
    "mixamorig:RightHandRing3",
    "mixamorig:RightHandPinky1",
    "mixamorig:RightHandPinky2",
    "mixamorig:RightHandPinky3",
)


def retrieve_json_paths(tokens: List[str]) -> List[Optional[Path]]:
    """
    Retrieve the motion from vocab json.

    For one vocab chunk, we could have multiple json files.
    Here for simplicity, we always load the first one.

    tokens  (List[str])      :  vocabulary tokens to retrieve motion from

    Return:
    (List[Optional[Path]])   :  list containing the resolved JSON path 
                                for each token
    """
    json_paths: List[Optional[Path]] = []

    for token in tokens:
        vocab_name = token.strip().lower().replace(" ", "_")
        json_path = SMPL_DATA_ROOT / vocab_name / f"{vocab_name}_0.json"

        json_paths.append(json_path if json_path.exists() else None)

    return json_paths


def normalize_rotation(rot: List[float]) -> List[float]:
    """
    Normalize a quaternion rotation to unit length.

    If the rotation is too close to zero length, return the identity rotation
    to avoid division by zero.

    rot  (List[float])  :  quaternion rotation to be normalized

    Return:
    (List[float])       :  normalized quaternion rotation
    """
    norm = sum(v * v for v in rot) ** 0.5

    if norm < 1e-8:
        return [0.0, 0.0, 0.0, 1.0]

    return [v / norm for v in rot]


def interpolate_rotation(
    rot_a: List[float], rot_b: List[float], t: float) -> List[float]:
    """
    Linearly interpolate between two quaternion rotations.

    This is used to create transition frames between two neighboring
    motion clips. The interpolated rotation is normalized before being returned.

    rot_a  (List[float])  :  source quaternion rotation
    rot_b  (List[float])  :  target quaternion rotation
    t      (float)        :  linear interp factor, typically between 0 and 1

    Return:
    (List[float])        :  interpolated quaternion rotation at step t
    """
    rot = [
        (1.0 - t) * rot_a[0] + t * rot_b[0],
        (1.0 - t) * rot_a[1] + t * rot_b[1],
        (1.0 - t) * rot_a[2] + t * rot_b[2],
        (1.0 - t) * rot_a[3] + t * rot_b[3],
    ]

    return normalize_rotation(rot)


def load_motion_json(json_path: str) -> Dict[str, Any]:
    """
    Load motion from json file under given path.

    json_path  (str)  :  motion json path

    Return:
    (Dict[str, Any])  :  loaded motion 
    """
    with open(json_path, "r") as f:
        return json.load(f)


def validate_threejs_motion_json(
    motion_json: Dict[str, Any], json_path: Optional[str]=None
) -> Tuple[List[str], int]:
    """
    Validate the structure and frame counts of a motion JSON object.

    Each bone must contain the same number of frames. Every frame item must
    include a frame number and a quaternion rotation with four components.

    Return bone names and frame count for motion if validation passed.

    motion_json  (Dict[str, Any])  :  loaded motion data
    json_path    (Optional[str])   :  motion source json file path

    Return:
    Tuple[List[str], int]          :  bone names and frame count for motion 
                                      if validation passed
    """
    if not isinstance(motion_json, dict) or "bones" not in motion_json:
        raise ValueError(f"Invalid Three.js motion JSON: missing bones: {json_path}")
    
    # check if bone exists
    bones = motion_json["bones"]
    if not isinstance(bones, dict) or not bones:
        raise ValueError(f"Empty Three.js motion JSON bones: {json_path}")

    # use the frame count for the first bone as the frame check anchor
    bone_names = list(bones.keys())
    ref_bone = bone_names[0]
    ref_frame_count = len(bones[ref_bone])

    for bone_name in bone_names:
        frames = bones[bone_name]

        if len(frames) != ref_frame_count:
            raise ValueError(
                f"Frame count mismatch in {json_path}: "
                f"{bone_name} has {len(frames)} frames, "
                f"expected {ref_frame_count}"
            )

        for item in frames:
            if "f" not in item or "rot" not in item:
                raise ValueError(
                    f"Invalid Three.js frame item in {json_path}, "
                    f"bone {bone_name}: {item}"
                )

            if len(item["rot"]) != 4:
                raise ValueError(
                    f"Invalid Three.js rotation length in {json_path}, "
                    f"bone {bone_name}: {item['rot']}"
                )

    return bone_names, ref_frame_count


def validate_allowed_animation_bones(
    bone_names: List[str], json_path: Optional[str] = None
) -> List[str]:
    """Return the allowed output bones after confirming they all exist."""
    available_bones = set(bone_names)
    missing_bones = [
        bone_name
        for bone_name in ALLOWED_ANIMATION_BONES
        if bone_name not in available_bones
    ]

    if missing_bones:
        raise ValueError(
            f"Missing allowed animation bones in {json_path}: "
            f"{', '.join(missing_bones)}"
        )

    return list(ALLOWED_ANIMATION_BONES)


def append_threejs_clip_to_output(
    output_json: Dict[str, Any], 
    motion_json: Dict[str, Any], 
    bone_names: List[str], 
    frame_counter: int,
) -> int:
    """
    Append all frames from one vocab motion into the final output json.

    Each frame is copied for every allowed bone and assigned a new sequential
    frame number.

    output_json    (Dict[str, Any])  :  output motion data keyed by bone name
    motion_json    (Dict[str, Any])  :  source motion data to be appended from
    bone_names     (List[str])       :  names of the bones from source motion
    frame_counter  (int)             :  frame number to be assigned to the 
                                        first appended frame

    Return:
    frame_counter  (int)             :  next available frame number after all 
                                        source frames are appended
    """
    num_frames = len(motion_json["bones"][bone_names[0]])

    for local_frame_idx in range(num_frames):
        for bone_name in bone_names:
            source_rotation = motion_json["bones"][bone_name][local_frame_idx]["rot"]

            output_json["bones"][bone_name].append(
                {
                    "f": frame_counter,
                    "rot": source_rotation,
                }
            )

        frame_counter += 1

    return frame_counter


def append_threejs_interpolation_to_output(
    output_json: Dict[str, Any],
    prev_motion: Dict[str, Any],
    next_motion: Dict[str, Any],
    bone_names: List[str],
    frame_counter: int,
    interpolation_frames: int,
) -> int:
    """
    Append transition frames between two neighboring motion clips.

    Each transition frame interpolates between the final rotation of prev_motion
    and the first rotation of next_motion for every allowed bone.

    The generated frames are appended directly to output_json.

    output_json           (Dict[str, Any])  :  output motion keyed by bone name
    prev_motion           (Dict[str, Any])  :  motion clip that appears before 
                                               the transition
    next_motion           (Dict[str, Any])  :  motion clip that appears after 
                                               the transition
    bone_names            (List[str])       :  names of bones to process
    frame_counter         (int)             :  frame number assigned to the 
                                               first transition frame
    interpolation_frames  (int)             :  number of transition frames

    Return:
    frame_counter         (int)             :  next available frame number after 
                                               the transition frames have been 
                                               appended
    """
    if interpolation_frames <= 0:
        return frame_counter

    for interp_idx in range(1, interpolation_frames + 1):
        t = interp_idx / (interpolation_frames + 1)

        for bone_name in bone_names:
            prev_rotation = prev_motion["bones"][bone_name][-1]["rot"]
            next_rotation = next_motion["bones"][bone_name][0]["rot"]
            interp_rotation = interpolate_rotation(
                prev_rotation, next_rotation, t
            )

            output_json["bones"][bone_name].append(
                {
                    "f": frame_counter,
                    "rot": interp_rotation,
                }
            )

        frame_counter += 1  # update frame counter per interp step (frame)

    return frame_counter


def build_concatenated_threejs_motion_json(
    json_paths: List[str],
    interpolation_frames: int = 5,
    output_path: str = "motion.json",
    animation_name: str ="motion",
    source_text: Optional[str] = None,
    traced_tokens: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Concatenate converted Three.js motion JSON files into one motion JSON,
    inserting linear interpolation frames between neighboring vocab motions.

    json_paths            (List[str])     :  Three.js motion JSON file paths
    interpolation_frames  (int)           :  number of transition frames between
                                             neighboring motion clips
    output_path           (str)           :  final motion json file path
    animation_name        (str)           :  output animation name
    source_text        (Optional[str])    :  text associated with the animation;
                                             basically llm answer
    traced_tokens  (Optional[List[str]])  :  vocab tokens used to build the 
                                             animation

    Return:
    output_json      (Dict[str, Any])  :  concatenated Three.js motion data
    """
    json_paths = list(json_paths)

    if not json_paths:
        raise ValueError("json_paths is empty.")

    motion_cache = {}

    def get_motion(path: str) -> Dict[str, Any]:
        """
        Load, validate, and cache a Three.js motion JSON file.

        path       (str)  :  path to the motion JSON file.

        Return:
        (Dict[str, Any])  :  motion data in the parsed JSON file
        """
        path = str(path)

        if path not in motion_cache:
            motion = load_motion_json(path)
            validate_threejs_motion_json(motion, path)
            motion_cache[path] = motion

        return motion_cache[path]

    # retrieve the first vocab motion
    first_motion = get_motion(json_paths[0])
    source_bone_names, _ = validate_threejs_motion_json(
        first_motion, json_paths[0]
    )
    bone_names = validate_allowed_animation_bones(
        source_bone_names, json_paths[0]
    )
    # Build output keys only for bones used by frontend animation playback.
    output_json = {
        "name": animation_name,
        "fps": first_motion.get("fps", 30),
        "bones": {bone_name: [] for bone_name in bone_names},
    }

    if source_text is not None:
        output_json["source_text"] = source_text
    if traced_tokens is not None:
        output_json["traced_tokens"] = list(traced_tokens)

    frame_counter = 0
    previous_motion = None

    # step through each motion file
    for json_path in json_paths:
        current_motion = get_motion(json_path)
        current_bone_names, _ = validate_threejs_motion_json(
            current_motion, json_path
        )

        validate_allowed_animation_bones(current_bone_names, json_path)

        # motion from the first vocab (first in the json path list)
        # will be added to the final motion directly
        # otherwise, we need to trace the frame counter
        # and also apply motion clip transition
        if previous_motion is not None:
            frame_counter = append_threejs_interpolation_to_output(
                output_json=output_json,
                prev_motion=previous_motion,
                next_motion=current_motion,
                bone_names=bone_names,
                frame_counter=frame_counter,
                interpolation_frames=interpolation_frames,
            )

        frame_counter = append_threejs_clip_to_output(
            output_json=output_json,
            motion_json=current_motion,
            bone_names=bone_names,
            frame_counter=frame_counter,
        )

        # update previous motion for next transition
        previous_motion = current_motion

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # write out final motion to the specified JSON path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)

    print(f"Saved concatenated Three.js motion to: {output_path}")
    print(f"Total frames: {frame_counter}")
    print(f"Total bones: {len(bone_names)}")

    return output_json


def flatten_traced_tokens(
    traced_tokens_by_sentence: List[List[str]]
) -> List[str]:
    """
    Flatten traced tokens grouped by sentence into a single list.
    Original format is that each sentence has a list of tokens.
    """
    return [
        token
        for sentence_tokens in traced_tokens_by_sentence
        for token in sentence_tokens
    ]


def build_threejs_motion_from_tokens(
    tokens: List[str],
    output_path: str = "tmp/answer_motion.json",
    interpolation_frames: int = 5,
    animation_name: str = "answer_motion",
    source_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete traced-token-to-Three.js-motion pipeline.

    tokens                (List[str])      :  flattened traced vocabulary tokens
    output_path           (str)            :  final motion json file path
    interpolation_frames  (int)            :  number of transition frames between
                                              neighboring motion clips
    animation_name        (str)            :  output animation name
    source_text           (Optional[str])  :  text associated with the animation;
                                              basically llm answer

    Return:
    (Dict[str, Any])                       :  a dictionary containing 
                                              1. final motion json file path
                                              2. final motion
                                              3. tokens failed to match motion file
                                              4. paths to token motion files that 
                                              are found
    """
    tokens = list(tokens)
    # retrieve all token motion paths in the order of the the input tokens
    resolved_paths = retrieve_json_paths(tokens)

    # log paths to token motion files that are found
    # and tokens failed to match any motion files
    found_paths = []
    missing_tokens = []
    for token, path in zip(tokens, resolved_paths):
        if path is None:
            missing_tokens.append(token)
        else:
            found_paths.append(path)

    if not found_paths:
        raise ValueError("No motion JSON files found for traced tokens.")

    # append motion clips and apply transition
    motion_json = build_concatenated_threejs_motion_json(
        found_paths,
        interpolation_frames=interpolation_frames,
        output_path=output_path,
        animation_name=animation_name,
        source_text=source_text,
        traced_tokens=tokens,
    )

    return {
        "output_path": Path(output_path),
        "motion_json": motion_json,
        "missing_tokens": missing_tokens,
        "found_paths": found_paths,
    }
