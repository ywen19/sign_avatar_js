"""
This module converts one SMPL-X quaternion motion JSON file to Mixamo
bone rotations and Three.js-compatible frontend animation data.

SMPL-X bone rotations are retargeted using per-bone quaternion offsets stored in a
local JSON file. The converted rotations are then combined with rest-pose rotations
read from a Mixamo GLB model and baked into the frontend keyframe structure.

All quaternion values use XYZW component ordering. 
"""


import json
import math
import re
import struct
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


REPO_ROOT: Path = Path(__file__).resolve().parent
INPUT_MOTION_PATH: Path = REPO_ROOT / "vocabs" / "art_gallery_0.json"
RETARGET_OFFSETS_PATH: Path = (
    REPO_ROOT / "vocabs" / "smplx_to_mixamo_body_rotation_offsets.json"
)
OUTPUT_MOTION_PATH: Path = REPO_ROOT / "vocabs" / "art_gallery_0_mixamo.json"
OUTPUT_FRONTEND_PATH: Path = REPO_ROOT / "vocabs" / "art_gallery_0_threejs.json"
MODEL_GLB_PATH: Path = REPO_ROOT / "models" / "model.glb"
OUTPUT_FPS: int = 30


def load_json(path: Path) -> Any:
    """
    Load data from a local JSON file.

    path  (Path)  :  JSON file path

    Return:
    (Any)         :  deserialized JSON data
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Write a dictionary to a local formatted JSON file.

    path  (Path)            :  output JSON file path
    data  (Dict[str, Any])  :  data to be flushed to JSON
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def normalize_bone_name(name: str) -> str:
    """
    Remove a Mixamo namespace prefix from a bone name.

    name  (str)  :  original model bone name

    Return:
    (str)        :  normalized bone name
    """
    return re.sub(
        r"^mixamorig[:_.]?", "", str(name).strip(), flags=re.IGNORECASE
    )


def quat_from_xyzw(values: Any) -> Tuple[float, ...]:
    """
    Convert input value to a XYZW-ordered quaternion and normalize.

    values  (Any)        :  quaternion components

    Return:
    (Tuple[float, ...])  :  normalized XYZW quaternion
    """
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"Expected [x, y, z, w] quaternion, got {values}")

    x, y, z, w = values
    return quat_normalize((float(x), float(y), float(z), float(w)))


def quat_to_xyzw(quat: Tuple[float, ...]) -> List[float]:
    """
    Normalize a quaternion.

    quat  (Tuple[float, ...])  :  quaternion using XYZW ordering

    Return: 
    (List[float])              :  normalized XYZW quaternion 
    """
    x, y, z, w = quat_normalize(quat)
    return [x, y, z, w]


def quat_normalize(quat: Tuple[float, ...]) -> Tuple[float, ...]:
    """
    Normalize a quaternion to unit length.

    quat  (Tuple[float, ...])  :  quaternion using XYZW ordering

    Return:
    (Tuple[float, ...])        :  normalized XYZW quaternion
    """
    x, y, z, w = quat
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length == 0.0:
        raise ValueError("Cannot normalize zero-length quaternion")
    return (x / length, y / length, z / length, w / length)


def quat_conjugate(quat: Tuple[float, ...]) -> Tuple[float, ...]:
    """
    Calculate the conjugate of an XYZW quaternion.

    For a unit quaternion, the conjugate is also its inverse rotation.

    quat  (Tuple[float, ...])  :  quaternion using XYZW ordering

    Return:
    (Tuple[float, ...])        :  conjugated XYZW quaternion
    """
    x, y, z, w = quat
    return (-x, -y, -z, w)


def quat_multiply(
    left: Tuple[float, ...], right: Tuple[float, ...]
) -> Tuple[float, ...]:
    """
    Multiply two XYZW quaternion rotations using Hamilton product.

    left   (Tuple[float, ...])  :  left-side XYZW quaternion
    right  (Tuple[float, ...])  :  right-side XYZW quaternion

    Return:
    (Tuple[float, ...])         :  resulted quaternion
    """
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def read_glb_json(glb_path: Path) -> Dict[str, Any]:
    """
    Load JSON chunk embedded in a binary GLB model.

    glb_path  (Path)  :  binary GLB model path

    Return:
    (Dict[str, Any])  :  deserialized glTF JSON document
    """
    data = glb_path.read_bytes()

    if data[:4] != b"glTF":
        raise ValueError(f"Expected GLB file: {glb_path}")

    # skip the 12-byte GLB header before reading chunks
    offset = 12
    while offset < len(data):
        # read the current chunk length and type from the GLB chunk header
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset : offset + chunk_length]
        offset += chunk_length

        # return the standard glTF JSON chunk identified by its GLB type constant
        if chunk_type == 0x4E4F534A:
            return json.loads(chunk_data.rstrip(b" \t\r\n\x00").decode("utf-8"))

    raise ValueError(f"No JSON chunk found in GLB: {glb_path}")


def load_model_rest_rotations(glb_path: Path) -> Dict[str, Tuple[float, ...]]:
    """
    Load rest-pose rotations for all skinned joints in a GLB model.

    Bone names are normalized by removing their Mixamo namespace prefixes.
    Nodes without an explicit rotation use the identity quaternion.

    glb_path  (Path)                :  binary Mixamo GLB model path

    Return:
    (Dict[str, Tuple[float, ...]])  :  rest-pose quaternion indexed by bone
    """
    gltf = read_glb_json(glb_path)
    nodes = gltf.get("nodes", [])
    rest_rotations = {}

    for skin in gltf.get("skins", []):
        for node_index in skin.get("joints", []):
            node = nodes[node_index]
            bone_name = normalize_bone_name(node.get("name", ""))
            rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
            rest_rotations[bone_name] = quat_from_xyzw(rotation)

    return rest_rotations


def quat_dot(left: Tuple[float, ...], right: Tuple[float, ...]) -> float:
    """
    Dot product of two quaternions.

    left   (Tuple[float, ...])  :  first XYZW quaternion
    right  (Tuple[float, ...])  :  second XYZW quaternion

    Return:
    (float)              :  quaternion dot product
    """
    return sum(a * b for a, b in zip(left, right))


def quat_negate(
    quat: Tuple[float, ...]
) -> Tuple[float, ...]:
    """
    Negate every component of a quaternion.

    quat  (Tuple[float, ...])  :  quaternion using XYZW ordering

    Return:
    (Tuple[float, ...])        :  quaternion with opposite component signs
    """
    return tuple(-v for v in quat)


def keep_quaternion_continuity(
    quat: Tuple[float, ...],
    previous_quat: Optional[Tuple[float, ...]],
) -> Tuple[float, ...]:
    """
    Choose the quaternion sign closest to the preceding keyframe.

    A quaternion and its negation describe the same rotation. 
    Keeping neighboring quaternion signs in the same hemisphere prevents 
    interpolation from taking an unintended path.

    quat           (Tuple[float, ...])            :  current XYZW quaternion
    previous_quat  (Optional[Tuple[float, ...]])  :  preceding XYZW quaternion

    Return:
    (Tuple[float, ...])                           :  sign-adjusted quaternion
    """
    if previous_quat is not None and quat_dot(quat, previous_quat) < 0.0:
        return quat_negate(quat)
    return quat


def iter_offset_groups(
    offsets_data: Dict[str, Any]
) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """
    Iterate over body and finger retarget-offset definitions.

    offsets_data  (Dict[str, Any])               :  retarget configuration 
                                                    grouped by bone category

    Return:
    (Iterator[Tuple[str, str, Dict[str, Any]]])  :  group name, source bone 
                                                    name, and offset data
    """
    for group_name in ("body_offsets", "finger_offsets"):
        for source_bone, offset_info in offsets_data.get(group_name, {}).items():
            yield group_name, source_bone, offset_info


def convert_motion(
    source_motion: Dict[str, Any], offsets_data: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str], int, int, Dict[str, int]]:
    """
    Retarget SMPL-X quaternion motion to Mixamo bone rotations.

    Each source rotation is converted using offset * source * offset_inverse. 
    Invalid keyframes are skipped, and quaternion signs are adjusted across 
    frames to preserve smooth interpolation.

    source_motion  (Dict[str, Any])  :  SMPL-X motion keyframes indexed by 
                                        source bone name
    offsets_data   (Dict[str, Any])  :  source-to-target bone names and 
                                        quaternion offsets

    Return:
    (Tuple[Dict[str, Any], List[str], int, int, Dict[str, int]])  :  
    converted Mixamo motion, missing source bones, skipped keyframe count, 
    converted keyframe count, and bone count by group
    """
    output_motion = {}
    skipped_bones = []
    skipped_entries = 0
    converted_entries = 0
    converted_by_group = {}

    for group_name, source_bone, offset_info in iter_offset_groups(offsets_data):
        if source_bone not in source_motion:
            skipped_bones.append(source_bone)
            continue

        target_bone = offset_info["target_bone"]
        offset = quat_from_xyzw(offset_info["offset"])
        offset_inv = quat_conjugate(offset)
        previous_target_quat = None
        output_entries = []

        for entry in source_motion[source_bone]:
            try:
                # validate the source frame index and XYZW quaternion
                frame = int(entry["frame"])
                source_quat = quat_from_xyzw(entry["rotation"])
            except Exception as exc:
                print(f"[WARN] Skipping {source_bone} entry {entry}: {exc}")
                skipped_entries += 1
                continue

            # offset * source * offset inverse
            target_quat = quat_multiply(
                quat_multiply(offset, source_quat), offset_inv
            )
            target_quat = quat_normalize(target_quat)
            # keep consecutive quaternion signs suitable for 
            # frontend interpolation
            target_quat = keep_quaternion_continuity(
                target_quat, previous_target_quat
            )
            previous_target_quat = target_quat

            output_entries.append(
                {
                    "frame": frame,
                    "rotation": quat_to_xyzw(target_quat),
                }
            )
            converted_entries += 1

        output_motion[target_bone] = output_entries
        converted_by_group[group_name] = converted_by_group.get(
            group_name, 0
        ) + 1

    return (
        output_motion, skipped_bones, skipped_entries, converted_entries, 
        converted_by_group,
    )


def bake_frontend_motion(
    raw_mixamo_motion: Dict[str, Any],
    rest_rotations: Dict[str, Tuple[float, ...]],
    animation_name: str,
) -> Dict[str, Any]:
    """
    Bake Mixamo pose rotations into Three.js-compatible frontend keyframes.

    Each pose rotation is applied after its model rest-pose rotation. 
    Output keyframes use f for frame indices, and rot for XYZW quaternion. 
    Frame indices starts at zero when the source animation begins at a 
    nonzero frame.

    raw_mixamo_motion  (Dict[str, Any])                :  retargeted Mixamo 
                                                          pose rotations by bone
    rest_rotations     (Dict[str, Tuple[float, ...]])  :  model rest rotations
    animation_name     (str)                           :  name stored in the 
                                                          frontend animation data

    Return:
    (Dict[str, Any])                                   :  
    frontend animation name, frame rate, and Three.js-compatible bone keyframes
    """
    normalized_bones = {}
    min_frame = None

    for bone_name, entries in raw_mixamo_motion.items():
        normalized_name = normalize_bone_name(bone_name)
        rest_rotation = rest_rotations.get(normalized_name, (0.0, 0.0, 0.0, 1.0))
        previous_quat = None
        keyframes = []

        for entry in entries:
            # combine the model rest pose with the current retargeted pose rotation
            frame = int(entry["frame"])
            pose_rotation = quat_from_xyzw(entry["rotation"])
            threejs_quat = quat_multiply(rest_rotation, pose_rotation)
            threejs_quat = quat_normalize(threejs_quat)
            threejs_quat = keep_quaternion_continuity(threejs_quat, previous_quat)
            previous_quat = threejs_quat

            keyframes.append({
                "f": frame,
                "rot": quat_to_xyzw(threejs_quat),
            })

            min_frame = frame if min_frame is None else min(min_frame, frame)

        normalized_bones[bone_name] = keyframes

    if min_frame is not None and min_frame != 0:
        # rebase all keyframes so frontend playback starts at frame zero
        for keyframes in normalized_bones.values():
            for keyframe in keyframes:
                keyframe["f"] -= min_frame

    return {
        "name": animation_name,
        "fps": OUTPUT_FPS,
        "bones": normalized_bones,
    }


def main() -> None:
    """
    Convert the configured motion file and write Mixamo and frontend JSON outputs.

    Conversion statistics and missing source bones are printed after both output 
    files have been written.
    """
    # load source SMPL-X motion and source-to-target rotation offsets
    source_motion = load_json(INPUT_MOTION_PATH)
    offsets_data = load_json(RETARGET_OFFSETS_PATH)

    # retarget source rotations from SMPL-X bones to Mixamo bones
    (
        output_motion,
        skipped_bones,
        skipped_entries,
        converted_entries,
        converted_by_group,
    ) = convert_motion(source_motion, offsets_data)

    # write raw Mixamo pose rotations before applying model rest rotations
    write_json(OUTPUT_MOTION_PATH, output_motion)

    # bake model rest rotations into the Three.js-compatible animation
    rest_rotations = load_model_rest_rotations(MODEL_GLB_PATH)
    frontend_motion = bake_frontend_motion(
        output_motion,
        rest_rotations,
        animation_name=INPUT_MOTION_PATH.stem,
    )
    # write the final frontend animation JSON file
    write_json(OUTPUT_FRONTEND_PATH, frontend_motion)

    print(f"Wrote {OUTPUT_MOTION_PATH}")
    print(f"Wrote {OUTPUT_FRONTEND_PATH}")
    print(f"Converted Mixamo bones: {len(output_motion)}")
    for group_name, count in converted_by_group.items():
        print(f"  {group_name}: {count}")
    print(f"Converted rotation entries: {converted_entries}")
    print(f"Skipped entries: {skipped_entries}")

    if skipped_bones:
        print("[WARN] Missing source body bones:")
        for bone_name in skipped_bones:
            print(f" - {bone_name}")


if __name__ == "__main__":
    main()
