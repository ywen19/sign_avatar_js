import json
import math
import re
import struct
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
INPUT_MOTION_PATH = REPO_ROOT / "vocabs" / "art_gallery_0.json"
RETARGET_OFFSETS_PATH = REPO_ROOT / "vocabs" / "smplx_to_mixamo_body_rotation_offsets.json"
OUTPUT_MOTION_PATH = REPO_ROOT / "vocabs" / "art_gallery_0_mixamo.json"
OUTPUT_FRONTEND_PATH = REPO_ROOT / "vocabs" / "art_gallery_0_threejs.json"
MODEL_GLB_PATH = REPO_ROOT / "models" / "model.glb"
OUTPUT_FPS = 30


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def normalize_bone_name(name):
    return re.sub(r"^mixamorig[:_.]?", "", str(name).strip(), flags=re.IGNORECASE)


def quat_from_xyzw(values):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"Expected [x, y, z, w] quaternion, got {values}")

    x, y, z, w = values
    return quat_normalize((float(x), float(y), float(z), float(w)))


def quat_to_xyzw(quat):
    x, y, z, w = quat_normalize(quat)
    return [x, y, z, w]


def quat_normalize(quat):
    x, y, z, w = quat
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length == 0.0:
        raise ValueError("Cannot normalize zero-length quaternion")
    return (x / length, y / length, z / length, w / length)


def quat_conjugate(quat):
    x, y, z, w = quat
    return (-x, -y, -z, w)


def quat_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def read_glb_json(glb_path):
    data = glb_path.read_bytes()

    if data[:4] != b"glTF":
        raise ValueError(f"Expected GLB file: {glb_path}")

    offset = 12
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset:offset + chunk_length]
        offset += chunk_length

        if chunk_type == 0x4E4F534A:
            return json.loads(chunk_data.rstrip(b" \t\r\n\x00").decode("utf-8"))

    raise ValueError(f"No JSON chunk found in GLB: {glb_path}")


def load_model_rest_rotations(glb_path):
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


def quat_dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def quat_negate(quat):
    return tuple(-v for v in quat)


def keep_quaternion_continuity(quat, previous_quat):
    if previous_quat is not None and quat_dot(quat, previous_quat) < 0.0:
        return quat_negate(quat)
    return quat


def iter_offset_groups(offsets_data):
    for group_name in ("body_offsets", "finger_offsets"):
        for source_bone, offset_info in offsets_data.get(group_name, {}).items():
            yield group_name, source_bone, offset_info


def convert_motion(source_motion, offsets_data):
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
                frame = int(entry["frame"])
                source_quat = quat_from_xyzw(entry["rotation"])
            except Exception as exc:
                print(f"[WARN] Skipping {source_bone} entry {entry}: {exc}")
                skipped_entries += 1
                continue

            target_quat = quat_multiply(quat_multiply(offset, source_quat), offset_inv)
            target_quat = quat_normalize(target_quat)
            target_quat = keep_quaternion_continuity(target_quat, previous_target_quat)
            previous_target_quat = target_quat

            output_entries.append(
                {
                    "frame": frame,
                    "rotation": quat_to_xyzw(target_quat),
                }
            )
            converted_entries += 1

        output_motion[target_bone] = output_entries
        converted_by_group[group_name] = converted_by_group.get(group_name, 0) + 1

    return output_motion, skipped_bones, skipped_entries, converted_entries, converted_by_group


def bake_frontend_motion(raw_mixamo_motion, rest_rotations, animation_name):
    normalized_bones = {}
    min_frame = None

    for bone_name, entries in raw_mixamo_motion.items():
        normalized_name = normalize_bone_name(bone_name)
        rest_rotation = rest_rotations.get(normalized_name, (0.0, 0.0, 0.0, 1.0))
        previous_quat = None
        keyframes = []

        for entry in entries:
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
        for keyframes in normalized_bones.values():
            for keyframe in keyframes:
                keyframe["f"] -= min_frame

    return {
        "name": animation_name,
        "fps": OUTPUT_FPS,
        "bones": normalized_bones,
    }


def main():
    source_motion = load_json(INPUT_MOTION_PATH)
    offsets_data = load_json(RETARGET_OFFSETS_PATH)

    output_motion, skipped_bones, skipped_entries, converted_entries, converted_by_group = convert_motion(
        source_motion,
        offsets_data,
    )

    write_json(OUTPUT_MOTION_PATH, output_motion)

    rest_rotations = load_model_rest_rotations(MODEL_GLB_PATH)
    frontend_motion = bake_frontend_motion(
        output_motion,
        rest_rotations,
        animation_name=INPUT_MOTION_PATH.stem,
    )
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
