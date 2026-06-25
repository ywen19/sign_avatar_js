import json
import sys
from pathlib import Path

import bpy
from mathutils import Quaternion


# Absolute defaults so this also works from Blender's Text Editor.
DEFAULT_JSON_PATH = Path("/home/ywen/Desktop/sign_avatar_js/vocabs/art_gallery_0_mixamo.json")
DEFAULT_ARMATURE_OBJECT_NAME = "Armature"

# The generated Mixamo JSON stores body and finger quaternions as [x, y, z, w].
JSON_QUATERNION_ORDER = "xyzw"

CLEAR_EXISTING_ANIMATION = True
CLEAR_POSE_BEFORE_APPLY = True
FRAME_OFFSET = 0


def parse_args():
    """Optional Blender args: -- <json_path> <armature_object_name>."""
    if "--" not in sys.argv:
        return DEFAULT_JSON_PATH, DEFAULT_ARMATURE_OBJECT_NAME

    extra = sys.argv[sys.argv.index("--") + 1 :]
    json_path = Path(extra[0]).expanduser() if len(extra) >= 1 else DEFAULT_JSON_PATH
    armature_name = extra[1] if len(extra) >= 2 else DEFAULT_ARMATURE_OBJECT_NAME
    return json_path, armature_name


def load_motion_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level object in {json_path}, got {type(data).__name__}")

    return data


def get_armature(name):
    armature = bpy.data.objects.get(name)
    if armature is None:
        available = [obj.name for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
        raise ValueError(f"Armature object not found: {name}. Available armatures: {available}")
    if armature.type != "ARMATURE":
        raise TypeError(f"Object is not an armature: {name} ({armature.type})")
    return armature


def clear_animation_and_pose(armature):
    if CLEAR_EXISTING_ANIMATION and armature.animation_data:
        armature.animation_data_clear()

    if CLEAR_POSE_BEFORE_APPLY:
        for pose_bone in armature.pose.bones:
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
            pose_bone.location = (0.0, 0.0, 0.0)
            pose_bone.scale = (1.0, 1.0, 1.0)


def as_blender_quaternion(values):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"Invalid quaternion: {values}")

    if JSON_QUATERNION_ORDER == "xyzw":
        x, y, z, w = values
        quat = Quaternion((w, x, y, z))
    elif JSON_QUATERNION_ORDER == "wxyz":
        w, x, y, z = values
        quat = Quaternion((w, x, y, z))
    else:
        raise ValueError(f"Unsupported quaternion order: {JSON_QUATERNION_ORDER}")

    quat.normalize()
    return quat


def keep_quaternion_continuity(quat, previous_quat):
    if previous_quat is not None and quat.dot(previous_quat) < 0.0:
        quat.negate()
    return quat


def apply_motion(data, armature):
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")

    clear_animation_and_pose(armature)

    pose_bones = armature.pose.bones
    applied_bones = []
    missing_bones = []
    skipped_entries = 0
    keyframe_count = 0
    min_frame = None
    max_frame = None

    for bone_name, entries in data.items():
        pose_bone = pose_bones.get(bone_name)
        if pose_bone is None:
            missing_bones.append(bone_name)
            continue

        if not isinstance(entries, list):
            print(f"[WARN] Bone {bone_name}: expected frame list, got {type(entries).__name__}")
            skipped_entries += 1
            continue

        pose_bone.rotation_mode = "QUATERNION"
        previous_quat = None
        applied_for_bone = 0

        for entry in entries:
            if not isinstance(entry, dict) or "frame" not in entry or "rotation" not in entry:
                print(f"[WARN] Bone {bone_name}: bad entry {entry}")
                skipped_entries += 1
                continue

            try:
                frame = int(entry["frame"]) + FRAME_OFFSET
                quat = as_blender_quaternion(entry["rotation"])
            except Exception as exc:
                print(f"[WARN] Bone {bone_name}: skipped entry {entry}: {exc}")
                skipped_entries += 1
                continue

            quat = keep_quaternion_continuity(quat, previous_quat)
            previous_quat = quat.copy()

            bpy.context.scene.frame_set(frame)
            pose_bone.rotation_quaternion = quat
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

            applied_for_bone += 1
            keyframe_count += 1
            min_frame = frame if min_frame is None else min(min_frame, frame)
            max_frame = frame if max_frame is None else max(max_frame, frame)

        if applied_for_bone:
            applied_bones.append((bone_name, applied_for_bone))

    if min_frame is not None and max_frame is not None:
        bpy.context.scene.frame_start = min_frame
        bpy.context.scene.frame_end = max_frame
        bpy.context.scene.frame_set(min_frame)

    bpy.context.view_layer.update()

    print("Mixamo body/finger motion applied")
    print(f"  Armature: {armature.name}")
    print(f"  Bones animated: {len(applied_bones)}")
    print(f"  Rotation keyframes: {keyframe_count}")
    print(f"  Frame range: {min_frame} -> {max_frame}")
    print(f"  Skipped entries: {skipped_entries}")

    if missing_bones:
        print(f"  Missing bones ({len(missing_bones)}):")
        for bone_name in missing_bones:
            print(f"    - {bone_name}")


def main():
    json_path, armature_name = parse_args()
    json_path = json_path.resolve()

    print(f"Loading Mixamo motion JSON: {json_path}")
    data = load_motion_json(json_path)
    armature = get_armature(armature_name)
    apply_motion(data, armature)


if __name__ == "__main__":
    main()
