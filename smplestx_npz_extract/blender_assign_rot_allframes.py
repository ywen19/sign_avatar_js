import bpy
import json
from mathutils import Quaternion

# ===== CONFIG =====
JSON_PATH = "/home/ywen/Desktop/SMPLest-X/a_billion_0_person00_smplx_quat_blender_names.json"
ARMATURE_OBJECT_NAME = "SMPLX-lh-female"
CLEAR_EXISTING_ROTATION = False
INSERT_KEYFRAMES = True
CLEAR_EXISTING_ANIMATION = False
USE_JSON_FRAME_DIRECTLY = True
FRAME_OFFSET = 0
# ==================


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def clear_armature_animation(arm_obj):
    if arm_obj.animation_data and arm_obj.animation_data.action:
        arm_obj.animation_data.action = None

    for pb in arm_obj.pose.bones:
        if pb.animation_data and pb.animation_data.action:
            pb.animation_data.action = None


def apply_animation_from_json(json_path, armature_name):
    data = load_json(json_path)

    arm_obj = bpy.data.objects.get(armature_name)
    if arm_obj is None:
        raise ValueError(f"Armature object not found: {armature_name}")

    if arm_obj.type != "ARMATURE":
        raise TypeError(f"Object is not an armature: {armature_name}")

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")

    pose_bones = arm_obj.pose.bones

    if CLEAR_EXISTING_ANIMATION:
        clear_armature_animation(arm_obj)

    if CLEAR_EXISTING_ROTATION:
        for pb in pose_bones:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))

    applied_bones = []
    missing_bones = []
    total_keyframes = 0
    min_frame = None
    max_frame = None

    for bone_name, frame_entries in data.items():
        if bone_name not in pose_bones:
            missing_bones.append(bone_name)
            continue

        if not isinstance(frame_entries, list):
            print(f"[WARN] Expected a list of frame entries for bone {bone_name}, got: {type(frame_entries)}")
            continue

        pb = pose_bones[bone_name]
        pb.rotation_mode = "QUATERNION"

        applied_this_bone = 0

        for entry in frame_entries:
            if not isinstance(entry, dict):
                print(f"[WARN] Invalid frame entry for bone {bone_name}: {entry}")
                continue

            if "frame" not in entry or "rotation" not in entry:
                print(f"[WARN] Missing 'frame' or 'rotation' in entry for bone {bone_name}: {entry}")
                continue

            frame_value = entry["frame"]
            quat_xyzw = entry["rotation"]

            if not isinstance(quat_xyzw, (list, tuple)) or len(quat_xyzw) != 4:
                print(f"[WARN] Invalid quaternion for bone {bone_name} at frame {frame_value}: {quat_xyzw}")
                continue

            try:
                frame_number = int(frame_value)
            except Exception:
                print(f"[WARN] Invalid frame number for bone {bone_name}: {frame_value}")
                continue

            if not USE_JSON_FRAME_DIRECTLY:
                frame_number = applied_this_bone + 1

            frame_number += FRAME_OFFSET

            x, y, z, w = quat_xyzw

            # Blender Quaternion order is (w, x, y, z)
            q_blender = Quaternion((w, x, y, z))
            q_blender.normalize()

            pb.rotation_quaternion = q_blender

            if INSERT_KEYFRAMES:
                pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)
                total_keyframes += 1

            applied_this_bone += 1

            if min_frame is None or frame_number < min_frame:
                min_frame = frame_number
            if max_frame is None or frame_number > max_frame:
                max_frame = frame_number

        if applied_this_bone > 0:
            applied_bones.append((bone_name, applied_this_bone))

    bpy.context.view_layer.update()

    if min_frame is not None and max_frame is not None:
        bpy.context.scene.frame_start = min_frame
        bpy.context.scene.frame_end = max_frame

    print(f"Applied animation to {len(applied_bones)} bones.")
    print(f"Inserted {total_keyframes} rotation keyframes.")

    if min_frame is not None and max_frame is not None:
        print(f"Frame range: {min_frame} -> {max_frame}")

    if applied_bones:
        print("Applied bones:")
        for bone_name, count in applied_bones:
            print(f" - {bone_name}: {count} keyframes")

    if missing_bones:
        print("[WARN] Missing bones in armature:")
        for name in missing_bones:
            print(" -", name)


apply_animation_from_json(JSON_PATH, ARMATURE_OBJECT_NAME)