import bpy
import json
from mathutils import Quaternion

# ===== CONFIG =====
JSON_PATH = "/home/ywen/Desktop/SMPLest-X/000001_person00_smplx_quat_blender_names.json"
ARMATURE_OBJECT_NAME = "SMPLX-lh-female"
CLEAR_EXISTING_ROTATION = False
INSERT_KEYFRAME = False
FRAME = bpy.context.scene.frame_current
# ==================


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def apply_pose_from_json(json_path, armature_name):
    data = load_json(json_path)

    arm_obj = bpy.data.objects.get(armature_name)
    if arm_obj is None:
        raise ValueError(f"Armature object not found: {armature_name}")

    if arm_obj.type != "ARMATURE":
        raise TypeError(f"Object is not an armature: {armature_name}")

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")

    pose_bones = arm_obj.pose.bones

    if CLEAR_EXISTING_ROTATION:
        for pb in pose_bones:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))

    applied = []
    missing = []

    for bone_name, quat_xyzw in data.items():
        if bone_name not in pose_bones:
            missing.append(bone_name)
            continue

        if not isinstance(quat_xyzw, (list, tuple)) or len(quat_xyzw) != 4:
            print(f"[WARN] Invalid quaternion for bone {bone_name}: {quat_xyzw}")
            continue

        x, y, z, w = quat_xyzw

        # Blender wants (w, x, y, z)
        q_blender = Quaternion((w, x, y, z))
        q_blender.normalize()

        pb = pose_bones[bone_name]
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = q_blender

        if INSERT_KEYFRAME:
            pb.keyframe_insert(data_path="rotation_quaternion", frame=FRAME)

        applied.append(bone_name)

    bpy.context.view_layer.update()

    print(f"Applied {len(applied)} bone rotations.")
    if missing:
        print("[WARN] Missing bones in armature:")
        for name in missing:
            print(" -", name)


apply_pose_from_json(JSON_PATH, ARMATURE_OBJECT_NAME)