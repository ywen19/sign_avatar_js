"""
Run inside Blender with the target armature selected.

Removes Mixamo-style bone prefixes such as:
  mixamorig:Hips  -> Hips
  mixamorig4:Hips -> Hips
  mixamorig_LeftArm -> LeftArm
"""

import re

import bpy


PREFIX_RE = re.compile(r"^mixamorig\d*[:_.]?", re.IGNORECASE)


def remove_mixamorig_prefix_from_selected_armature():
    obj = bpy.context.object

    if obj is None or obj.type != "ARMATURE":
        raise RuntimeError("Select the armature object first.")

    armature = obj.data
    renamed = []
    skipped = []

    for bone in armature.bones:
        old_name = bone.name
        new_name = PREFIX_RE.sub("", old_name)

        if new_name == old_name:
            continue

        if not new_name:
            skipped.append((old_name, "empty name after prefix removal"))
            continue

        if new_name in armature.bones and new_name != old_name:
            skipped.append((old_name, f"target already exists: {new_name}"))
            continue

        bone.name = new_name
        renamed.append((old_name, new_name))

    print(f"Renamed {len(renamed)} bones.")
    for old_name, new_name in renamed:
        print(f"{old_name} -> {new_name}")

    if skipped:
        print(f"Skipped {len(skipped)} bones:")
        for old_name, reason in skipped:
            print(f"{old_name}: {reason}")


remove_mixamorig_prefix_from_selected_armature()
