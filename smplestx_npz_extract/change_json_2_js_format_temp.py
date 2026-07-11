"""
Testing module. 
This module converts Blender quaternion rotations from a flat JSON file into the animation JSON format 
used by the GLB model and frontend. 

The input quaternions use Blender's WXYZ component ordering: [w, x, y, z], which is gained from 
./axisangle_2_quaternion_frames.py.

They are first reordered to SciPy's XYZW convention: [x, y, z, w] 
A fixed root-space correction is then applied to every bone so that the rotations match the 
orientation of the target GLB skeleton. 

The converted animation is stored as: 
{ 
    "name": str, 
    "fps": int, 
    "bones": 
        { bone_name: 
            [
                {
                    "f": int, 
                    "rot": [x, y, z, w]
                } 
            ] 
        } 
} 
This script currently converts one rotation per bone and assigns it to the frame specified 
by FRAME_INDEX.
"""


import json
from scipy.spatial.transform import Rotation as R


INPUT_JSON_PATH = "./000001_person00_body_quat_blender_names.json"
OUTPUT_JSON_PATH = "./000001_person00_smplx_anim.json"

ANIM_NAME = "smplx"
FPS = 30
FRAME_INDEX = 0

# derived from your dump comparison ./:
# delta_root = q_glb_root_rest * inverse(q_blender_root_rest)
ROOT_DELTA_XYZW = [0.7071067811865476, 0.0, 0.0, -0.7071067811865476]
ROOT_DELTA = R.from_quat(ROOT_DELTA_XYZW)


def reorder_wxyz_to_xyzw(quat_wxyz):
    w, x, y, z = quat_wxyz
    return [x, y, z, w]


def convert_blender_quat_to_glb_quat(quat_wxyz):
    q_src_xyzw = reorder_wxyz_to_xyzw(quat_wxyz)
    r_src = R.from_quat(q_src_xyzw)

    # Apply the SAME root-space correction to every bone
    r_out = ROOT_DELTA * r_src

    return r_out.as_quat().tolist()

with open(INPUT_JSON_PATH, "r") as f:
    flat_data = json.load(f)

output_json = {
    "name": ANIM_NAME,
    "fps": FPS,
    "bones": {
        bone_name: [
            {
                "f": FRAME_INDEX,
                "rot": convert_blender_quat_to_glb_quat(quat_wxyz)
            }
        ]
        for bone_name, quat_wxyz in flat_data.items()
    }
}


with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(output_json, f, indent=2)

print(f"Saved to: {OUTPUT_JSON_PATH}")
print("ROOT_DELTA_XYZW =", ROOT_DELTA_XYZW)