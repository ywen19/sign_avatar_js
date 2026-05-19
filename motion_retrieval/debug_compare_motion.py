import json
from pathlib import Path

from motion_sequence_builder import temp_json_paths


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_num_frames(motion_json):
    first_bone = next(iter(motion_json.keys()))
    return len(motion_json[first_bone])


def get_bone_names(motion_json):
    return list(motion_json.keys())


def rotations_equal(rot_a, rot_b, eps=1e-6):
    if len(rot_a) != len(rot_b):
        return False

    return all(abs(a - b) <= eps for a, b in zip(rot_a, rot_b))


def compare_clip_segment(
    original_motion,
    final_motion,
    bone_names,
    final_start_idx,
    eps=1e-6,
):
    """
    Compare one original vocab motion against its corresponding segment
    inside the final concatenated motion.

    final_start_idx is 0-based.
    """
    num_frames = get_num_frames(original_motion)
    mismatches = []

    for local_idx in range(num_frames):
        final_idx = final_start_idx + local_idx

        for bone_name in bone_names:
            original_item = original_motion[bone_name][local_idx]
            final_item = final_motion[bone_name][final_idx]

            original_rot = original_item["rotation"]
            final_rot = final_item["rotation"]

            if not rotations_equal(original_rot, final_rot, eps=eps):
                mismatches.append(
                    {
                        "bone": bone_name,
                        "local_frame_index": local_idx,
                        "final_frame": final_item["frame"],
                        "original_rotation": original_rot,
                        "final_rotation": final_rot,
                    }
                )

    return mismatches


def collect_transition_debug(
    prev_motion,
    next_motion,
    final_motion,
    bone_names,
    prev_clip_final_end_idx,
    interp_start_idx,
    interpolation_frames,
):
    """
    Collect previous end frame, interpolation frames, and next start frame.

    Indices are 0-based.
    """
    transition_debug = {}

    for bone_name in bone_names:
        previous_end = final_motion[bone_name][prev_clip_final_end_idx]

        interp_items = [
            final_motion[bone_name][interp_start_idx + i]
            for i in range(interpolation_frames)
        ]

        next_start = final_motion[bone_name][
            interp_start_idx + interpolation_frames
        ]

        transition_debug[bone_name] = {
            "previous_motion_last_frame_original": prev_motion[bone_name][-1],
            "next_motion_first_frame_original": next_motion[bone_name][0],
            "previous_motion_last_frame_in_final": previous_end,
            "interpolated_frames_in_final": interp_items,
            "next_motion_first_frame_in_final": next_start,
        }

    return transition_debug


def debug_compare_concatenated_motion(
    json_paths,
    motion_json_path="motion.json",
    interpolation_frames=5,
    output_debug_path="motion_interpolation_debug.json",
    eps=1e-6,
):
    json_paths = list(json_paths)

    if not json_paths:
        raise ValueError("json_paths is empty.")

    original_motions = [load_json(path) for path in json_paths]
    final_motion = load_json(motion_json_path)

    bone_names = get_bone_names(original_motions[0])

    print("\n[ORIGINAL MOTION FILES]")
    for path, motion in zip(json_paths, original_motions):
        print(f"{path}: {get_num_frames(motion)} frames")

    print("\n[FINAL MOTION]")
    print(f"{motion_json_path}: {get_num_frames(final_motion)} frames")

    expected_total_frames = sum(get_num_frames(m) for m in original_motions)
    expected_total_frames += interpolation_frames * (len(original_motions) - 1)

    actual_total_frames = get_num_frames(final_motion)

    print("\n[FRAME COUNT CHECK]")
    print(f"Expected total frames: {expected_total_frames}")
    print(f"Actual total frames:   {actual_total_frames}")

    if expected_total_frames != actual_total_frames:
        raise ValueError(
            f"Total frame count mismatch: expected {expected_total_frames}, "
            f"got {actual_total_frames}"
        )

    all_mismatches = []
    transition_debug_all = {}

    final_cursor = 0

    for clip_idx, original_motion in enumerate(original_motions):
        clip_path = json_paths[clip_idx]
        clip_frame_count = get_num_frames(original_motion)

        print(
            f"\n[CLIP {clip_idx}] {clip_path}\n"
            f"Original frames: {clip_frame_count}\n"
            f"Final segment start frame: {final_motion[bone_names[0]][final_cursor]['frame']}"
        )

        mismatches = compare_clip_segment(
            original_motion=original_motion,
            final_motion=final_motion,
            bone_names=bone_names,
            final_start_idx=final_cursor,
            eps=eps,
        )

        if mismatches:
            print(f"Mismatch found: {len(mismatches)}")
            all_mismatches.extend(
                {
                    "clip_index": clip_idx,
                    "clip_path": clip_path,
                    **item,
                }
                for item in mismatches
            )
        else:
            print("Clip segment matches final motion.")

        clip_start_idx = final_cursor
        clip_end_idx = final_cursor + clip_frame_count - 1

        final_cursor += clip_frame_count

        if clip_idx < len(original_motions) - 1:
            interp_start_idx = final_cursor

            transition_name = (
                f"{Path(json_paths[clip_idx]).stem}"
                f"__to__"
                f"{Path(json_paths[clip_idx + 1]).stem}"
            )

            transition_debug_all[transition_name] = collect_transition_debug(
                prev_motion=original_motions[clip_idx],
                next_motion=original_motions[clip_idx + 1],
                final_motion=final_motion,
                bone_names=bone_names,
                prev_clip_final_end_idx=clip_end_idx,
                interp_start_idx=interp_start_idx,
                interpolation_frames=interpolation_frames,
            )

            print(
                f"Interpolation frames: "
                f"{final_motion[bone_names[0]][interp_start_idx]['frame']} "
                f"to "
                f"{final_motion[bone_names[0]][interp_start_idx + interpolation_frames - 1]['frame']}"
            )

            final_cursor += interpolation_frames

    debug_output = {
        "motion_json_path": motion_json_path,
        "interpolation_frames": interpolation_frames,
        "expected_total_frames": expected_total_frames,
        "actual_total_frames": actual_total_frames,
        "all_clip_segments_match": len(all_mismatches) == 0,
        "mismatches": all_mismatches,
        "transitions": transition_debug_all,
    }

    with open(output_debug_path, "w") as f:
        json.dump(debug_output, f, indent=2)

    print("\n[SUMMARY]")
    print(f"All clip segments match: {len(all_mismatches) == 0}")
    print(f"Mismatch count: {len(all_mismatches)}")
    print(f"Saved transition debug JSON to: {output_debug_path}")

    return debug_output


if __name__ == "__main__":
    debug_compare_concatenated_motion(
        json_paths=temp_json_paths,
        motion_json_path="motion.json",
        interpolation_frames=1,
        output_debug_path="motion_interpolation_debug.json",
    )