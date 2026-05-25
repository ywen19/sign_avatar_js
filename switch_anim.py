import json
from pathlib import Path


class TestAnimLoader:
    def __init__(self, default_json="Dancing_mixamo_com_frames.json", base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent
        else:
            base_dir = Path(base_dir).resolve()

        self.base_dir = base_dir
        self.default_json = default_json

    def _resolve_path(self, json_path: str) -> Path:
        path = Path(json_path)

        if path.is_absolute():
            return path

        return (self.base_dir / path).resolve()

    def _load_json(self, json_path: str):
        path = self._resolve_path(json_path)

        if not path.exists():
            raise FileNotFoundError(f"Missing JSON file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize_frames_data(self, frames_data, animation_name):
        if not isinstance(frames_data, dict):
            raise TypeError(f"Expected animation JSON object, got {type(frames_data).__name__}")

        if "bones" in frames_data:
            return frames_data

        normalized_bones = {}
        min_frame = None

        for bone_name, keyframes in frames_data.items():
            if not isinstance(keyframes, list):
                continue

            normalized_keyframes = []
            for keyframe in keyframes:
                if not isinstance(keyframe, dict):
                    continue

                if "frame" not in keyframe or "rotation" not in keyframe:
                    continue

                frame = int(keyframe["frame"])
                rotation = keyframe["rotation"]

                normalized_keyframes.append({
                    "f": frame,
                    "rot": rotation,
                })

                min_frame = frame if min_frame is None else min(min_frame, frame)

            if normalized_keyframes:
                normalized_bones[bone_name] = normalized_keyframes

        if min_frame is not None and min_frame != 0:
            for keyframes in normalized_bones.values():
                for keyframe in keyframes:
                    keyframe["f"] -= min_frame

        return {
            "name": animation_name,
            "fps": 30,
            "bones": normalized_bones,
        }

    def load_payload(self, json_path: str, animation_name=None, camera_state="start"):
        frames_data = self._load_json(json_path)

        if animation_name is None:
            animation_name = Path(json_path).stem

        frames_data = self._normalize_frames_data(frames_data, animation_name)

        return {
            "animation": animation_name,
            "camera": camera_state,
            "frames": frames_data,
            "source": str(self._resolve_path(json_path)),
        }

    def get_default_payload(self):
        return self.load_payload(
            self.default_json,
            animation_name="dance",
            camera_state="start",
        )
