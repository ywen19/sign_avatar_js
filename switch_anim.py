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

    def load_payload(self, json_path: str, animation_name=None, camera_state="start"):
        frames_data = self._load_json(json_path)

        if animation_name is None:
            animation_name = Path(json_path).stem

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