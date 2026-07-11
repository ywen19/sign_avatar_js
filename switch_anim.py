"""
This module provides loading and normalizing local animation JSON files for the
sign avatar application frontend.

Animation files that already contain the frontend-compatible bones structure preserved. 
Quaternion JSON files stored by bone name are converted to the frontend animation 
structure, and their frame indices are rebased to start at zero.

The loaded animation is returned in an application payload containing the animation
name, camera state, normalized frame data, and resolved source file path.
"""


import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


class TestAnimLoader:
    def __init__(
        self,
        default_json: str = "Dancing_mixamo_com_frames.json",
        base_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent
        else:
            base_dir = Path(base_dir).resolve()

        self.base_dir = base_dir
        self.default_json = default_json

    def _resolve_path(self, json_path: str) -> Path:
        """
        Resolve an absolute animation JSON path against the loader base directory.

        json_path  (str)   :  animation JSON filename or path

        Return:
        (Path)             :  resolved absolute animation JSON path
        """
        path = Path(json_path)

        if path.is_absolute():
            return path

        return (self.base_dir / path).resolve()

    def _load_json(self, json_path: str) -> Any:
        """
        Load animation data from a local JSON file.

        json_path  (str)  :  animation JSON filename or path

        Return:
        (Any)             :  deserialized JSON animation data
        """
        path = self._resolve_path(json_path)

        if not path.exists():
            raise FileNotFoundError(f"Missing JSON file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize_frames_data(
        self, frames_data: Any, animation_name: str
    ) -> Dict[str, Any]:
        """
        Normalize loaded animation data to the frontend-compatible frame structure.

        Data containing a top-level bones key is already frontend-compatible and is 
        returned unchanged. 
        Otherwise, data stored as bone-name keyframe lists is converted from
        frame and rotation fields to f and rot fields. Nonzero frame indices are
        rebased so that the animation starts at frame zero.

        frames_data     (Any)  :  deserialized animation JSON data
        animation_name  (str)  :  name assigned to normalized animation data

        Return:
        (Dict[str, Any])       :  frontend-compatible animation frame data containing 
                                  the animation name, frame rate, and bone keyframes
        """
        if not isinstance(frames_data, dict):
            raise TypeError(
                f"Expected animation JSON object, got {type(frames_data).__name__}"
            )

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

    def load_payload(
        self,
        json_path: str,
        animation_name: Optional[str] = None,
        camera_state: str = "start",
    ) -> Dict[str, Any]:
        """
        Load an animation JSON file and build its application payload.

        If no animation name is provided, the source JSON filename stem is used. 
        The frame data is normalized before being added to the returned payload.

        json_path       (str)            :  animation JSON path
        animation_name  (Optional[str])  :  animation name included in the payload
        camera_state    (str)            :  camera state requested by the frontend;
                                            it is a placeholder to connect with 
                                            camera threadings in the future

        Return:
        (Dict[str, Any])                 :  application payload containing animation 
                                            name, camera state, normalized frames, 
                                            and source path
        """
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

    def get_default_payload(self) -> Dict[str, Any]:
        """
        Load the configured default animation and build its application payload.

        The default animation uses default_motion as its animation name and start 
        as its requested camera state.

        Return:
        (Dict[str, Any])  :  application payload for the configured default 
                             animation
        """
        return self.load_payload(
            self.default_json,
            animation_name="default_motion",
            camera_state="start",
        )
