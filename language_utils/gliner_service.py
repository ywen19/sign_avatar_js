import gc
import os

import torch
from gliner import GLiNER
from huggingface_hub import snapshot_download


MODEL_NAME = "urchade/gliner_medium-v2.1"
HF_TOKEN = None


class GLiNERService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GLiNERService, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls._instance

    def cleanup(self):
        print("\nCleaning up GLiNER model and GPU memory...")

        try:
            if self.model is not None:
                del self.model
                self.model = None

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            print("GLiNER cleanup done.")
        except Exception as e:
            print(f"GLiNER cleanup error: {e}")

    def get_local_model_path(self, model_name: str, hf_token: str = None):
        try:
            local_path = snapshot_download(
                repo_id=model_name,
                local_files_only=True,
            )
            print(f"GLiNER model found in local cache: {local_path}")
            return local_path, True
        except Exception:
            print("GLiNER model not found in local cache. Downloading from Hugging Face...")

            if hf_token:
                os.environ["HF_TOKEN"] = hf_token

            local_path = snapshot_download(
                repo_id=model_name,
                local_files_only=False,
            )
            print(f"GLiNER model downloaded to local cache: {local_path}")
            return local_path, False

    def load_model(self):
        if self.model is not None:
            print("GLiNER model already loaded.")
            return

        local_model_path, was_cached = self.get_local_model_path(MODEL_NAME, HF_TOKEN)

        if was_cached:
            os.environ["HF_HUB_OFFLINE"] = "1"

        print("Loading GLiNER model...")
        self.model = GLiNER.from_pretrained(
            local_model_path,
            local_files_only=was_cached,
        )

        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        print(f"GLiNER model loaded on: {self.device}")

    def predict_entities(self, text: str, labels: list[str], threshold: float = 0.65):
        if self.model is None:
            raise RuntimeError("GLiNER model not loaded. Call load_model() first.")

        return self.model.predict_entities(
            text,
            labels,
            threshold=threshold,
        )


gliner_service = GLiNERService()


def load_gliner_model():
    gliner_service.load_model()


def predict_entities(text: str, labels: list[str], threshold: float = 0.65):
    return gliner_service.predict_entities(text, labels, threshold=threshold)


def cleanup_gliner():
    gliner_service.cleanup()