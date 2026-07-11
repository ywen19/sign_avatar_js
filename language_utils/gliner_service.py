"""
GLiNER: https://github.com/urchade/GLiNER

GLiNER is a framework for training and deploying small Named Entity Recognition (NER) models 
with zero-shot capabilities.
This is mainly used since sign language syntax is different from spoken or written language, 
also for entities such as places, address, person names, BSL users tend to spell 
per-character wise, therefore they need to be identified when converting to motions for 
specific operations.

GLiNER service is built as singleton service class, 
meaning only one instance of this class exists for one main program running process. 

This singleton service ensures that only one GLiNER model instance is created and shared 
within the same process. It prevents the model from being loaded multiple times, 
reducing memory/GPU usage and providing a central place to load, run, and clean up the model.
"""


import gc
import os

import torch
from gliner import GLiNER
from huggingface_hub import snapshot_download

from typing import List, Dict, Tuple, Optional


# global variable for pretrained gliner version
MODEL_NAME: str = "urchade/gliner_medium-v2.1"
HF_TOKEN: Optional[str] = None


class GLiNERService:
    _instance = None  # for singleton service

    def __new__(cls):
        """
        For singleton service.
        Only create instance if no instance of this class exists;
        if exists, return the living instance.
        """
        if cls._instance is None:
            cls._instance = super(GLiNERService, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls._instance

    def cleanup(self) -> None:
        """
        Unloads the GLiNER model from the service and clears Python and CUDA memory.
        """
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

    def get_local_model_path(
        self, model_name: str, hf_token: str = None
    ) -> Tuple[str, bool]:
        """
        Locate the GLiNER model weight file in the local Hugging Face cache first, 
        and downloading it only if it is not already cached.

        Return:
        local_path  (str)   : path to the model files on machine
        was_cached  (bool)  : true if the model was already available locally, 
                              or False if it had to download it
        """
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

    def load_model(self) -> None:
        """
        Actual loading the model weight from local file to the GLiNER model.
        """
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

    def predict_entities(
        self, text: str, labels: list[str], threshold: float = 0.65
    ) -> List[Dict]:
        """
        Run entity prediction on input textual contents.
        text       (str)        :  input texts to extract entity from 
        labels     (List[str])  :  entity labels to detect; defined in ./text_analyzer.py
        threshold  (float)      :  minimum confidence score for a prediction to be returned

        Return:
            List[Dict]  :  Predicted entities with their text span, label, and confidence
        """
        if self.model is None:
            raise RuntimeError("GLiNER model not loaded. Call load_model() first.")

        return self.model.predict_entities(
            text,
            labels,
            threshold=threshold,
        )


# singleton instance initialization
gliner_service = GLiNERService()


# module-level wrappers provide a simple public API for loading, predicting,
# and cleaning up the singleton GLiNER service without exposing the service instance
def load_gliner_model() -> None:
    gliner_service.load_model()


def predict_entities(
    text: str, labels: list[str], threshold: float = 0.65
) -> List[Dict]:
    return gliner_service.predict_entities(text, labels, threshold=threshold)


def cleanup_gliner() -> None:
    gliner_service.cleanup()