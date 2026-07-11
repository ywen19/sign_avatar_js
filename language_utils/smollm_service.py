"""
SmoLLM: https://huggingface.co/collections/HuggingFaceTB/smollm

Provides a singleton service for loading and running the SmolLM text model.
It is used to generate short text responses, classify user question types, and decide
whether a message needs conversation context.

The singleton design ensures that only one model and tokenizer instance are loaded within
the same process, reducing repeated loading and memory/GPU usage.
"""


import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from typing import List, Dict, Tuple, Optional


# global variable for pretrained smollm version
MODEL_NAME: str = "HuggingFaceTB/SmolLM3-3B"
# pin the model to a specific Hugging Face revision
# avoids fetching newer versions unexpectedly, which can cause slow downloads
# or inconsistent model behaviour
MODEL_REVISION: str = "a07cc9a04f16550a088caea529712d1d335b0ac1"
HF_TOKEN: Optional[str] = None

# default system prompt used to keep model responses short and direct
DEFAULT_SYSTEM_PROMPT: str = (
    "/no_think "
    "You answer briefly. "
    "Use one short sentence. "
    "Do not explain unless asked."
)

# fixed response for identity-related questions
IDENTITY_REPLY: str = (
    "I'm an AI text-based model. "
    "I can provide information and answer questions with BSL and text."
)
# fixed response for sign language scope questions
SIGN_LANGUAGE_SCOPE_REPLY: str = "For now, I only support BSL."


class SmolLMService:
    _instance = None  # for singleton service

    def __new__(cls):
        """
        For singleton service.
        Only create instance if no instance of this class exists;
        if exists, return the living instance.
        """
        if cls._instance is None:
            cls._instance = super(SmolLMService, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls._instance

    def cleanup(self) -> None:
        """
        Release the loaded SmolLM model, tokenizer, and unused CPU/GPU memory.

        This method removes the model and tokenizer from the service, runs Python
        garbage collection, and clears unused CUDA memory if GPU is available.
        """
        print("\nCleaning up model and GPU memory...")

        try:
            if self.model is not None:
                del self.model
                self.model = None

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            print("Cleanup done.")
        except Exception as e:
            print(f"Cleanup error: {e}")

    def _missing_snapshot_files(self, local_path: str) -> List[str]:
        """
        Check whether a local Hugging Face model snapshot is missing required weight files.

        This helper reads the model.safetensors.index.json file, finds all weight files
        required by the model, and returns any files that are not present in the local
        snapshot directory.

        local_path  (str)  :  local path to the cached model snapshot

        Return:
        list[str]  :  Names of missing weight files, returns an empty list if the
                      snapshot appears complete or cannot be validated.
        """
        local_path = os.fspath(local_path)
        index_path = os.path.join(local_path, "model.safetensors.index.json")

        if not os.path.exists(index_path):
            return []

        try:
            import json

            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        except Exception as e:
            print(f"Could not validate model index: {e}")
            return []

        required_files = set(index_data.get("weight_map", {}).values())
        return [
            file_name
            for file_name in sorted(required_files)
            if not os.path.exists(os.path.join(local_path, file_name))
        ]

    def get_local_model_path(
        self, model_name: str, hf_token: str = None
    ) -> Tuple[str, bool]:
        """
        Locate the local Hugging Face model snapshot, downloading it if needed.

        This method first checks whether the pinned model revision exists in the local
        Hugging Face cache. If the cached snapshot exists, it also validates that all
        required weight files are present. If the snapshot is missing or incomplete,
        the method downloads the model files from Hugging Face.

        model_name  (str)  :  hugging Face model repository name
        hf_token    (str)  :  optional Hugging Face token for restricted models

        Return:
        local_path   (str)   :  local model path
        was_chached  (bool)  :  true if the model was already available locally, 
                                or False if it had to be downloaded
        """
        try:
            local_path = snapshot_download(
                repo_id=model_name,
                revision=MODEL_REVISION,
                local_files_only=True
            )
            missing_files = self._missing_snapshot_files(local_path)

            if not missing_files:
                print(f"Model found in local cache: {local_path}")
                return local_path, True

            print("Local model cache is incomplete. Missing files:")
            for file_name in missing_files:
                print(f"- {file_name}")
            print("Downloading missing model files from Hugging Face...")
        except Exception:
            print("Model not found in local cache. Downloading from Hugging Face...")

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        os.environ.pop("HF_HUB_OFFLINE", None)

        local_path = snapshot_download(
            repo_id=model_name,
            revision=MODEL_REVISION,
            local_files_only=False
        )

        missing_files = self._missing_snapshot_files(local_path)
        if missing_files:
            missing = ", ".join(missing_files)
            raise FileNotFoundError(
                f"Model download/cache is still incomplete. Missing: {missing}"
            )

        print(f"Model downloaded to local cache: {local_path}")
        return local_path, False

    def load_model(self) -> None:
        """
        Load the SmolLM tokenizer and model into memory.

        First locates or downloads the pinned Hugging Face model snapshot.
        It then loads the tokenizer and causal language model from that local path,
        moves the model to the selected device, and sets it to evaluation mode.
        """
        if self.model is not None and self.tokenizer is not None:
            print("Model already loaded.")
            return

        local_model_path, was_cached = self.get_local_model_path(MODEL_NAME, HF_TOKEN)

        if was_cached:
            os.environ["HF_HUB_OFFLINE"] = "1"

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            local_files_only=was_cached
        )

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            local_files_only=was_cached
        ).to(self.device)

        self.model.eval()
        print(f"Model loaded on: {self.device}")

    def classify_question_type(self, user_prompt: str) -> str:
        """
        Classify the user's prompt into a predefined question type.

        Mainly used to detect identity-related questions so the service can
        return a fixed project-specific identity reply. Other question types are
        currently passed to normal model response generation.

        user_prompt  (str)  :  user input text

        Return:
        result  (str)  :  predicted question type label, such as "IDENTITY", 
                          "CAPABILITY", or "OTHER". Now only "IDENTITY" goes to 
                          fixed reply
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # classifier prompt that asks the model to return only one question-type label
        messages = [
            {
                "role": "system",
                "content": (
                    "/no_think "
                    "You are a classifier. "
                    "Classify the user message into exactly one label: "
                    "IDENTITY, MEMORY, HOW_IT_WORKS, CAPABILITY, or OTHER. "
                    "IDENTITY means the user is asking what you are, who you are, whether you are AI, "
                    "or whether you are human. "
                    "MEMORY means the user is asking what you remember or whether you have long context memory. "
                    "HOW_IT_WORKS means the user is asking how your answers are produced, based on code, model, or data. "
                    "CAPABILITY means the user is asking what you can do. "
                    "Reply with only one label."
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        # convert the chat messages into the smollm's expected input format
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            # generated_ids is the raw token IDs output by the model
            # include [[input prompt tokens] + [newly generated output tokens], ...]
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # we only need output
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        # decode the generated tokens into a normalized uppercase classification label
        result = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip().upper()
        return result

    def classify_context_need(self, user_prompt: str) -> str:
        """
        Classify whether the user's prompt needs conversation context.

        This method uses the loaded SmolLM model with a classifier prompt to decide
        whether the message is self-contained, depends on recent conversation context,
        or refers to older archived conversation.

        user_prompt  (str)  :  user input text

        Return:
        result  (str)  :  Predicted context-need label, such as "SELF_CONTAINED",
                          "RECENT_CONTEXT", or "ARCHIVE_CONTEXT"
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # classifier prompt that asks the model to return only one context-need label
        messages = [
            {
                "role": "system",
                "content": (
                    "/no_think "
                    "You are a classifier. "
                    "Classify the user's message into exactly one label: "
                    "SELF_CONTAINED, RECENT_CONTEXT, or ARCHIVE_CONTEXT. "
                    "SELF_CONTAINED means the message can be answered without prior conversation. "
                    "RECENT_CONTEXT means the message depends on recent chat context such as pronouns or follow-ups. "
                    "ARCHIVE_CONTEXT means the message refers to older conversation that may not be in recent context, "
                    "such as 'earlier', 'before', 'previously', or past recommendations. "
                    "Reply with only one label."
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        # convert the chat messages into the smollm's expected input format
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        result = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip().upper()
        return result

    def ask_model(self, user_prompt: str, conversation_history=None) -> str:
        """
        Generate a text response from the SmolLM model.

        Builds a chat-style prompt using the default system prompt, optional
        conversation history, and the current user message. It then generates and decodes
        the model response.

        user_prompt           (str)                          :  user input text
        conversation_history  (list[dict[str, str]] | None)  :  Optional previous chat messages
                                                                to include as context

        Return:
        result  (str)  :  generated model response
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        ]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_prompt})

        # convert the chat messages into the smollm's expected input format
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        result = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return result

    def is_sign_language_scope_question(self, user_prompt: str) -> bool:
        """
        Check whether the user is asking about supported sign languages.

        Rule-based check detects prompts that mention sign language or BSL and
        ask about language support, knowledge, or scope. It is used to return a fixed
        response before normal model generation.

        user_prompt  (str)  :  user input text

        Return:
        (bool)  :  true if the prompt asks about supported sign languages, otherwise False
        """
        prompt = user_prompt.lower()

        mentions_sign_language = (
            "sign language" in prompt
            or "bsl" in prompt
            or "british sign language" in prompt
        )
        asks_knowledge_or_scope = any(
            phrase in prompt
            for phrase in (
                "what sign language",
                "which sign language",
                "what signed language",
                "which signed language",
                "do you know",
                "can you do",
                "can you understand",
                "can you use",
                "besides british sign language",
                "beside british sign language",
                "besides bsl",
                "beside bsl",
                "other than british sign language",
                "other than bsl",
                "apart from british sign language",
                "apart from bsl",
            )
        )

        return mentions_sign_language and asks_knowledge_or_scope

    def get_response(self, user_prompt: str, conversation_history=None) -> str:
        """
        Main response wrapper for the service.

        First checks whether the prompt should receive a fixed project-specific answer, 
        such as sign language scope or identity replies. If no fixed answer is needed, 
        it passes the prompt and optional conversation history to the SmolLM model for 
        general dialogue generation.

        user_prompt           (str)                          :  user input text
        conversation_history  (list[dict[str, str]] | None)  :  Optional previous chat messages
                                                                to include as context

        Return:
        (str)  :  generated model response
        """
        if self.is_sign_language_scope_question(user_prompt):
            return SIGN_LANGUAGE_SCOPE_REPLY

        label = self.classify_question_type(user_prompt)

        if label.startswith("IDENTITY"):
            return IDENTITY_REPLY

        return self.ask_model(user_prompt, conversation_history=conversation_history)



# singleton instance initialization
smollm_service = SmolLMService()


# module-level wrappers provide a simple public API for loading, predicting,
# and cleaning up the singleton SmoLLM service without exposing the service instance
def load_model() -> None:
    smollm_service.load_model()


def get_response(user_prompt: str, conversation_history=None) -> str:
    return smollm_service.get_response(user_prompt, conversation_history=conversation_history)


def classify_context_need(user_prompt: str) -> str:
    return smollm_service.classify_context_need(user_prompt)


def cleanup() -> None:
    smollm_service.cleanup()
