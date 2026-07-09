import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
MODEL_REVISION = "a07cc9a04f16550a088caea529712d1d335b0ac1"
HF_TOKEN = None

DEFAULT_SYSTEM_PROMPT = (
    "/no_think "
    "You answer briefly. "
    "Use one short sentence. "
    "Do not explain unless asked."
)

IDENTITY_REPLY = (
    "I'm an AI text-based model. I can provide information and answer questions with BSL and text."
)

SIGN_LANGUAGE_SCOPE_REPLY = "For now, I only support BSL."


class SmolLMService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SmolLMService, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls._instance

    def cleanup(self):
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

    def _missing_snapshot_files(self, local_path):
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

    def get_local_model_path(self, model_name: str, hf_token: str = None):
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

    def load_model(self):
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
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        messages = [
            {
                "role": "system",
                "content": (
                    "/no_think "
                    "You are a classifier. "
                    "Classify the user message into exactly one label: "
                    "IDENTITY, MEMORY, HOW_IT_WORKS, CAPABILITY, or OTHER. "
                    "IDENTITY means the user is asking what you are, who you are, whether you are AI, or whether you are human. "
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
        result = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip().upper()
        return result

    def classify_context_need(self, user_prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

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

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=500, #todo: is 200 suitable?
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        result = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip().upper()
        return result

    def ask_model(self, user_prompt: str, conversation_history=None) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        ]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_prompt})

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
        if self.is_sign_language_scope_question(user_prompt):
            return SIGN_LANGUAGE_SCOPE_REPLY

        label = self.classify_question_type(user_prompt)

        if label.startswith("IDENTITY"):
            return IDENTITY_REPLY

        return self.ask_model(user_prompt, conversation_history=conversation_history)


smollm_service = SmolLMService()


def load_model():
    smollm_service.load_model()


def get_response(user_prompt: str, conversation_history=None) -> str:
    return smollm_service.get_response(user_prompt, conversation_history=conversation_history)


def classify_context_need(user_prompt: str) -> str:
    return smollm_service.classify_context_need(user_prompt)


def cleanup():
    smollm_service.cleanup()