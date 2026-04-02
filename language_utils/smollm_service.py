import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
HF_TOKEN = None

DEFAULT_SYSTEM_PROMPT = (
    "/no_think "
    "You answer briefly. "
    "Use one short sentence. "
    "Do not explain unless asked."
)

IDENTITY_REPLY = (
    "I'm an AI text-based model. I can provide information and answer questions with BSL and texts."
)


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

    def get_local_model_path(self, model_name: str, hf_token: str = None):
        try:
            local_path = snapshot_download(
                repo_id=model_name,
                local_files_only=True
            )
            print(f"Model found in local cache: {local_path}")
            return local_path, True
        except Exception:
            print("Model not found in local cache. Downloading from Hugging Face...")

            if hf_token:
                os.environ["HF_TOKEN"] = hf_token

            local_path = snapshot_download(
                repo_id=model_name,
                local_files_only=False
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

    def get_response(self, user_prompt: str, conversation_history=None) -> str:
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