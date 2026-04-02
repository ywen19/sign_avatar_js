import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
HF_TOKEN = "your_hf_token_here"  # replace for first-time download

device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
tokenizer = None

DEFAULT_SYSTEM_PROMPT = (
    "/no_think "
    "You answer briefly. "
    "Use one short sentence. "
    "Do not explain unless asked."
)

IDENTITY_REPLY = (
    "I'm an AI text-based model. I can provide information and answer questions with BSL and texts."
)


def cleanup():
    global model, tokenizer

    print("\nCleaning up model and GPU memory...")

    try:
        if model is not None:
            del model
            model = None

        if tokenizer is not None:
            del tokenizer
            tokenizer = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("Cleanup done.")
    except Exception as e:
        print(f"Cleanup error: {e}")


def get_local_model_path(model_name: str, hf_token: str = None):
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


def load_model():
    global model, tokenizer

    local_model_path, was_cached = get_local_model_path(MODEL_NAME, HF_TOKEN)

    if was_cached:
        os.environ["HF_HUB_OFFLINE"] = "1"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=was_cached
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        local_files_only=was_cached
    ).to(device)

    model.eval()
    print(f"Model loaded on: {device}")


def classify_question_type(user_prompt: str) -> str:
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

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    result = tokenizer.decode(output_ids, skip_special_tokens=True).strip().upper()

    # print(f"[CLASSIFIER] {result}")
    return result


def ask_model(user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    result = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return result


def get_response(user_prompt: str) -> str:
    label = classify_question_type(user_prompt)

    if label.startswith("IDENTITY"):
        return IDENTITY_REPLY

    return ask_model(user_prompt)


def main():
    try:
        load_model()

        while True:
            user_prompt = input("\nEnter text (or 'quit'): ").strip()

            if user_prompt.lower() == "quit":
                print("Exiting...")
                break

            if not user_prompt:
                continue

            result = get_response(user_prompt)

            print("\nOUTPUT:")
            print(result)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Exiting...")
    finally:
        cleanup()


if __name__ == "__main__":
    main()