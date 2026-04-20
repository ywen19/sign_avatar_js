from language_utils import *

MAX_HISTORY_MESSAGES = 15
TRIM_TO_MESSAGES = 6


def process_answer_text(answer_text: str):
    raw_sentences = break_into_sentences(answer_text)

    entities = []
    for sentence in raw_sentences:
        entities.extend(detect_entities(sentence))

    normalized_sentences = [
        normalize_sentence_for_match(sentence)
        for sentence in raw_sentences
    ]

    tokenized_sentences = [
        tokenize_with_entities(sentence, entities)
        for sentence in normalized_sentences
    ]

    traced_tokens = []
    reorder_tags = []
    reordered_tokens = []

    for tokens in tokenized_sentences:
        traced, tags = trace_tokens(tokens, entities)
        reordered = reorder_by_tags(traced, tags)

        traced_tokens.append(traced)
        reorder_tags.append(tags)
        reordered_tokens.append(reordered)

    return {
        "sentences": raw_sentences,
        "entities": entities,
        "normalized_sentences": normalized_sentences,
        "tokenized_sentences": tokenized_sentences,
        "traced_tokens": traced_tokens,
        "reorder_tags": reorder_tags,
        "reordered_tokens": reordered_tokens,
    }


def print_pipeline_debug(debug_data: dict):
    print("\n--- PIPELINE DEBUG ---")
    print("[SENTENCES]")
    for i, sentence in enumerate(debug_data["sentences"], 1):
        print(f"{i}. {sentence}")

    print("\n[ENTITIES]")
    print(debug_data["entities"])

    print("\n[NORMALIZED]")
    for item in debug_data["normalized_sentences"]:
        print(item)

    print("\n[TOKENIZED WITH ENTITIES]")
    for item in debug_data["tokenized_sentences"]:
        print(item)

    print("\n[TRACED TOKENS]")
    for item in debug_data["traced_tokens"]:
        print(item)

    print("\n[REORDER TAGS]")
    for item in debug_data["reorder_tags"]:
        print(item)

    print("\n[REORDERED TOKENS]")
    for item in debug_data["reordered_tokens"]:
        print(item)

    print("----------------------\n")


def main():
    conversation_history = []
    history_store = ChatHistoryStore("chat_history.jsonl")

    try:
        print("Loading SmolLM, GLiNER text analyzer, and vocab tree...")
        load_model()
        load_text_analyzer()
        load_vocab_tree(
            "./vocabs/all_vocabs.json",
            "./vocabs/all_vocabs_metadata.jsonl",
        )
        print("Ready.")
        print("Type 'quit' to exit.")
        print("Type ':history' to inspect recent in-memory history.")
        print("Type ':clear' to clear in-memory history.\n")

        while True:
            user_text = input("You: ").strip()

            if not user_text:
                continue

            if user_text.lower() == "quit":
                print("Exiting...")
                break

            if user_text == ":history":
                print("\n[RECENT IN-MEMORY HISTORY]")
                for msg in conversation_history:
                    print(f"{msg['role']}: {msg['content']}")
                print()
                continue

            if user_text == ":clear":
                conversation_history = []
                print("In-memory history cleared.\n")
                continue

            print(f"\n[TEXT INPUT] {user_text}")

            context_type = classify_context_need(user_text)
            print(f"[CONTEXT TYPE] {context_type}")

            if context_type == "SELF_CONTAINED":
                answer_text = get_response(user_text)

            elif context_type == "RECENT_CONTEXT":
                answer_text = get_response(
                    user_text,
                    conversation_history=conversation_history[-10:],
                )

            elif context_type == "ARCHIVE_CONTEXT":
                archived_messages = history_store.search_messages(
                    user_text,
                    limit=6,
                )
                combined_history = archived_messages + conversation_history[-10:]
                answer_text = get_response(
                    user_text,
                    conversation_history=combined_history,
                )

            else:
                answer_text = get_response(
                    user_text,
                    conversation_history=conversation_history[-10:],
                )

            print(f"\nAssistant: {answer_text}")

            debug_data = process_answer_text(answer_text)
            print_pipeline_debug(debug_data)

            user_message = {
                "role": "user",
                "content": user_text,
            }
            assistant_message = {
                "role": "assistant",
                "content": answer_text,
            }

            conversation_history.append(user_message)
            conversation_history.append(assistant_message)
            history_store.append_messages([user_message, assistant_message])

            if len(conversation_history) > MAX_HISTORY_MESSAGES:
                conversation_history = conversation_history[-TRIM_TO_MESSAGES:]

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Exiting...")

    finally:
        try:
            cleanup()
        except Exception as e:
            print(f"Cleanup failed: {e}")


if __name__ == "__main__":
    main()