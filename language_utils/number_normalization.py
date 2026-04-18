import re
from num2words import num2words


def digits_to_words(text: str) -> str:
    return " ".join(num2words(int(ch)) for ch in text)


def normalize_numeric_token(token: str) -> str:
    if token.isdigit():
        value = int(token)
        if value <= 500:
            return num2words(value).replace("-", " ")
        return digits_to_words(token)

    if re.search(r"[A-Za-z]", token) and re.search(r"\d", token):
        parts = re.findall(r"[A-Za-z]+|\d+", token)
        out = []

        for part in parts:
            if part.isdigit():
                out.append(digits_to_words(part))
            else:
                out.append(part)

        return " ".join(out)

    return token


def normalize_numbers_in_sentence(sentence: str) -> str:
    pattern = r"\b[A-Za-z]*\d+[A-Za-z\d]*\b"
    matches = re.findall(pattern, sentence)

    normalized_sentence = sentence
    for match in matches:
        normalized = normalize_numeric_token(match)
        normalized_sentence = normalized_sentence.replace(match, normalized, 1)

    return normalized_sentence


def main():
    test_sentences = [
        "I have 25 books.",
        "There are 371 people.",
        "It happened in 2024.",
        "Code BH12 is here.",
        "Room A7 please.",
        "I need 500 files.",
        "I need 501 files.",
        "Postcode BH12 3AB.",
    ]

    for sentence in test_sentences:
        normalized = normalize_numbers_in_sentence(sentence)
        print(f"INPUT : {sentence}")
        print(f"OUTPUT: {normalized}")
        print("-" * 50)


if __name__ == "__main__":
    main()