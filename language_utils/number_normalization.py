"""
Provides helper functions for converting numeric tokens in text into
spoken-word form.

Used before sign language motion generation, where numbers may need to 
be represented as words instead of raw digits. The module handles plain numeric
tokens, mixed letter-number tokens, room codes, postcodes, and other alphanumeric
expressions.

For smaller numbers, the full number is converted into words, such as "25" to
"twenty five". For larger numbers, each digit is converted separately, such as
"2024" to "two zero two four". Mixed tokens such as "BH12" are split into letter
and digit parts, and the digit part is converted per character.
"""


import re
from num2words import num2words


def digits_to_words(text: str) -> str:
    """
    Convert each digit in a string into its word form.

    text  (str)  :  string containing digit character

    Return:
    (str)  :  space-separated word form of each digit
    """
    return " ".join(num2words(int(ch)) for ch in text)


def normalize_numeric_token(token: str) -> str:
    """
    Normalize a numeric or alphanumeric token into word form.

    Plain numeric tokens up to 500 are converted as full numbers. Numeric tokens
    greater than 500 are converted digit by digit. Alphanumeric tokens are split
    into letter and digit parts, with digit parts converted digit by digit.

    token  (str)  :  text token (chunk) to normalize

    Return:
    (str)  :  normalized token with numbers converted into words
    """
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
    """
    Normalize all numeric and alphanumeric tokens in a sentence.

    This function finds tokens that contain at least one digit, normalizes each
    matched token with normalize_numeric_token(), and replaces it in the original
    sentence.

    sentence  (str)  :  input sentence to normalize

    Return:
    (str)  :  sentence with numeric and alphanumeric tokens converted into word form
    """
    pattern = r"\b[A-Za-z]*\d+[A-Za-z\d]*\b"
    matches = re.findall(pattern, sentence)

    normalized_sentence = sentence
    for match in matches:
        normalized = normalize_numeric_token(match)
        normalized_sentence = normalized_sentence.replace(match, normalized, 1)

    return normalized_sentence


def main():
    # mainly for local run test
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