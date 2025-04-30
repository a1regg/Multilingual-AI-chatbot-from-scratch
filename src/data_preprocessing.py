import json
import os

import nltk
import numpy as np
from nltk.tokenize import word_tokenize

# Ensure the tokenizer is ready
nltk.download("punkt", quiet=False)
nltk.download("punkt_tab", quiet=False)
DATA_PATH: str = os.path.join("data", "intents.json")


def load_intents(path: str = DATA_PATH):
    """Load the bilingual intents JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["intents"]


def tokenize(text: str):
    """Lowercase and tokenize a string into words."""
    return word_tokenize(text.lower())


def build_tag_index(intents):
    """
    Map each unique intent tag to a sequential index 0..C-1,
    preserving first occurrence order.
    """
    unique_tags = list(dict.fromkeys(intent["tag"] for intent in intents))
    return {tag: idx for idx, tag in enumerate(unique_tags)}


def build_vocab_for_language(intents, lang_code: str):
    """
    Collect all tokens from patterns[intent][lang_code] into a sorted vocabulary.
    """
    vocab = set()
    for intent in intents:
        patterns = intent["patterns"].get(lang_code, [])
        for pat in patterns:
            vocab.update(tokenize(pat))
    return sorted(vocab)


def bag_of_words(tokens, vocab):
    """
    Create a binary Bag‑of‑Words vector.
    1 if vocab word is present in tokens, else 0.
    """
    return np.array([1 if word in tokens else 0 for word in vocab])


def prepare_data_for_language(intents, vocab, tag_to_idx, lang_code):
    """
    Build X (features) and y (labels) arrays for one language.
    """
    X, y = [], []
    for intent in intents:
        idx = tag_to_idx[intent["tag"]]
        for pat in intent["patterns"].get(lang_code, []):
            tokens = tokenize(pat)
            X.append(bag_of_words(tokens, vocab))
            y.append(idx)
    return np.array(X), np.array(y)
