import numpy as np

from src.data_preprocessing import (
    bag_of_words,
    build_tag_index,
    build_vocab_for_language,
    tokenize,
)

# Sample data mimicking intents.json structure
SAMPLE_INTENTS = [
    {
        "tag": "greeting",
        "patterns": {
            "en": ["Hi", "Hello there", "Good morning"],
            "uk": ["Привіт", "Доброго ранку"],
        },
        "responses": {"en": ["Hello!", "Hi there!"], "uk": ["Привіт!", "Вітаю!"]},
    },
    {
        "tag": "goodbye",
        "patterns": {"en": ["Bye", "See you later"], "uk": ["Бувай", "До побачення"]},
        "responses": {
            "en": ["Goodbye!", "See you!"],
            "uk": ["До побачення!", "Бувай!"],
        },
    },
]


def test_tokenize():
    """Test the tokenize function."""
    assert tokenize("Hello there!") == ["hello", "there", "!"]
    assert tokenize("Привіт") == ["привіт"]
    assert tokenize("") == []
    assert tokenize("UPPERCASE words") == ["uppercase", "words"]


def test_build_tag_index():
    """Test building the tag-to-index mapping."""
    expected_tag_index = {"greeting": 0, "goodbye": 1}
    assert build_tag_index(SAMPLE_INTENTS) == expected_tag_index


def test_build_vocab_for_language_en():
    """Test building the English vocabulary."""
    expected_vocab_en = sorted(
        ["hi", "hello", "there", "good", "morning", "bye", "see", "you", "later"]
    )
    assert build_vocab_for_language(SAMPLE_INTENTS, "en") == expected_vocab_en


def test_build_vocab_for_language_uk():
    """Test building the Ukrainian vocabulary."""
    expected_vocab_uk = sorted(
        ["привіт", "доброго", "ранку", "бувай", "до", "побачення"]
    )
    assert build_vocab_for_language(SAMPLE_INTENTS, "uk") == expected_vocab_uk


def test_build_vocab_for_language_missing():
    """Test building vocabulary for a language with no patterns."""
    assert build_vocab_for_language(SAMPLE_INTENTS, "fr") == []


def test_bag_of_words():
    """Test the bag_of_words function."""
    vocab = ["a", "bag", "of", "words", "test"]
    tokens_present = ["bag", "of", "words"]
    tokens_missing = ["hello", "world"]
    tokens_mixed = ["a", "test", "hello"]
    tokens_empty = []

    expected_present = np.array([0, 1, 1, 1, 0])
    expected_missing = np.array([0, 0, 0, 0, 0])
    expected_mixed = np.array([1, 0, 0, 0, 1])
    expected_empty = np.array([0, 0, 0, 0, 0])

    assert np.array_equal(bag_of_words(tokens_present, vocab), expected_present)
    assert np.array_equal(bag_of_words(tokens_missing, vocab), expected_missing)
    assert np.array_equal(bag_of_words(tokens_mixed, vocab), expected_mixed)
    assert np.array_equal(bag_of_words(tokens_empty, vocab), expected_empty)
