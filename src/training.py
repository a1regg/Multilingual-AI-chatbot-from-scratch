import os
from typing import Any, Dict, List

import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim

import config
from data_preprocessing import (
    build_tag_index,
    build_vocab_for_language,
    load_intents,
    prepare_data_for_language,
)
from model import ChatBotModel

EPOCHS: int = 100
LR: float = 0.01
HIDDEN_SIZE: int = 8


def train_lang(intents: List[Dict[str, Any]], lang: str, tag_to_idx: Dict[str, int]):
    """
    Train a chatbot model for a single language.

    Args:
        intents: List of intent dicts.
        lang: Language code ('en' or 'uk').
        tag_to_idx: Mapping from intent tags to integer labels.

    Returns:
        (model, vocab) or (None, vocab) if no patterns for that language.
    """
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    vocab = build_vocab_for_language(intents, lang)
    X, y = prepare_data_for_language(intents, vocab, tag_to_idx, lang)
    if X.size == 0:
        return None, vocab

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    model = ChatBotModel(len(vocab), HIDDEN_SIZE, len(tag_to_idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"[{lang}] Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

    return model, vocab


def main() -> None:
    """
    Load intents, train English and Ukrainian models, log artifacts to MLflow.
    """
    intents = load_intents()
    tag_to_idx = build_tag_index(intents)
    DATA_DIR = "data"
    # MLflow config already applied in config.py
    with config.start_run(run_name="Bilingual_Models"):
        for lang in ("en", "uk"):
            model, vocab = train_lang(intents, lang, tag_to_idx)
            if model:
                # Log the PyTorch model
                mlflow.pytorch.log_model(model, f"model_{lang}")

                # Write and log vocabulary
                vocab_path = os.path.join(DATA_DIR, f"vocab_{lang}.txt")
                with open(vocab_path, "w", encoding="utf-8") as vf:
                    vf.write("\n".join(vocab))
                mlflow.log_artifact(vocab_path)

        # Write and log tag-to-index mapping
        tag_map_path = os.path.join(DATA_DIR, "tag_to_idx.txt")
        with open(tag_map_path, "w", encoding="utf-8") as tf:
            for tag, idx in tag_to_idx.items():
                tf.write(f"{tag}:{idx}\n")
        mlflow.log_artifact(tag_map_path)

    print("✅ Training complete. Models and artifacts logged to MLflow.")


if __name__ == "__main__":
    main()
