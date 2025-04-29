import mlflow.pytorch
import numpy as np
import torch
from langdetect import LangDetectException, detect

import config
from data_preprocessing import (
    bag_of_words,
    build_tag_index,
    build_vocab_for_language,
    load_intents,
)


def load_models(run_id):
    """Load English & Ukrainian models logged under the same run."""
    print(f"Loading models from run ID: {run_id}")
    en = mlflow.pytorch.load_model(f"runs:/{run_id}/model_en")
    uk = mlflow.pytorch.load_model(f"runs:/{run_id}/model_uk")
    return en, uk


def chat_loop(model_en, model_uk, vocab_en, vocab_uk, idx_to_tag, intents):
    """Interactive console chat."""
    dev_en = next(model_en.parameters()).device
    dev_uk = next(model_uk.parameters()).device
    print("Type 'quit' to exit.")
    while True:
        text = input("You: ").strip()
        if text.lower() == "quit":
            break
        try:
            lang = detect(text)
        except LangDetectException:
            lang = "en"
        if lang not in ("en", "uk"):
            lang = "en"
        model, vocab, dev = (
            (model_uk, vocab_uk, dev_uk)
            if lang == "uk"
            else (model_en, vocab_en, dev_en)
        )
        vec = bag_of_words(text.lower().split(), vocab)
        with torch.no_grad():
            out = model(torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(dev))
        tag = idx_to_tag[out.argmax(dim=1).item()]
        for intent in intents:
            if intent["tag"] == tag:
                resp = intent["responses"].get(lang) or intent["responses"]["en"]
                print("Bob:", np.random.choice(resp))
                break


def main():
    intents = load_intents()
    tag_to_idx = build_tag_index(intents)
    idx_to_tag = {i: t for t, i in tag_to_idx.items()}
    vocab_en = build_vocab_for_language(intents, "en")
    vocab_uk = build_vocab_for_language(intents, "uk")

    # 1. Get the Experiment ID using the experiment name
    experiment_id = config.get_or_create_experiment_id(config.MLFLOW_EXPERIMENT_NAME)
    # 2. Get the latest Run ID using the integer experiment_id
    run_id = config.get_latest_run_id(experiment_id)

    model_en, model_uk = load_models(run_id)

    chat_loop(model_en, model_uk, vocab_en, vocab_uk, idx_to_tag, intents)


if __name__ == "__main__":
    main()
