import os
import sys

import mlflow.pytorch
import numpy as np
import torch
from langdetect import LangDetectException, detect
from PyQt6.QtWidgets import QApplication, QLineEdit, QTextEdit, QVBoxLayout, QWidget

import config
from data_preprocessing import (
    bag_of_words,
    build_tag_index,
    build_vocab_for_language,
    load_intents,
)


class ChatGUI(QWidget):
    def __init__(self, model_en, model_uk, vocab_en, vocab_uk, tag_to_idx, intents):
        super().__init__()
        self.models = {"en": model_en, "uk": model_uk}
        self.vocabs = {"en": vocab_en, "uk": vocab_uk}
        self.tag_to_idx = tag_to_idx
        self.idx_to_tag = {i: t for t, i in tag_to_idx.items()}
        self.intents = intents
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Bob")
        layout = QVBoxLayout()
        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        self.entry = QLineEdit()
        self.entry.setPlaceholderText("Type in English or Ukrainian…")
        self.entry.returnPressed.connect(self.process_input)
        layout.addWidget(self.chat_log)
        layout.addWidget(self.entry)
        self.setLayout(layout)

    def process_input(self):
        text = self.entry.text().strip()
        if not text:
            return
        self.entry.clear()
        self.display("You", text)

        try:
            lang = detect(text)
        except LangDetectException:
            lang = "en"
        if lang not in ("en", "uk"):
            lang = "en"

        resp = self.get_response(text, lang)
        self.display("Bob", resp)

    def get_response(self, text: str, lang: str) -> str:
        model = self.models[lang]
        vocab = self.vocabs[lang]
        device = next(model.parameters()).device

        vec = bag_of_words(text.lower().split(), vocab)
        tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
        tag = self.idx_to_tag[out.argmax(dim=1).item()]

        # pick a response in the same language (fallback to English)
        for intent in self.intents:
            if intent["tag"] == tag:
                return np.random.choice(
                    intent["responses"].get(lang, intent["responses"]["en"])
                )
        return "No matching intent."

    def display(self, sender: str, msg: str) -> None:
        self.chat_log.append(f"<b>{sender}:</b> {msg}")


def run():
    # Load intents and vocabularies
    intents = load_intents()
    tag_to_idx = build_tag_index(intents)

    # First get the experiment ID
    experiment_id = config.get_or_create_experiment_id(config.MLFLOW_EXPERIMENT_NAME)
    # Then get the latest run ID using the experiment ID
    run_id = os.getenv("MLFLOW_RUN_ID") or config.get_latest_run_id(experiment_id)
    if not run_id:
        raise RuntimeError(
            "Could not determine MLflow run ID. "
            "Set MLFLOW_RUN_ID or check your tracking server."
        )  # explicit failure

    # Load the two models from MLflow
    model_en = mlflow.pytorch.load_model(f"runs:/{run_id}/model_en")
    model_uk = mlflow.pytorch.load_model(f"runs:/{run_id}/model_uk")

    vocab_en = build_vocab_for_language(intents, "en")
    vocab_uk = build_vocab_for_language(intents, "uk")

    # Launch the GUI
    app = QApplication(sys.argv)
    window = ChatGUI(model_en, model_uk, vocab_en, vocab_uk, tag_to_idx, intents)
    window.resize(600, 400)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
