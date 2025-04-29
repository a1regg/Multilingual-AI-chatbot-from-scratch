#!/bin/sh
set -euo

echo "⏳ [app] Training model…"
python3 src/training.py

echo "✅ [app] Training finished. Launching chat…"
exec python3 src/main.py
