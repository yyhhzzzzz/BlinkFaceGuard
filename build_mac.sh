#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

python3 -m pip install -r requirements_app.txt

rm -rf build dist

pyinstaller \
  --noconfirm \
  --windowed \
  --name BlinkFilterGUI \
  --collect-all ultralytics \
  --collect-all mediapipe \
  --collect-all cv2 \
  --hidden-import pandas \
  --hidden-import tqdm \
  --add-data "models:models" \
  gui_blink_filter.py

echo "Build finished: dist/BlinkFilterGUI.app"
