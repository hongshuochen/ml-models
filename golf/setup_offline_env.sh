#!/usr/bin/env bash
# One-shot Python env for the offline golf box (SRA0402, RTX 3090, driver 555 / CUDA 12.5). Pins the
# exact versions proven here for BOTH training and the raw-head->NPU tflite export. Idempotent.
#
# Installs into a target venv chosen by $VENV (installs are pinned with `uv pip install --python`, so
# they land in exactly that venv regardless of CWD and NEVER trigger a uv project re-sync):
#
#   # (A) fix the REPO venv so you can work entirely inside ~/ml-models (no copying):
#   VENV=~/ml-models/.venv  bash ~/ml-models/golf/setup_offline_env.sh
#
#   # (B) or a standalone venv outside the repo (the old ~/golf_offline way):
#   bash ~/ml-models/golf/setup_offline_env.sh          # defaults to ~/golf_offline/.venv
#
# THE ONE RULE for the repo venv (A): activate it and use `python`, NEVER `uv run`/`uv sync` in the
# repo — those re-sync to uv.lock and revert the GPU torch. `python` (venv active) never re-syncs.
set -euo pipefail

VENV="${VENV:-$HOME/golf_offline/.venv}"
PY="$VENV/bin/python"
echo ">> target venv: $VENV"

# 1) uv (self-contained; fetches its own Python). Skip if present.
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# 2) create the venv (Python 3.12) if it doesn't exist yet
[ -x "$PY" ] || uv venv --python 3.12 "$VENV"

# 3) training + inference stack  (installs pinned to THIS venv; no project sync)
echo ">> [1/3] ultralytics + opencv"
uv pip install --python "$PY" -q ultralytics opencv-python

# 4) tflite export deps (raw-head -> QNN NPU). onnx2tf 1.28.8 + tensorflow 2.19.0 = the proven combo.
echo ">> [2/3] tflite export deps"
uv pip install --python "$PY" -q "onnx2tf==1.28.8" "tensorflow==2.19.0" tf_keras \
  onnx onnxsim onnx_graphsurgeon sng4onnx ai_edge_litert onnxruntime

# 5) GPU torch LAST so nothing overrides it. cu121 matches this box's CUDA 12.5 driver (proven: 2.5.1).
echo ">> [3/3] GPU torch (cu121) — authoritative, installed last"
uv pip install --python "$PY" -q "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121

# 6) verify — the line to photograph
echo; echo ">> verify:"
"$PY" - <<'PYEOF'
import torch, cv2, ultralytics, onnx2tf, tensorflow as tf
gpu = torch.cuda.is_available()
print(f"torch {torch.__version__} | GPU {gpu} | {torch.cuda.get_device_name(0) if gpu else 'NO CUDA'}")
print(f"ultralytics {ultralytics.__version__} | onnx2tf {onnx2tf.__version__} | tensorflow {tf.__version__} | opencv {cv2.__version__}")
print("READY ✅" if gpu else "GPU=False ❌  (photograph this for Claude)")
PYEOF

echo
echo "============================================================"
case "$VENV" in
  *ml-models*)
    echo "REPO venv ready. Work INSIDE ~/ml-models — no copying:"
    echo "    cd ~/ml-models && source .venv/bin/activate"
    echo "    python golf/build_and_train_golf.py ...        # use python, NOT 'uv run'"
    echo "  ⚠️  In the repo, NEVER 'uv run' / 'uv sync' — they revert the GPU torch." ;;
  *)
    echo "Standalone venv ready. Run repo scripts by path (no copying):"
    echo "    cd $(dirname "$VENV") && uv run python ~/ml-models/golf/<script>.py ..." ;;
esac
echo "============================================================"
