#!/usr/bin/env bash
# One-shot environment setup for the offline golf box (SRA0402, RTX 3090 24GB). Builds a STANDALONE
# uv venv OUTSIDE the repo — running uv *inside* the cloned repo re-syncs it to the repo's uv.lock on
# every call and silently reverts the GPU torch (undefined-symbol import errors). Pins the exact
# versions proven on this box for BOTH training and the raw-head->NPU tflite export. Idempotent.
#
# Run (from anywhere on SRA0402):
#     bash ~/ml-models/golf/setup_offline_env.sh
# Then use it as:  cd ~/golf_offline && uv run python ~/ml-models/golf/<script>.py ...
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/golf_offline}"        # standalone venv lives here, NOT in the repo
mkdir -p "$WORKDIR"; cd "$WORKDIR"
echo ">> setting up env in $WORKDIR (outside the repo, no uv re-sync)"

# 1) uv (self-contained; fetches its own Python). Skip if already present.
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# 2) Python 3.12 venv (uv grabs the interpreter itself)
[ -d .venv ] || uv venv --python 3.12

# 3) training + inference stack
echo ">> [1/3] ultralytics + opencv"
uv pip install -q ultralytics opencv-python

# 4) tflite export deps (raw-head -> QNN NPU). onnx2tf 1.28.8 + tensorflow 2.19.0 = the proven combo.
echo ">> [2/3] tflite export deps"
uv pip install -q "onnx2tf==1.28.8" "tensorflow==2.19.0" tf_keras \
  onnx onnxsim onnx_graphsurgeon sng4onnx ai_edge_litert onnxruntime

# 5) GPU torch LAST so nothing overrides it. cu121 matches this box's driver (torch 2.5.1 ran training).
echo ">> [3/3] GPU torch (cu121) — authoritative, installed last"
uv pip install -q "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121

# 6) verify — the line to photograph
echo; echo ">> verify:"
uv run python - <<'PY'
import torch, cv2, ultralytics, onnx2tf, tensorflow as tf
gpu = torch.cuda.is_available()
print(f"torch {torch.__version__} | GPU {gpu} | {torch.cuda.get_device_name(0) if gpu else 'NO CUDA'}")
print(f"ultralytics {ultralytics.__version__} | onnx2tf {onnx2tf.__version__} | tensorflow {tf.__version__} | opencv {cv2.__version__}")
print("READY ✅" if gpu else "GPU=False ❌  (driver/torch mismatch — photograph this for Claude)")
PY
echo
echo "============================================================"
echo "Use it as:  cd $WORKDIR && uv run python ~/ml-models/golf/<script>.py ..."
echo "  train:   ~/ml-models/golf/build_and_train_golf.py"
echo "  export:  ~/ml-models/golf/deploy_golf_to_android.sh"
echo "============================================================"
