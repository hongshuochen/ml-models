#!/usr/bin/env bash
# Swap the Android golf model with a freshly-exported RAW-HEAD float16 TFLite — runs ENTIRELY on the
# offline box, nothing leaves it. Turns a detector best.pt into android-golf/.../assets/golf.tflite
# (the QNN-NPU raw-head format), then tells you how to rebuild + install.
#
# Prereq: `git pull` the repo first (gets the updated Kotlin decoder + the export script). The uv env
# in WORKDIR must already have torch + ultralytics (your training env). This script adds the export-only
# deps (onnx2tf/tensorflow) into that same .venv.
#
# Usage (edit the 3 paths, or pass them as env vars):
#   REPO=~/ml-models WEIGHTS=~/golf_offline/runs/detect/golf_ego_v4_hole/weights/best.pt \
#   WORKDIR=~/golf_offline  bash golf/deploy_golf_to_android.sh
set -euo pipefail

# ---- EDIT THESE (or set as env vars) --------------------------------------
REPO="${REPO:-$HOME/ml-models}"                                                    # cloned repo on this box
WORKDIR="${WORKDIR:-$HOME/golf_offline}"                                           # dir holding the uv .venv (torch+ultralytics)
WEIGHTS="${WEIGHTS:-$REPO/runs/detect/golf_ego_v4_hole/weights/best.pt}"           # detector best.pt to deploy
# ---------------------------------------------------------------------------

ASSETS="$REPO/android-golf/app/src/main/assets/golf.tflite"
SCRIPT="$REPO/golf/export_golf_rawhead_tflite.py"

for f in "$WEIGHTS" "$SCRIPT"; do
  [ -f "$f" ] || { echo "!! not found: $f"; exit 1; }
done
[ -d "$WORKDIR" ] || { echo "!! WORKDIR not found: $WORKDIR"; exit 1; }
cd "$WORKDIR"

echo ">> [1/3] ensuring export deps in $WORKDIR/.venv (onnx2tf 1.28.8 + tensorflow 2.19.0 = the proven combo)"
uv pip install -q \
  "onnx2tf==1.28.8" "tensorflow==2.19.0" tf_keras \
  onnx onnxsim onnx_graphsurgeon sng4onnx ai_edge_litert onnxruntime

echo ">> [2/3] backing up current model -> ${ASSETS}.bak"
cp "$ASSETS" "${ASSETS}.bak" 2>/dev/null || true

echo ">> [3/3] exporting $(basename "$WEIGHTS") -> assets/golf.tflite"
uv run python "$SCRIPT" --weights "$WEIGHTS" --out "$ASSETS"

cat <<EOF

============================================================================
✅ Model swapped. It self-verified above — you want to see:
     output: [[1,80,80,7],[1,40,40,7],[1,20,20,7]]   (7 = 4 box + 3 classes)
   (a 2-class model would show 6; if you see 6, LABELS in GolfDetector.kt must drop 'hole'.)
   >>> photograph that "output:" line for Claude to confirm. <<<

Now rebuild + install the app ON THIS BOX:
   cd "$REPO/android-golf"
   JAVA_HOME=/opt/android-studio/jbr ./gradlew :app:assembleDebug
   adb install -r app/build/outputs/apk/debug/app-debug.apk

In the app: ball=cyan, club_head=amber, hole=green; HUD shows "NPU • ~18 ms"
(first launch compiles ~1.4 s). To revert: cp "${ASSETS}.bak" "$ASSETS"
============================================================================
EOF
