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
WEIGHTS="${WEIGHTS:-$REPO/runs/detect/golf_ego_v5_nomined/weights/best.pt}"        # detector best.pt to deploy
VER="${VER:-}"                                                                     # version tag; auto-derived from WEIGHTS (e.g. v5) if empty
# ---------------------------------------------------------------------------

# derive a version tag (golf_ego_v5_nomined -> v5) so the asset is golf_v5.tflite; the app auto-loads
# the highest golf_v<N>.tflite and shows it in the HUD. Falls back to the legacy golf.tflite name.
[ -n "$VER" ] || VER="$(echo "$WEIGHTS" | grep -oE 'v[0-9]+' | head -1)"
ASSET_DIR="$REPO/android-golf/app/src/main/assets"
ASSET_NAME="${VER:+golf_$VER.tflite}"; ASSET_NAME="${ASSET_NAME:-golf.tflite}"
ASSETS="$ASSET_DIR/$ASSET_NAME"
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

echo ">> [2/3] clearing old golf*.tflite so the app loads exactly one (backed up to *.bak)"
for old in "$ASSET_DIR"/golf*.tflite; do
  [ -e "$old" ] || continue
  case "$old" in *.bak) continue;; esac
  mv -f "$old" "$old.bak"
done

echo ">> [3/3] exporting $(basename "$WEIGHTS") -> assets/$ASSET_NAME"
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

Shipped as: $ASSET_NAME  (the app auto-loads the highest golf_v<N>.tflite)
In the app: ball=cyan, club_head=amber, hole=green; HUD shows "NPU $VER • ~11 ms"
(first launch compiles ~1.4 s). Old models are kept as *.bak next to it.
============================================================================
EOF
