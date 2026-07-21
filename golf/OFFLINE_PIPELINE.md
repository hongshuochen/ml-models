# Offline golf pipeline (3-class: ball / club_head / hole)

Run-book for the **video machine** (Claude not available there). Label new footage and fine-tune
the 3-class detector offline. Everything here is self-contained: the standalone scripts + a uv env
(`ultralytics` + `opencv-python`) + a few carried-over files. GPU is used automatically if present;
CPU works, just slower. **Do the setup in a dir OUTSIDE the cloned repo** — see the env section.

## 0. What to copy onto the machine

Scripts (from `golf/`):
- `select_review_frames.py`  — pick the most useful frames to hand-label + write Label Studio tasks
- `mine_golf_videos.py`      — auto-mine high-confidence frames (incl hole) as extra train labels
- `build_and_train_golf.py`  — merge everything and fine-tune
- `annotate_status.py`       — (optional) sanity-check overlay video (boxes / MADE PUTT / ball speed)

Weights (git-ignored — copy the actual files):
- `runs/detect/golf_ego_v3_hole/weights/best.pt`  → the **3-class** model. Used for BOTH pre-labeling
  (`--model`) and as the fine-tune start (`--base-weights`). This one file is all you need.

Carry-over anchors (keep metrics comparable + prevent forgetting). **Copy them preserving the
`images/`+`labels/` layout** — e.g. into `~/golf_offline/data/` — and point `--val`/`--old` at the
`images/<split>` dir (see the path rule in step 3):
- `datasets/golf_ego_v1/{images,labels}/val`  → the **FIXED val** (`--val`, REQUIRED). It is ball+club
  only (no hole) — that's fine, see the note in step 3.
- `datasets/golf_ego_v2/{images,labels}/train` → hand-labeled ego train (`--old`, recommended).
  2-class ball/club; holes are legitimately absent there, so it is a correct negative for hole.

> ⚠️ Do **not** use `datasets/golf_hole` as `--old`: that set is the 1-class hole teacher where
> **class 0 = hole**, so its ids collide with ball. Only `golf_ego_v2` / `golf_ego_v1` are safe.

### Set up the environment with uv

uv is a single binary and can fetch its own Python, so the machine needs nothing pre-installed
except internet for the one-time downloads.

> ⚠️ **Work in a dir OUTSIDE the cloned repo** (e.g. `~/golf_offline`, NOT inside `ml-models/`).
> If you run uv inside the repo, `uv run` sees the repo's `pyproject.toml` and **re-syncs the venv
> to `uv.lock` on every call** (you'll see `Uninstalled N / Installed N` each run) — that silently
> reverts any torch you install and leaves a mismatched torch/nccl:
> `ImportError: …/libtorch_cuda.so: undefined symbol: ncclCommResume`. A standalone dir has no
> `pyproject.toml`, so `uv run` just uses its own `./.venv` and never re-syncs.

```bash
# 1) install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh   # Windows PS: irm https://astral.sh/uv/install.ps1 | iex
#    re-open the shell (or `source ~/.bashrc`) so `uv` lands on PATH

# 2) make a work dir OUTSIDE the repo, copy the 4 scripts (+ best.pt) into it
mkdir -p ~/golf_offline && cd ~/golf_offline
cp /path/to/ml-models/golf/{select_review_frames,mine_golf_videos,build_and_train_golf,annotate_status}.py .
#   also copy golf_ego_v3_hole's best.pt here (rename it best.pt to match the commands below)

# 3) MINIMAL env (uv grabs Python 3.12 itself)
uv venv --python 3.12
uv pip install ultralytics opencv-python

# 4) verify — no pyproject here, so uv run just uses ./.venv (no re-sync)
uv run python -c "import torch, cv2, ultralytics; print('torch', torch.__version__, 'GPU:', torch.cuda.is_available())"

# 5) sanity-check the model + that your video folder is reachable (catches wrong path/weights early)
uv run python -c "from ultralytics import YOLO; from pathlib import Path; \
m=YOLO('best.pt'); print('classes:', m.names); \
print('videos found:', len(list(Path('/path/to/new_videos').rglob('*.mp4'))))"
# expect  classes: {0:'ball',1:'club_head',2:'hole'}  and a non-zero video count
```

**If step 4 prints `GPU: False`** (or warns the NVIDIA driver is too old): the default torch wheel is
built for a newer CUDA than this machine's driver. Check the driver's CUDA version (`nvidia-smi`,
top-right), then install a matching torch **and torchvision together** — e.g. for a CUDA 12.1–12.5
driver:
```bash
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv run python -c "import torch; print('GPU:', torch.cuda.is_available())"   # expect True
```
Pin **both** (a torch/torchvision version mismatch also throws `undefined symbol` import errors).
No GPU / too much trouble? A CPU build works, just slower:
`uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

- Do **not** `uv sync` the whole repo (and don't run uv inside it) — that pulls the heavy
  training/export deps you don't need here, plus a torch build that may not match this driver.

## 1. Pick frames to hand-label  →  correct in Label Studio

```
uv run python select_review_frames.py /path/to/new_videos out_review \
    --model best.pt --total 500
```
Writes `out_review/images/*.jpg`, `out_review/labels/*.txt` (YOLO pre-labels), and
`out_review/ls_tasks.json` (Label Studio import with pre-annotations, **including hole**).

Set up Label Studio (isolated — its Django deps conflict with torch): `uv tool install label-studio`.
Start it with local-file serving pointed at **this batch's out dir** (so `?d=images/…` resolves —
`DOCUMENT_ROOT` must be the out dir itself, not its parent). Put the env vars INLINE so they can't
be lost to the wrong shell:
```bash
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/abs/path/to/out_review \
label-studio start          # http://localhost:8080
```
Then in the UI:
1. Create a project → Labeling Setup → **Code** → paste a 3-label RectangleLabels config
   (values `ball` / `club_head` / `hole`, `<Image name="image">` + `<RectangleLabels name="label">`).
2. **Import** `out_review/ls_tasks.json` via the **Import** button (NOT Sync).
3. **Add a Local Storage or the images 404** — the env vars alone are NOT enough; this LS version
   serves `/data/local-files/?d=…` only after a storage under the document root is registered:
   Settings → Cloud Storage → Add Source Storage → Local files → Absolute path =
   `/abs/path/to/out_review/images` (a **subdir** of DOCUMENT_ROOT) → Save, **do NOT Sync** (Sync
   creates duplicate prediction-less tasks). A plain **404** on the image URL = this step is missing
   (or DOCUMENT_ROOT is wrong); a 403 = serving not enabled.
4. Fix the boxes (the model is weakest on ball recall + the new hole class — the selector already
   favors those frames). **Label every visible hole** — a frame that shows a cup but leaves it
   unlabeled teaches "hole = background" (partial-label trap).
5. Export **YOLO** format → a dir with `images/` + `labels/`. Call it `out_review_corrected`.

The 3-class model emits hole directly, so no separate `--hole-model` teacher is needed. (The
`--hole-model` flag is the legacy bootstrap for when only a 2-class model existed.)

## 2. Auto-mine extra labels (optional but recommended)

```
uv run python mine_golf_videos.py /path/to/new_videos out_mined --model best.pt
```
Class-count-aware: mines ball, club **and** hole. It only keeps frames it can label *fully*
(confident + temporally supported detections; track-gap recovery + low-conf promotion are
zoom-verified), so mined putting frames get their cup recovered alongside the ball. Auto labels are
lower-trust than human ones — step 3 caps them to a fraction of the train set automatically.

## 3. Fine-tune the 3-class model

```
uv run python build_and_train_golf.py \
    --base-weights best.pt \
    --val      data/golf_ego_v1/images/val \
    --old      data/golf_ego_v2/images/train \
    --reviewed out_review_corrected \
    --mined    out_mined \
    --names ball,club_head,hole \
    --name golf_ego_v4_hole --imgsz 1280 --epochs 40 --batch 6
```
> **Path rule:** `--val` / `--old` must point at the `images/<split>` dir itself (e.g.
> `.../golf_ego_v1/images/val`, NOT the parent `golf_ego_v1/`) — labels resolve via Ultralytics'
> `images/`→`labels/` swap, so the path must contain `images/`. `--reviewed` / `--mined` point at
> the output dir (they already have an `images/` subdir). Trained weights land in
> `runs/detect/golf_ego_v4_hole/weights/best.pt`, relative to where you run the command.

**`--protect-hole` (use it if hole detection is weak):** auto-mined `hole` labels come from the
current (weak) detector and easily swamp + poison the few hand-labeled holes — mined frames on the
green that hide a hole the teacher missed teach "hole = background" (partial-label trap). With this
flag, every mined frame from a video that produced ANY hole is dropped, so `hole` is learned ONLY
from your trusted `--reviewed`/`--old` labels while mined still supplies ball/club from non-green
footage. Check the ratio first: `cat out_mined/labels/*.txt | awk '$1==2' | wc -l` (mined holes) vs
the same on `out_review_corrected` — if mined ≫ hand-labeled, turn it on and hand-label more holes.

Fixed val + old-set mix-in + capped auto labels + early stopping guard against drift. The report
prints per-class recall, e.g.:
```
  mAP50=0.83 mAP50-95=0.61  ball R=0.90  club_head R=0.88  hole R=n/a
  NOTE: the fixed val has NO 'hole' labels, so hole recall is not evaluated here...
```
`hole R=n/a` is expected — the carried val predates hole. To actually **track cup recall**, add a
few held-out putting clips (hand-labeled with holes) into the `--val` dir; then it prints a real
`hole R=…` and the note disappears. Otherwise you're flying blind on the new class.

Output weights: `runs/detect/golf_ego_v4_hole/weights/best.pt`.

## 4. (optional) Eyeball the result

```
uv run python annotate_status.py /path/to/a_clip.mp4 out.mp4 --model runs/detect/golf_ego_v4_hole/weights/best.pt
```
Overlays ball/club/hole boxes, trail, ball speed, and a MADE-PUTT / HITS tally — a fast way to
confirm the new model behaves before copying `best.pt` back for export/deployment.

## 5. Export the phone model (raw-head TFLite) + deploy

The Android app runs on the Qualcomm NPU, which needs the **raw one-to-many head** (the end-to-end
head's TopK/GatherNd can't be delegated). Don't use `yolo export` — use the committed script, which
bakes in the two tricks that keep the graph delegate-friendly (raw head + unrolled attention +
`onnx2tf enable_batchmatmul_unfold=False`):

```bash
uv pip install onnx onnxsim onnx2tf tensorflow      # once (heavier than the train env)
uv run python export_golf_rawhead_tflite.py \
    --weights runs/detect/golf_ego_v4_hole/weights/best.pt \
    --out golf.tflite
# verify it prints:  output: [[1,80,80,7],[1,40,40,7],[1,20,20,7]]  (7 = 4 box + 3 classes)
```

Then on the deploy machine: drop `golf.tflite` into `android-golf/app/src/main/assets/`, rebuild
(`JAVA_HOME=/opt/android-studio/jbr ./gradlew :app:assembleDebug`), `adb install` to the phone.
The Kotlin decoder is class-count-generic (reads `nc` from the tensor); `LABELS` in `GolfDetector.kt`
already lists `ball,club_head,hole`. `OverlayView` draws ball=cyan, club_head=amber, hole=green.

> The `golf.tflite` (~19 MB) is the ONE file that must leave this box for a phone test — it's a
> derived model, not source data. If it must stay in, do the Android build + `adb install` here too
> (the app source arrives via `git pull`). Either way, get the model onto the phone; there's no
> phone test without it.

**Shortcut:** you don't strictly need to export on this box. `best.pt` (~20 MB) is the minimal
thing to move out, and the deploy machine already has the export deps + the phone — running the
export there lets it be verified/parity-checked before shipping. `golf_ego_v3_hole/best.pt` already
lives on the deploy machine (a 3-class model ~= v4), so a first phone test needs nothing out at all.
