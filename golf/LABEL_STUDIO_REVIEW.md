# Golf labeling — detector pre-label → 3 Label Studio projects (train / val / test)

Full-label workflow on the 3090 box (SRA0402): the trained detector pre-labels every sampled frame,
then a team reviews all of it in Label Studio, split by person into train/val/test projects. Fast +
stable (no SAM / OOM). ~15 annotators, 5 fps → ~10 h each.

Paths here: data `~/ml-models/data/golf` · detector `~/golf_offline/runs/detect/golf_ego_v5_nomined/weights/best.pt`.

## 1. Pre-label every frame with the detector (v5) @ 5 fps  (~1–2 h compute)
```bash
 cd ~/ml-models && git pull
~/ml-models/.venv/bin/python golf/detector_prelabel.py ~/ml-models/data/golf out_prelabel \
    --model ~/golf_offline/runs/detect/golf_ego_v5_nomined/weights/best.pt \
    --fps 5 --imgsz 1280
# -> out_prelabel/images/*.jpg + out_prelabel/labels/*.txt (YOLO box) + classes.txt
```
> Only frames with ≥1 detection are saved. For a thorough val/test GT (catch detector misses too),
> add `--keep-empty` — many more frames, so maybe only for val/test.

## 2. Person → split manifest (same pins as the frozen split)
```bash
~/ml-models/.venv/bin/python golf/split_golf_dataset.py ~/ml-models/data/golf --by minutes \
    --val-frac 0 --test-frac 0 \
    --pin-val  "Michael,Joy,Hiro,Ramu,Aryan" \
    --pin-test "Yujin,Alex,Kun,Madhu,AJ" \
    --out golf_split_manifest.csv
```

## 3. Split into 3 project task files
```bash
~/ml-models/.venv/bin/python golf/make_ls_projects.py out_prelabel --manifest golf_split_manifest.csv
# -> out_prelabel/ls_train.json, ls_val.json, ls_test.json, ls_config.xml
```

## 4. Start Label Studio with local-file serving pointed at out_prelabel
Isolated install (its Django deps clash with torch): `uv tool install label-studio`. DOCUMENT_ROOT
must be the **out_prelabel dir itself** (images live in its `images/` subdir). Put the env inline:
```bash
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/h.chen1/ml-models/out_prelabel \
label-studio start                       # http://localhost:8080
```

## 5. Create 3 projects — golf-train / golf-val / golf-test
For **each** project:
1. **Create Project** → name it.
2. **Settings → Labeling Interface → Code** → paste `out_prelabel/ls_config.xml` (3-class box).
3. **Import** → the matching `ls_<split>.json` via the **Import** button (NOT Sync — Sync makes dup, prediction-less tasks).
4. **Settings → Cloud Storage → Add Source Storage → Local files** → Absolute path =
   `/home/h.chen1/ml-models/out_prelabel/images` (a **subdir** of DOCUMENT_ROOT) → **Save, do NOT Sync**.
   - A **404** on images = this storage step missing / DOCUMENT_ROOT wrong. A **403** = serving not enabled.

## 6. Multi-annotator (15 people)
- **Organization → Members** → add the annotators' accounts.
- One project can be labeled by many people at once — LS hands each a different task.
- Suggested split of effort: 2–3 careful people on **val + test** (ground truth, ~11k frames each);
  the rest share **train** (~46k). The detector already drew the boxes → reviewers just delete FPs,
  add misses, fix boxes.
- **Class consistency rule** (decide once, apply everywhere): does `club_head` include clubs sitting
  in the bag, or only the active/in-hand club? Keep it uniform.

## 6b. Live progress website (everyone can watch)
A shared leaderboard page so the whole team sees who's labeled how much + overall %. Runs on this
box, polls the LS API in the background, serves an auto-refreshing HTML page. No token on the
viewers' side — the server holds it.
```bash
# get an API token in LS: avatar -> Account & Settings -> Access Token
~/ml-models/.venv/bin/python golf/ls_leaderboard_server.py \
    --url http://105.145.25.32:8080 --token <TOKEN> --project 1 --port 8090 --refresh 120
# team opens:  http://105.145.25.32:8090/       (add --project 1 2 3 once val/test exist)
```
Leave it running (e.g. `nohup … &` or a second terminal). CLI-only variant: `golf/ls_progress.py`.

## 7. Export → training
Each project → **Export → YOLO** → a dir with `images/` + `labels/`.
- **val / test** exports = the held-out eval sets (finally a real `hole` recall).
- **train** export = `--reviewed` source for `golf/build_and_train_golf.py` (see OFFLINE_PIPELINE.md).
