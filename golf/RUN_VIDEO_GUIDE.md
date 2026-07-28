# Run the golf detector on a video → annotated video (shareable guide)

Give a video, get back the same video with **ball / club_head / hole** boxes drawn. Two ways:
- **A — `.pt` model** (recommended, simplest): runs the trained detector via Ultralytics.
- **B — `.tflite` model**: runs the *exact phone model* (matches what the on-device app draws).

You need **Python 3.10+**, the repo scripts, and a model file. Nothing here is tied to any one machine —
you make your own environment and just call `python`.

---

## 1. Get the scripts + a model

```bash
git clone <this-repo-url> ml-models      # or copy the golf/ folder
cd ml-models
```

You also need a **model file** (ask whoever shared this):
- Path A wants a `best.pt` (e.g. `golf_ego_v5.pt`).
- Path B wants a `golf_v5.tflite` (the phone model).

Put the model anywhere and point `--model` at it.

---

## 2. Make an environment (once)

Pick ONE. Both create an isolated env so nothing touches your system Python.

**Plain venv + pip (works everywhere):**
```bash
python3 -m venv .venv-golf
source .venv-golf/bin/activate            # Windows: .venv-golf\Scripts\activate
pip install --upgrade pip

# for Path A (.pt):
pip install ultralytics opencv-python-headless
# for Path B (.tflite) instead/also:
pip install ai-edge-litert opencv-python-headless numpy
```

**Or uv (faster, if you have it):**
```bash
uv venv .venv-golf && source .venv-golf/bin/activate
uv pip install ultralytics opencv-python-headless          # Path A
uv pip install ai-edge-litert opencv-python-headless numpy # Path B
```

After `activate`, just type `python` — it's the env's Python. (Deactivate later with `deactivate`.)

---

## 3. Run it

### Path A — `.pt` (recommended)
```bash
python golf/annotate_video.py YOUR_VIDEO.mp4 \
    --model /path/to/golf_ego_v5.pt \
    --imgsz 640 --conf 0.5 --device 0
```
- Output: `YOUR_VIDEO_annotated.mp4` next to the input (a **folder** works too — does every video inside).
- `--imgsz 640 --conf 0.5` ≈ the phone. Use `--imgsz 1280` for best accuracy (small ball), `--conf 0.25`
  to also see lower-confidence detections.
- **No GPU?** use `--device cpu` (slower but works).
- Quick test first: add `--max-frames 300`.

### Path B — `.tflite` (exact phone parity)
```bash
python golf/annotate_video_tflite.py YOUR_VIDEO.mp4 \
    --model /path/to/golf_v5.tflite --conf 0.5
```
- Output: `YOUR_VIDEO_tflite.mp4`. Mirrors the phone app's decode 1:1 (640, score≥0.5, per-class NMS).
- CPU-only, no GPU flag needed.

Box colors: **ball = cyan, club_head = amber, hole = green.**

---

## 4. Troubleshooting

| Symptom | Fix |
|---|---|
| `SSLError … CERTIFICATE_VERIFY_FAILED` on pip or first run | Corporate proxy. Prefix with the system certs: `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt python golf/annotate_video.py …` and `pip install --cert /etc/ssl/certs/ca-certificates.crt …` |
| Ultralytics tries to **download** a model | Your `--model` path is wrong/missing → it treats the name as something to fetch. Give the real path to the `.pt` file. |
| `no TFLite runtime` (Path B) | `pip install ai-edge-litert` (or `tflite-runtime`). |
| Output video won't open | Install ffmpeg, or re-encode: `ffmpeg -i out.mp4 -c:v libx264 out_h264.mp4`. |
| Boxes look aspect-distorted (Path B) | The app may letterbox instead of squash — tell the maintainer to adjust preprocessing. |

That's it — one env, one command, an annotated video out.
