# Run the golf detector on a video → annotated video

Give a video, get back the same video with **ball / club_head / hole** boxes drawn — exactly as the
on-device phone app draws them (this runs the phone's `.tflite` model).

You only need **3 files** (no repo to clone):
1. `annotate_video_tflite.py` — the runner (this folder)
2. `golf_v5.tflite` — the model (ask whoever shared this)
3. this guide

Requirements: **Python 3.10+**. That's it — no GPU, no PyTorch, no Ultralytics.

---

## 1. Make an environment (once)

Creates an isolated env so nothing touches your system Python.

```bash
python3 -m venv .venv-golf
source .venv-golf/bin/activate            # Windows: .venv-golf\Scripts\activate
pip install --upgrade pip
pip install ai-edge-litert opencv-python-headless numpy
```

(Have `uv`? `uv venv .venv-golf && source .venv-golf/bin/activate && uv pip install ai-edge-litert opencv-python-headless numpy`.)

After `activate`, just type `python` — it's the env's Python. Later, `deactivate` to exit.

---

## 2. Run it

Put `annotate_video_tflite.py` and `golf_v5.tflite` in the same folder as your video, then:

```bash
python annotate_video_tflite.py YOUR_VIDEO.mp4 --model golf_v5.tflite --conf 0.5
```

- Output: **`YOUR_VIDEO_tflite.mp4`** next to the input.
- Point the first argument at a **folder** to process every video inside it.
- Box colors: **ball = cyan, club_head = amber, hole = green.**
- Quick test first: add `--max-frames 300` (only the first 300 frames).

Useful flags:

| Flag | Meaning |
|---|---|
| `--conf 0.5` | keep boxes at score ≥ 0.5 (same threshold as the phone). Lower it (e.g. `0.25`) to also see weaker detections. |
| `--max-frames 300` | stop after N frames — for a fast check |
| `--names ball,club_head,hole` | class names (only change if using a different model) |

---

## 3. Troubleshooting

| Symptom | Fix |
|---|---|
| `no TFLite runtime` | `pip install ai-edge-litert` (or `tflite-runtime`). |
| `SSLError … CERTIFICATE_VERIFY_FAILED` during `pip install` | Corporate network. Use system certs: `pip install --cert /etc/ssl/certs/ca-certificates.crt ai-edge-litert opencv-python-headless numpy` |
| Output video won't open | Install ffmpeg, or re-encode: `ffmpeg -i out_tflite.mp4 -c:v libx264 out_h264.mp4` |
| Boxes look aspect-distorted | The app may letterbox instead of squash — tell the maintainer to adjust preprocessing. |

That's it — one env, one command, an annotated video out.
