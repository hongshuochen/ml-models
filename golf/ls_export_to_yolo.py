#!/usr/bin/env python3
"""Export the human-CORRECTED Label Studio annotations (golf-train/val/test projects) to YOLO
`images/ + labels/` dirs, ready to feed build_and_train_golf.py (--reviewed / --val) + a test eval.

Run this ONCE LABELING IS DONE. For each project it pulls the export JSON (only annotated tasks,
with the reviewer's boxes), converts each RectangleLabels box -> YOLO `cls cx cy w h`, writes a real
label .txt, and SYMLINKS the image from the pre-label dir (no copy). A reviewed frame with all boxes
deleted -> empty .txt (a valid background/negative). Tasks nobody labeled yet are skipped (reported).

    ~/ml-models/.venv/bin/python golf/ls_export_to_yolo.py \
        --url http://105.145.25.32:8080 --token '<TOKEN>' \
        --train 15 --val 18 --test 20 \
        --images-dir out_prelabel/images --out golf_reviewed

Output:
    golf_reviewed/train/{images/*.jpg (symlink), labels/*.txt}
    golf_reviewed/val/{images,labels}
    golf_reviewed/test/{images,labels}
Then it prints the exact build_and_train_golf.py command (train on train, eval on the new real val)
and the final test-set eval command.
"""
import argparse
import os
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

AUTH = {"scheme": None, "bearer": None}  # same Token/Bearer/refresh-JWT auto-detect as ls_progress.py


def _refresh_access(url, token):
    r = requests.post(url.rstrip("/") + "/api/token/refresh", json={"refresh": token}, timeout=30)
    r.raise_for_status()
    AUTH["bearer"] = r.json()["access"]


def get(url, token, path, **params):
    def _try(hdr):
        return requests.get(url.rstrip("/") + path, headers=hdr, params=params, timeout=600)
    if AUTH["scheme"] == "Token":
        r = _try({"Authorization": f"Token {token}"})
    elif AUTH["scheme"] == "Bearer":
        r = _try({"Authorization": f"Bearer {AUTH['bearer']}"})
        if r.status_code == 401:
            _refresh_access(url, token); r = _try({"Authorization": f"Bearer {AUTH['bearer']}"})
    else:
        r = _try({"Authorization": f"Token {token}"})
        if r.status_code != 401:
            AUTH["scheme"] = "Token"
        else:
            r = _try({"Authorization": f"Bearer {token}"})
            if r.status_code != 401:
                AUTH["scheme"], AUTH["bearer"] = "Bearer", token
            else:
                _refresh_access(url, token)
                r = _try({"Authorization": f"Bearer {AUTH['bearer']}"}); AUTH["scheme"] = "Bearer"
    r.raise_for_status()
    return r.json()


def image_name(data_image):
    """'/data/local-files/?d=images/foo_f000030.jpg' (or a plain path/URL) -> 'foo_f000030.jpg'."""
    q = parse_qs(urlparse(data_image).query)
    if "d" in q:
        return Path(q["d"][0]).name
    return Path(urlparse(data_image).path).name


def to_yolo_lines(result, cls_index):
    """LS RectangleLabels (percent, top-left) -> YOLO 'cls cx cy w h' (normalized center)."""
    lines = []
    for r in result:
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        labels = v.get("rectanglelabels") or []
        if not labels or labels[0] not in cls_index:
            continue
        c = cls_index[labels[0]]
        cx = (v["x"] + v["width"] / 2) / 100
        cy = (v["y"] + v["height"] / 2) / 100
        w, h = v["width"] / 100, v["height"] / 100
        lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def export_split(url, token, pid, split, images_dir, out, cls_index, names):
    dst_img = out / split / "images"
    dst_lbl = out / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    tasks = get(url, token, f"/api/projects/{pid}/export", exportType="JSON")
    labeled = skipped_noann = skipped_noimg = 0
    box_by_cls = Counter()
    for t in (tasks or []):
        anns = [a for a in (t.get("annotations") or []) if not a.get("was_cancelled")]
        if not anns:
            skipped_noann += 1
            continue
        name = image_name(t["data"]["image"])
        src = (images_dir / name).resolve()
        if not src.is_file():
            skipped_noimg += 1
            continue
        lines = to_yolo_lines(anns[0].get("result") or [], cls_index)
        stem = Path(name).stem
        (dst_lbl / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        link = dst_img / name
        if not link.exists():
            os.symlink(src, link)
        labeled += 1
        for ln in lines:
            box_by_cls[int(ln.split()[0])] += 1
    per_cls = ", ".join(f"{names[i]}={box_by_cls.get(i, 0):,}" for i in range(len(names)))
    print(f"  {split:5s} (proj {pid}): {labeled:,} labeled frames | boxes: {per_cls}"
          + (f" | skipped {skipped_noann:,} unlabeled" if skipped_noann else "")
          + (f" | {skipped_noimg:,} missing images" if skipped_noimg else ""))
    return labeled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--train", type=int, help="golf-train project id")
    ap.add_argument("--val", type=int, help="golf-val project id")
    ap.add_argument("--test", type=int, help="golf-test project id")
    ap.add_argument("--images-dir", default="out_prelabel/images", help="where the pre-labeled frames live")
    ap.add_argument("--out", default="golf_reviewed", help="output dataset root")
    ap.add_argument("--names", default="", help="class order (comma). Default: read out_prelabel/classes.txt")
    ap.add_argument("--base-weights", default="runs/detect/golf_ego_v5_nomined/weights/best.pt",
                    help="only for the printed train command")
    args = ap.parse_args()

    images_dir = Path(args.images_dir).expanduser()
    out = Path(args.out)
    if args.names:
        names = [n.strip() for n in args.names.split(",")]
    else:
        cf = images_dir.parent / "classes.txt"
        names = cf.read_text().split() if cf.is_file() else ["ball", "club_head", "hole"]
    cls_index = {n: i for i, n in enumerate(names)}
    print(f"classes {names} | images {images_dir} -> {out}/")

    splits = [("train", args.train), ("val", args.val), ("test", args.test)]
    for split, pid in splits:
        if pid is not None:
            export_split(args.url, args.token, pid, split, images_dir, out, cls_index, names)

    # next-step commands
    py = "~/ml-models/.venv/bin/python"
    print("\n── next: train golf_ego_v6 (train on the reviewed train, eval on the NEW real val) ──")
    if args.train and args.val:
        print(f"{py} golf/build_and_train_golf.py \\\n"
              f"    --base-weights {args.base_weights} \\\n"
              f"    --val      {out}/val \\\n"
              f"    --reviewed {out}/train \\\n"
              f"    --names {','.join(names)} \\\n"
              f"    --name golf_ego_v6 --imgsz 1280 --epochs 40 --batch 6")
    if args.test:
        # write a ready data.yaml (val-> test images) so `yolo val` gives TEST metrics directly
        test_yaml = out / "test.yaml"
        test_yaml.write_text(
            f"train: {(out / 'test' / 'images').resolve()}\n"
            f"val: {(out / 'test' / 'images').resolve()}\n"
            "names:\n" + "".join(f"  {i}: {nm}\n" for i, nm in enumerate(names)))
        print(f"\n── then: final metrics on the held-out TEST set (real hole recall) ──\n"
              f"~/ml-models/.venv/bin/yolo val model=runs/detect/golf_ego_v6/weights/best.pt \\\n"
              f"    data={test_yaml} imgsz=1280      # mAP50 + per-class recall on TEST")


if __name__ == "__main__":
    main()
