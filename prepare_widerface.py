#!/usr/bin/env python3
"""Convert WIDER FACE annotations to YOLO detection format.

WIDER FACE ground-truth (`wider_face_<split>_bbx_gt.txt`) format, per image:

    <relative/image/path.jpg>
    <num_faces>
    x1 y1 w h blur expression illumination invalid occlusion pose
    ... (num_faces lines)

Boxes are absolute pixel `x1 y1 w h`. We convert each to a YOLO line
`0 cx cy w h` with coordinates normalized to [0, 1] (class 0 = face).
Boxes are clipped to the image; degenerate boxes (w<=0 or h<=0 after
clipping) are dropped. Images listed with 0 faces still carry one dummy
`0 0 0 0 ...` line in the gt file, which is skipped (an empty label file
is written so Ultralytics treats the image as background).
"""
import argparse
from pathlib import Path

from PIL import Image


def convert_split(gt_file: Path, images_dir: Path, labels_dir: Path) -> dict:
    stats = {"images": 0, "boxes": 0, "dropped": 0, "missing": 0, "empty": 0}
    lines = gt_file.read_text().splitlines()
    i, n = 0, len(lines)
    while i < n:
        rel = lines[i].strip()
        i += 1
        if not rel:
            continue
        if i >= n:
            break
        count = int(lines[i].strip())
        i += 1
        # WIDER writes a single dummy line even when count == 0.
        raw = lines[i:i + max(count, 1)] if count == 0 else lines[i:i + count]
        i += max(count, 1)

        img_path = images_dir / rel
        if not img_path.exists():
            stats["missing"] += 1
            continue
        with Image.open(img_path) as im:
            W, H = im.size
        if W <= 0 or H <= 0:
            stats["missing"] += 1
            continue

        yolo_lines = []
        if count > 0:
            for row in raw:
                parts = row.split()
                if len(parts) < 4:
                    continue
                x1, y1, w, h = (float(v) for v in parts[:4])
                # Clip box corners to image bounds.
                x2, y2 = x1 + w, y1 + h
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(W), x2), min(float(H), y2)
                bw, bh = x2 - x1, y2 - y1
                if bw <= 0 or bh <= 0:
                    stats["dropped"] += 1
                    continue
                cx = (x1 + bw / 2) / W
                cy = (y1 + bh / 2) / H
                nw, nh = bw / W, bh / H
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        out = labels_dir / Path(rel).with_suffix(".txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        stats["images"] += 1
        stats["boxes"] += len(yolo_lines)
        if not yolo_lines:
            stats["empty"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path, help="wider_face_<split>_bbx_gt.txt")
    ap.add_argument("--images", required=True, type=Path, help="images/<split> dir (contains event subdirs)")
    ap.add_argument("--labels", required=True, type=Path, help="output labels/<split> dir")
    args = ap.parse_args()

    args.labels.mkdir(parents=True, exist_ok=True)
    s = convert_split(args.gt, args.images, args.labels)
    print(
        f"[{args.gt.name}] images={s['images']} boxes={s['boxes']} "
        f"empty_imgs={s['empty']} dropped_boxes={s['dropped']} missing_imgs={s['missing']}"
    )


if __name__ == "__main__":
    main()
