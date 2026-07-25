#!/usr/bin/env python3
"""Split a SAM-auto-labeled set into THREE Label Studio projects — train / val / test — by the
dataset person-split, with the SAM boxes imported as pre-annotations to correct (not draw).

Routing: each frame is named <domain>_<person>_..._f######.jpg (from sam3_label_golf.py), so the
person is field [1]; the split manifest (golf_split_manifest.csv, person->split) decides which project
it goes to. So one person's frames never straddle projects — review val/test as ground truth, train
separately.

In:  out_sam/images/*.jpg + out_sam/labels/*.txt (+ classes.txt)   and   golf_split_manifest.csv
Out: out_sam/ls_train.json, ls_val.json, ls_test.json  (+ ls_config.xml)

    python golf/make_ls_projects.py out_sam --manifest golf_split_manifest.csv

Then in Label Studio (one project per split): Create project -> Labeling Setup -> Code -> paste
ls_config.xml; Import ls_<split>.json; add a Local Storage (Absolute path = out_sam/images) so the
/data/local-files URLs resolve (see golf/OFFLINE_PIPELINE.md / the golf-label-studio note).
"""
import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


def yolo_to_results(txt, W, H, names):
    out = []
    for j, line in enumerate(txt.strip().splitlines()):
        p = line.split()
        if len(p) < 5:
            continue
        c = int(p[0]); cx, cy, bw, bh = map(float, p[1:5])
        out.append({
            "id": f"p{j}", "type": "rectanglelabels", "from_name": "label", "to_name": "image",
            "original_width": W, "original_height": H, "image_rotation": 0,
            "value": {"x": clamp((cx - bw / 2) * 100), "y": clamp((cy - bh / 2) * 100),
                      "width": clamp(bw * 100), "height": clamp(bh * 100), "rotation": 0,
                      "rectanglelabels": [names[c] if c < len(names) else str(c)]},
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sam_dir", help="SAM output dir (images/ + labels/ + classes.txt)")
    ap.add_argument("--manifest", default="golf_split_manifest.csv", help="person->split CSV")
    ap.add_argument("--person-index", type=int, default=1, help="which '_'-field of the frame name is the person")
    ap.add_argument("--ls-prefix", default="/data/local-files/?d=images/",
                    help="LS local-files URL prefix (DOCUMENT_ROOT must be the sam_dir)")
    ap.add_argument("--default-split", default="train", help="split for people missing from the manifest")
    ap.add_argument("--val-test-stride", type=int, default=1,
                    help="subsample val/test to every Nth frame PER VIDEO (train unaffected). The frames were "
                         "pre-labeled at 5fps, so 5 -> ~1fps. Lets you label a light val/test now and add the "
                         "rest later (no re-label, no dup).")
    ap.add_argument("--val-test-keep", default="0",
                    help="with --val-test-stride N, which phase(s) 0..N-1 to KEEP now (comma-sep). Start '0'; "
                         "to ADD the rest later re-run with e.g. '1,2,3,4' -> disjoint frames, import as new tasks.")
    args = ap.parse_args()

    stride = max(1, args.val_test_stride)
    keep_phases = {int(x) for x in args.val_test_keep.split(",") if x.strip() != ""}

    def vid_key(s):   # frame name is <...>_f######; group by everything before _f (one video)
        return s.rsplit("_f", 1)[0]

    def frame_idx(s):
        tail = s.rsplit("_f", 1)
        return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else 0

    D = Path(args.sam_dir)
    names = (D / "classes.txt").read_text().split() if (D / "classes.txt").is_file() else ["ball", "club_head", "hole"]

    # person -> split
    split_of = {}
    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            split_of[row["person"].strip()] = row["split"].strip()
    print(f"manifest: {len(split_of)} people -> splits {Counter(split_of.values())}")

    miss = Counter()
    imgs = sorted(glob.glob(str(D / "images" / "*.jpg")))
    recs = []  # (path, stem, split)
    for p in imgs:
        stem = Path(p).stem
        fields = stem.split("_")
        person = fields[args.person_index] if len(fields) > args.person_index else "?"
        split = split_of.get(person, args.default_split)
        if person not in split_of:
            miss[person] += 1
        recs.append((p, stem, split))

    # subsample val/test to every Nth frame per video (train keeps all); rest is deferred for a later add
    drop = set()
    deferred = Counter()
    if stride > 1:
        from collections import defaultdict
        groups = defaultdict(list)
        for i, (p, stem, split) in enumerate(recs):
            if split in ("val", "test"):
                groups[(split, vid_key(stem))].append((frame_idx(stem), i))
        for (split, _), lst in groups.items():
            lst.sort()
            for rank, (_, i) in enumerate(lst):
                if (rank % stride) not in keep_phases:
                    drop.add(i); deferred[split] += 1

    tasks = {"train": [], "val": [], "test": []}
    for i, (p, stem, split) in enumerate(recs):
        if i in drop:
            continue
        with Image.open(p) as im:
            W, H = im.size
        lp = D / "labels" / f"{stem}.txt"
        results = yolo_to_results(lp.read_text(), W, H, names) if lp.is_file() else []
        tasks[split].append({
            "data": {"image": f"{args.ls_prefix}{Path(p).name}"},
            "predictions": [{"model_version": "detector", "result": results}],
        })

    if stride > 1:
        print(f"val/test stride {stride}, keeping phase(s) {sorted(keep_phases)} (~1fps of the 5fps frames); "
              f"deferred for a later add: {dict(deferred)}")
    for split, t in tasks.items():
        outp = D / f"ls_{split}.json"
        outp.write_text(json.dumps(t, indent=1))
        print(f"  {split}: {len(t)} tasks -> {outp}")
    if miss:
        print(f"  ⚠️ {sum(miss.values())} frames from people NOT in the manifest -> '{args.default_split}': "
              f"{dict(miss.most_common(8))}")

    cfg = ('<View>\n  <Image name="image" value="$image" zoom="true" zoomControl="true"/>\n'
           '  <RectangleLabels name="label" toName="image">\n'
           + "".join(f'    <Label value="{n}"/>\n' for n in names)
           + '  </RectangleLabels>\n</View>\n')
    (D / "ls_config.xml").write_text(cfg)
    print(f"  label config -> {D / 'ls_config.xml'}")
    print("\nNext (per split project): LS -> Create project -> Labeling Setup -> Code -> paste ls_config.xml;\n"
          f"  Import ls_<split>.json; add Local Storage (Absolute path = {D.resolve()}/images) then review.")


if __name__ == "__main__":
    main()
