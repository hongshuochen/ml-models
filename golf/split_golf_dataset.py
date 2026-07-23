#!/usr/bin/env python3
"""Assign the by-person golf capture set into train / val / test — WHOLE PEOPLE only (no leakage).

Layout expected (any depth below the person folder is fine — .mp4 are found recursively):
    <root>/Indoor/<Person>/.../*.mp4      (+ sibling audio / imu files, ignored)
    <root>/Outdoor/<Person>/.../*.mp4

Rules baked in (see the discussion — this is why the split is trustworthy):
  * A PERSON is the atomic unit: every clip of one person lands in ONE split. Same person seen in
    both Indoor and Outdoor is merged and assigned once (identity leakage otherwise).
  * Indoor and Outdoor are stratified independently, so BOTH domains appear in train/val/test.
  * Fractions target VIDEO COUNT (not people), since people have very different amounts of footage.
  * --pin-val / --pin-test force specific people (e.g. whoever has putting/HOLE footage) into the
    eval sets, so 'hole' is actually measurable. Everything is deterministic (--seed).

val = model selection / early-stop during training. test = touched ONCE at the end. Both are FROZEN:
never mine / review / train on these people (feed the train list to --exclude-trained downstream).

    python golf/split_golf_dataset.py ~/2026/dataset/golf \
        --pin-test "Simon,Joy" --pin-val "Vivek,Michael" --out golf_split_manifest.csv
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

DOMAINS = ("Indoor", "Outdoor")
SKIP = {"_template_folder"}


def scan(root: Path):
    """person -> {domains:set, videos:int, mb:float}."""
    people = defaultdict(lambda: {"domains": set(), "videos": 0, "mb": 0.0})
    for dom in DOMAINS:
        ddir = root / dom
        if not ddir.is_dir():
            print(f"  (no {dom}/ under {root})")
            continue
        for pdir in sorted(p for p in ddir.iterdir() if p.is_dir() and p.name not in SKIP):
            vids = [v for v in pdir.rglob("*.mp4") if v.is_file()]
            if not vids:
                continue
            key = pdir.name.strip()
            rec = people[key]
            rec["domains"].add(dom)
            rec["videos"] += len(vids)
            rec["mb"] += sum(v.stat().st_size for v in vids) / 1e6
    return people


def pack(names, videos_of, targets):
    """Greedy: assign each person (largest first) to the split with the biggest remaining deficit."""
    got = {s: 0 for s in targets}
    out = {}
    for name in sorted(names, key=lambda n: -videos_of[n]):
        s = max(targets, key=lambda k: targets[k] - got[k])   # most-behind-its-target split
        out[name] = s
        got[s] += videos_of[name]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--pin-val", default="", help="comma names forced into val (e.g. hole footage)")
    ap.add_argument("--pin-test", default="", help="comma names forced into test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="golf_split_manifest.csv")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    people = scan(root)
    if not people:
        raise SystemExit(f"no people with .mp4 found under {root}")
    videos_of = {n: r["videos"] for n, r in people.items()}
    pin = {}
    for n in (x.strip() for x in args.pin_val.split(",") if x.strip()):
        pin[n] = "val"
    for n in (x.strip() for x in args.pin_test.split(",") if x.strip()):
        pin[n] = "test"
    for n in pin:
        if n not in people:
            print(f"  WARNING: pinned '{n}' not found — check spelling")

    rng = random.Random(args.seed)
    assign = dict(pin)
    # stratify by domain membership so Indoor & Outdoor both spread across splits
    groups = defaultdict(list)
    for n, r in people.items():
        if n in assign:
            continue
        groups["+".join(sorted(r["domains"]))].append(n)

    for _, names in groups.items():
        rng.shuffle(names)                                    # tie-break stability across equal sizes
        gv = sum(videos_of[n] for n in names)
        tgt = {"test": gv * args.test_frac, "val": gv * args.val_frac,
               "train": gv * (1 - args.val_frac - args.test_frac)}
        assign.update(pack(names, videos_of, tgt))

    # ---- write manifest + summary ----
    rows = []
    for n in sorted(people, key=lambda n: (assign[n], n)):
        r = people[n]
        rows.append({"person": n, "domains": "+".join(sorted(r["domains"])),
                     "videos": r["videos"], "size_mb": round(r["mb"], 1), "split": assign[n]})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["person", "domains", "videos", "size_mb", "split"])
        w.writeheader(); w.writerows(rows)

    tot_v = sum(videos_of.values())
    print(f"\n{len(people)} people, {tot_v} videos -> {args.out}\n")
    print(f"{'split':6} {'people':>6} {'videos':>7} {'share':>6}   Indoor / Outdoor videos")
    for s in ("train", "val", "test"):
        sp = [n for n in people if assign[n] == s]
        v = sum(videos_of[n] for n in sp)
        indoor = sum(people[n]["videos"] for n in sp if "Indoor" in people[n]["domains"])
        outdoor = sum(people[n]["videos"] for n in sp if "Outdoor" in people[n]["domains"])
        print(f"{s:6} {len(sp):>6} {v:>7} {v/tot_v:>5.0%}   {indoor:>6} / {outdoor}")
    print("\nval people :", ", ".join(sorted(n for n in people if assign[n] == "val")))
    print("test people:", ", ".join(sorted(n for n in people if assign[n] == "test")))
    print("\n>> confirm val/test include the putting/HOLE people; re-run with --pin-val/--pin-test to force them.")


if __name__ == "__main__":
    main()
