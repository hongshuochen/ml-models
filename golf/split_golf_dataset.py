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


def allocate(sizes, total):
    """Spread an integer count across groups proportional to size (largest-remainder)."""
    tot = sum(sizes.values()) or 1
    raw = {g: total * n / tot for g, n in sizes.items()}
    base = {g: int(raw[g]) for g in raw}
    for g in sorted(raw, key=lambda g: raw[g] - base[g], reverse=True)[: total - sum(base.values())]:
        base[g] += 1
    return base


def pack(names, videos_of, vtgt, caps):
    """Largest people first to whichever eval split is furthest below its VIDEO target -> a tight
    70/15/15 by video count (chunky whole-people can't hit it exactly, but this is closest and never
    over-holds-out). `caps` optionally hard-limits people per split (None = uncapped, video target
    decides). Everyone else -> train. Use explicit pins to hand-pick eval identities."""
    got = {"val": 0, "test": 0}
    used = {"val": 0, "test": 0}
    out = {}
    for name in sorted(names, key=lambda n: -videos_of[n]):       # largest first = tight video balance
        opts = [s for s in ("val", "test")
                if got[s] < vtgt[s] and (caps.get(s) is None or used[s] < caps[s])]
        if opts:
            s = max(opts, key=lambda k: vtgt[k] - got[k])
            out[name] = s; got[s] += videos_of[name]; used[s] += 1
        else:
            out[name] = "train"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-people", type=int, default=0, help="exact # people in val (0 = derive from --val-frac)")
    ap.add_argument("--test-people", type=int, default=0, help="exact # people in test (0 = derive from --test-frac); "
                    "raise for more identity diversity in the final benchmark")
    ap.add_argument("--pin-val", default="", help="comma names forced into val")
    ap.add_argument("--pin-test", default="", help="comma names forced into test")
    ap.add_argument("--pin-train", default="", help="comma names forced into train (e.g. mixed-identity "
                    "folders like 'Friend Capture' that shouldn't be a clean eval unit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="golf_split_manifest.csv")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    people = scan(root)
    if not people:
        raise SystemExit(f"no people with .mp4 found under {root}")
    videos_of = {n: r["videos"] for n, r in people.items()}
    pin = {}
    for names_s, split in ((args.pin_val, "val"), (args.pin_test, "test"), (args.pin_train, "train")):
        for n in (x.strip() for x in names_s.split(",") if x.strip()):
            pin[n] = split
    for n in pin:
        if n not in people:
            print(f"  WARNING: pinned '{n}' not found — check spelling")

    rng = random.Random(args.seed)
    assign = dict(pin)
    # optional HARD people caps per eval split (else uncapped — video target decides)
    caps_total = {"val": args.val_people or None, "test": args.test_people or None}

    # stratify by domain membership so Indoor & Outdoor both spread across splits
    groups = {}
    for n, r in people.items():
        if n in assign:
            continue
        groups.setdefault("+".join(sorted(r["domains"])), []).append(n)
    sizes = {g: len(ns) for g, ns in groups.items()}
    # spread any people-caps across groups proportionally (subtract pins already placed)
    cap_group = {}
    for s in ("val", "test"):
        if caps_total[s] is None:
            cap_group[s] = {g: None for g in groups}
        else:
            rem = max(0, caps_total[s] - sum(1 for v in pin.values() if v == s))
            cap_group[s] = allocate(sizes, rem)

    for g, names in groups.items():
        rng.shuffle(names)                                    # deterministic tie-break
        gv = sum(videos_of[n] for n in names)
        vtgt = {"val": gv * args.val_frac, "test": gv * args.test_frac}
        caps = {"val": cap_group["val"][g], "test": cap_group["test"][g]}
        assign.update(pack(names, videos_of, vtgt, caps))

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
