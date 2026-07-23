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
    """person -> {domains:set, paths:[Path], mb:float}."""
    people = defaultdict(lambda: {"domains": set(), "paths": [], "mb": 0.0})
    for dom in DOMAINS:
        ddir = root / dom
        if not ddir.is_dir():
            print(f"  (no {dom}/ under {root})")
            continue
        for pdir in sorted(p for p in ddir.iterdir() if p.is_dir() and p.name not in SKIP):
            vids = [v for v in pdir.rglob("*.mp4") if v.is_file()]
            if not vids:
                continue
            rec = people[pdir.name.strip()]
            rec["domains"].add(dom)
            rec["paths"] += vids
            rec["mb"] += sum(v.stat().st_size for v in vids) / 1e6
    return people


def probe_minutes(paths, cache_path):
    """Total minutes across `paths` via cv2 header (frames/fps — no decode). Cached by abspath so
    re-runs are instant. Returns {person-agnostic} total minutes for the given list."""
    import json
    cache = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    todo = [p for p in paths if str(p) not in cache]
    if todo:
        import cv2
        print(f"  probing duration of {len(todo)} new videos (cv2 header; cached after)...", flush=True)
        for i, p in enumerate(todo):
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cap.release()
            cache[str(p)] = (n / fps) if (fps > 0 and n > 0) else 0.0
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{len(todo)}", flush=True)
        cache_path.write_text(json.dumps(cache))
    return sum(cache.get(str(p), 0.0) for p in paths) / 60.0


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
    ap.add_argument("--by", choices=("videos", "minutes"), default="videos",
                    help="balance & report by video COUNT (fast) or actual DURATION in minutes "
                         "(cv2-probes each clip once, cached) — minutes is the truer measure of footage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="golf_split_manifest.csv")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    people = scan(root)
    if not people:
        raise SystemExit(f"no people with .mp4 found under {root}")
    videos_of = {n: len(r["paths"]) for n, r in people.items()}
    if args.by == "minutes":
        cache = Path(args.out).with_name(Path(args.out).stem + "_durations.json")
        minutes_of = {n: probe_minutes(r["paths"], cache) for n, r in people.items()}
        weight_of = minutes_of
    else:
        minutes_of = None
        weight_of = videos_of
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
        gw = sum(weight_of[n] for n in names)                 # group weight (videos or minutes)
        vtgt = {"val": gw * args.val_frac, "test": gw * args.test_frac}
        caps = {"val": cap_group["val"][g], "test": cap_group["test"][g]}
        assign.update(pack(names, weight_of, vtgt, caps))     # balance on the chosen metric

    # ---- write manifest + summary ----
    fields = ["person", "domains", "videos", "size_mb"] + (["minutes"] if minutes_of else []) + ["split"]
    rows = []
    for n in sorted(people, key=lambda n: (assign[n], n)):
        r = people[n]
        row = {"person": n, "domains": "+".join(sorted(r["domains"])),
               "videos": videos_of[n], "size_mb": round(r["mb"], 1), "split": assign[n]}
        if minutes_of:
            row["minutes"] = round(minutes_of[n], 1)
        rows.append(row)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    tot_v = sum(videos_of.values())
    tot_m = sum(minutes_of.values()) if minutes_of else 0
    hdr = f"\n{len(people)} people, {tot_v} videos" + (f", {tot_m:.0f} min" if minutes_of else "") + f" -> {args.out}"
    print(hdr + f"   (balanced by {args.by})\n")
    mcol = f"{'minutes':>8}" if minutes_of else ""
    print(f"{'split':6} {'people':>6} {'videos':>7}{mcol} {'share':>6}   Indoor / Outdoor {'min' if minutes_of else 'videos'}")
    for s in ("train", "val", "test"):
        sp = [n for n in people if assign[n] == s]
        v = sum(videos_of[n] for n in sp)
        w = sum(weight_of[n] for n in sp)
        tot_w = tot_m if minutes_of else tot_v
        indoor = sum((minutes_of or videos_of)[n] for n in sp if "Indoor" in people[n]["domains"])
        outdoor = sum((minutes_of or videos_of)[n] for n in sp if "Outdoor" in people[n]["domains"])
        mc = f"{sum(minutes_of[n] for n in sp):>8.0f}" if minutes_of else ""
        print(f"{s:6} {len(sp):>6} {v:>7}{mc} {w/tot_w:>5.0%}   {indoor:>6.0f} / {outdoor:.0f}")
    print("\nval people :", ", ".join(sorted(n for n in people if assign[n] == "val")))
    print("test people:", ", ".join(sorted(n for n in people if assign[n] == "test")))
    print("\n>> all Indoor footage has holes, so any Indoor person in val/test covers the hole class.")


if __name__ == "__main__":
    main()
