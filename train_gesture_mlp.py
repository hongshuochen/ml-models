#!/usr/bin/env python3
"""Rotation/flip-invariant 'L' (thumb-index) hand-gesture classifier.

Features = pairwise distances between the 21 hand landmarks, normalized by hand size.
Pairwise distances are invariant to rotation, translation AND mirror by construction —
so one tiny MLP handles any orientation and both hands. Positives = thumb_index /
thumb_index2 hands from HaGRID; negatives = all other gestures.
"""
import glob
import json
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path("/home/max/2026/ml-models")
LM = ROOT / "datasets/hagrid-landmark"
POS = {"thumb_index", "thumb_index2"}
NK = 21
PAIRS = list(combinations(range(NK), 2))  # 210 unique point pairs


def gesture_of(path):  # hagrid__<gesture>__<uuid>.txt
    return Path(path).stem.split("__")[1]


def features(kp_img, box, W, H):
    """image-normalized 21 (x,y) + bbox -> 210 rotation/flip-invariant distances."""
    cx, cy, bw, bh = box
    side = max(bw * W, bh * H) * 1.3          # square crop in pixels (matches training)
    x0, y0 = cx * W - side / 2, cy * H - side / 2
    px = (kp_img[:, 0] * W - x0) / side        # crop-normalized -> square/isotropic
    py = (kp_img[:, 1] * H - y0) / side
    pts = np.stack([px, py], 1)
    scale = np.linalg.norm(pts[0] - pts[9]) + 1e-6   # wrist -> middle-MCP (hand size)
    return np.array([np.linalg.norm(pts[a] - pts[b]) for a, b in PAIRS], np.float32) / scale


ANN = ROOT / "datasets/hagrid_raw/annotations"          # official annots (all images + 21 landmarks)
IMG = ROOT / "datasets/hagrid_raw/HaGRIDv2_dataset_512"  # per-gesture image folders


def _add(items, gesture, lab, X, y):
    folder = IMG / gesture
    for uuid, e in items:
        try:
            W, H = Image.open(folder / f"{uuid}.jpg").size
        except Exception:
            continue
        lms = e.get("hand_landmarks", [])
        for i, box in enumerate(e.get("bboxes", [])):
            if i >= len(lms) or not lms[i] or len(lms[i]) != NK:
                continue
            x, yy, w, h = box  # HaGRID bbox is top-left xywh -> convert to center
            kp = np.array(lms[i], np.float32)  # 21x2, image-normalized
            X.append(features(kp, [x + w / 2, yy + h / 2, w, h], W, H)); y.append(lab)


def load(split, pos_cap, neg_per_gesture):
    X, y = [], []
    for g in POS:                                        # positives: all thumb_index/-2
        jf = ANN / split / f"{g}.json"
        if jf.exists():
            _add(list(json.load(open(jf)).items())[:pos_cap], g, 1, X, y)
    for jf in sorted((ANN / split).glob("*.json")):      # negatives: sample other gestures
        if jf.stem in POS:
            continue
        _add(list(json.load(open(jf)).items())[:neg_per_gesture], jf.stem, 0, X, y)
    return np.array(X, np.float32), np.array(y, np.float32)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def main():
    Xtr, ytr = load("train", pos_cap=8000, neg_per_gesture=500)
    Xva, yva = load("val", pos_cap=600, neg_per_gesture=80)
    print(f"train {Xtr.shape} pos={int(ytr.sum())} neg={int((ytr==0).sum())} | "
          f"val {Xva.shape} pos={int(yva.sum())} neg={int((yva==0).sum())}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = MLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    pw = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
    Xv, yv = torch.tensor(Xva, device=dev), torch.tensor(yva, device=dev)
    for ep in range(300):
        m.train(); opt.zero_grad()
        lossf(m(Xt), yt).backward(); opt.step()

    m.eval()
    with torch.no_grad():
        pv = (torch.sigmoid(m(Xv)) > 0.5).float()
    tp = int(((pv == 1) & (yv == 1)).sum()); fp = int(((pv == 1) & (yv == 0)).sum())
    fn = int(((pv == 0) & (yv == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    print(f"val: acc={(pv==yv).float().mean():.3f} precision={prec:.3f} recall={rec:.3f} "
          f"f1={2*prec*rec/max(prec+rec,1e-6):.3f}")

    # rotation-invariance sanity: rotate a hand's points 73 deg -> features must be identical
    rng = np.random.default_rng(0)
    pts = rng.random((NK, 2)); th = np.deg2rad(73)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    f0 = np.array([np.linalg.norm(pts[a]-pts[b]) for a, b in PAIRS])
    f1 = np.array([np.linalg.norm((pts@R.T)[a]-(pts@R.T)[b]) for a, b in PAIRS])
    print(f"rotation check: max |feat diff| after 73deg rotation = {np.abs(f0-f1).max():.2e} (≈0 = invariant)")

    Path("runs/landmark").mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), "runs/landmark/L_gesture_mlp.pt")
    print("saved runs/landmark/L_gesture_mlp.pt")


if __name__ == "__main__":
    main()
