#!/usr/bin/env python3
"""Binary "is there a face?" classifier reusing the FROZEN YOLO pico-p4p5 detector backbone.

Take the detector's backbone (nodes 0-10, a plain Sequential — all `from:-1`), freeze it, and train
a tiny classification head on a face/no-face ImageFolder (see prepare_coco_face_cls.py). YOLO inputs
are RGB/255 (no ImageNet mean/std), so we match that. Only the head trains — fast.
"""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from ultralytics import YOLO


class FaceCls(nn.Module):
    def __init__(self, backbone, feat_ch, hidden=256):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(feat_ch, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 2),
        )

    def train(self, mode=True):           # keep the frozen backbone's BN in eval (running stats)
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            f = self.backbone(x)
        return self.head(f)


def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()  # class 0 = face, 1 = noface (ImageFolder alpha order)
            for p, t in zip(pred, y):
                if t == 0 and p == 0: tp += 1
                elif t == 1 and p == 0: fp += 1
                elif t == 0 and p == 1: fn += 1
                else: tn += 1
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return acc, prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/coco_face_cls")
    ap.add_argument("--ckpt", default="runs/detect/face_hand_pico_p45_hagrid_ft/weights/pico_hagrid.pt")
    ap.add_argument("--input", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/cls/face_cls_pico")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    det = YOLO(args.ckpt)
    layers = det.model.model                                  # nodes 0..17
    backbone = nn.Sequential(*[layers[i] for i in range(11)])  # nodes 0-10 (backbone, all from:-1)
    backbone.eval().to(device)
    with torch.no_grad():
        feat_ch = backbone(torch.zeros(1, 3, args.input, args.input, device=device)).shape[1]
    model = FaceCls(backbone, feat_ch).to(device)
    n_head = sum(p.numel() for p in model.head.parameters())
    print(f"backbone frozen (feat {feat_ch}ch)  head params={n_head:,}")

    norm = T.Lambda(lambda t: t)  # YOLO uses RGB/255 only (ToTensor already gives [0,1])
    tf_tr = T.Compose([T.Resize((args.input, args.input)), T.RandomHorizontalFlip(),
                       T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.ToTensor(), norm])
    tf_va = T.Compose([T.Resize((args.input, args.input)), T.ToTensor(), norm])
    tr = datasets.ImageFolder(f"{args.data}/train", tf_tr)
    va = datasets.ImageFolder(f"{args.data}/val", tf_va)
    print(f"classes={tr.classes}  train={len(tr)}  val={len(va)}")
    tl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va, args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    # class-balanced loss (face/noface counts differ)
    import numpy as np
    cnt = np.bincount([y for _, y in tr.samples], minlength=2).astype(float)
    w = torch.tensor((cnt.sum() / (2 * np.maximum(cnt, 1))), dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = 0.0
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in tl:
            opt.zero_grad()
            lossf(model(x.to(device)), y.to(device)).backward()
            opt.step()
        sched.step()
        acc, prec, rec = evaluate(model, vl, device)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        print(f"ep {ep:2d}/{args.epochs}  acc={acc:.3f} P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        if f1 > best:
            best = f1
            torch.save({"head": model.head.state_dict(), "feat_ch": feat_ch, "classes": tr.classes},
                       f"{args.out}/best.pt")
    print(f"done. best F1={best:.3f}  -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
