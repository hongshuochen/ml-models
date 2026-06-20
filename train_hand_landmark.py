#!/usr/bin/env python3
"""Hand landmark regressor (stage 2 of detect -> crop -> regress).

Takes a hand crop (from a detector bbox) and regresses 21 keypoints (x,y),
normalized to the crop. Backbone: MobileNetV3-small (ImageNet-pretrained).
Trained on the Ultralytics hand-keypoints dataset by cropping each annotated
hand bbox on the fly with jitter so it tolerates imperfect detector boxes.

Usage:
  uv run python train_hand_landmark.py --epochs 60 --batch 64 --device cuda
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

NUM_KPTS = 21
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class HandCropDataset(Dataset):
    def __init__(self, root, split, input_size=224, train=True, pad=1.3):
        self.input_size, self.train, self.pad = input_size, train, pad
        img_dir = Path(root) / "images" / split
        lbl_dir = Path(root) / "labels" / split
        self.samples = []
        for lbl in sorted(lbl_dir.glob("*.txt")):
            img = img_dir / f"{lbl.stem}.jpg"
            if not img.exists():
                continue
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) < 5 + NUM_KPTS * 3:
                    continue
                box = [float(v) for v in p[1:5]]  # cx,cy,w,h (image-normalized)
                kp = np.array([float(v) for v in p[5 : 5 + NUM_KPTS * 3]], np.float32).reshape(NUM_KPTS, 3)
                self.samples.append((str(img), box, kp))
        self.jitter = T.ColorJitter(0.3, 0.3, 0.3, 0.05)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, box, kp = self.samples[i]
        im = Image.open(path).convert("RGB")
        W, H = im.size
        cx, cy = box[0] * W, box[1] * H
        bw, bh = box[2] * W, box[3] * H
        side = max(bw, bh, 8.0) * self.pad
        if self.train:  # box jitter ~ imperfect detector boxes
            side *= random.uniform(0.85, 1.25)
            cx += random.uniform(-0.15, 0.15) * side
            cy += random.uniform(-0.15, 0.15) * side
        x0, y0 = cx - side / 2, cy - side / 2
        crop = im.crop((round(x0), round(y0), round(x0 + side), round(y0 + side)))
        crop = crop.resize((self.input_size, self.input_size), Image.BILINEAR)

        kx = (kp[:, 0] * W - x0) / side
        ky = (kp[:, 1] * H - y0) / side
        vis = (kp[:, 2] > 0).astype(np.float32)

        if self.train:
            if random.random() < 0.5:  # horizontal flip: mirror x, indices unchanged
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
                kx = 1.0 - kx
            crop = self.jitter(crop)

        img = TF.normalize(TF.to_tensor(crop), MEAN, STD)
        target = torch.from_numpy(np.stack([kx, ky], 1).reshape(-1).astype(np.float32))  # [42]
        mask = torch.from_numpy(np.repeat(vis, 2))  # [42]
        return img, target, mask


def build_model(backbone="torchvision", pretrained=True):
    # torchvision MobileNetV3-small (full width) ...
    if backbone in ("torchvision", "mnv3s"):
        w = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.mobilenet_v3_small(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_KPTS * 2)
        return m
    # ... or any timm backbone, e.g. mobilenetv3_small_050 / mobilenetv2_035
    import timm
    return timm.create_model(backbone, pretrained=pretrained, num_classes=NUM_KPTS * 2, in_chans=3)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_err, tot_n, pck = 0.0, 0, 0.0
    for img, target, mask in loader:
        img = img.to(device)
        pred = torch.sigmoid(model(img)).cpu()
        p = pred.view(-1, NUM_KPTS, 2)
        t = target.view(-1, NUM_KPTS, 2)
        m = mask.view(-1, NUM_KPTS, 2)[..., 0]  # per-keypoint visible flag
        d = ((p - t) ** 2).sum(-1).sqrt()  # [N,21] normalized L2
        tot_err += (d * m).sum().item()
        pck += ((d < 0.1).float() * m).sum().item()
        tot_n += m.sum().item()
    return tot_err / max(tot_n, 1), pck / max(tot_n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/hand-keypoints")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--input", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/landmark/hand_landmark")
    ap.add_argument("--backbone", default="torchvision", help="'torchvision' or a timm model name")
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch (no ImageNet weights)")
    ap.add_argument("--limit", type=int, default=0, help="cap samples for a smoke test")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tr = HandCropDataset(args.data, "train", args.input, train=True)
    va = HandCropDataset(args.data, "val", args.input, train=False)
    if args.limit:
        tr.samples = tr.samples[: args.limit]
        va.samples = va.samples[: args.limit]
    print(f"train hands={len(tr)}  val hands={len(va)}  device={device}")

    tl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va, args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(args.backbone, pretrained=not args.no_pretrained).to(device)
    print(f"backbone={args.backbone}  pretrained={not args.no_pretrained}  params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = math.inf
    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for img, target, mask in tl:
            img, target, mask = img.to(device), target.to(device), mask.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = torch.sigmoid(model(img))
                loss = (F.smooth_l1_loss(pred, target, reduction="none", beta=0.02) * mask).sum() / mask.sum().clamp(min=1)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()
        err, pck = evaluate(model, vl, device)
        print(f"ep{ep:3d}/{args.epochs}  loss={run/len(tl):.4f}  val_err={err:.4f}  PCK@0.1={pck:.3f}  lr={sched.get_last_lr()[0]:.2e}", flush=True)
        torch.save(model.state_dict(), out / "last.pt")
        if err < best:
            best = err
            torch.save(model.state_dict(), out / "best.pt")
            print(f"   -> saved best (val_err={err:.4f})", flush=True)
    print(f"done. best val_err={best:.4f}. weights in {out}")


if __name__ == "__main__":
    main()
