#!/usr/bin/env python3
"""Export the hand landmark regressor to ONNX (normalize + sigmoid baked in) and
render a sanity-check visualization. Run onnx2tf afterwards for TFLite."""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

from train_hand_landmark import MEAN, STD, NUM_KPTS, build_model

CKPT = "runs/landmark/hand_landmark/best.pt"
OUT_ONNX = "runs/landmark/hand_landmark/hand_landmark.onnx"
EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],
         [11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]]


class Wrapped(nn.Module):
    """Input: [B,3,224,224] RGB in [0,1]. Output: [B,42] keypoints in [0,1]."""
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        return torch.sigmoid(self.m((x - self.mean) / self.std))


def main():
    m = build_model()
    m.load_state_dict(torch.load(CKPT, map_location="cpu"))
    w = Wrapped(m).eval()

    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params:,}")

    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(w, dummy, OUT_ONNX, input_names=["images"],
                      output_names=["keypoints"], opset_version=13, dynamo=False)
    print(f"exported ONNX -> {OUT_ONNX}")

    # sanity viz on a real hand crop (replicate eval cropping)
    import glob
    p = sorted(glob.glob("datasets/hand-keypoints/images/val/*.jpg"))[0]
    lbl = p.replace("/images/", "/labels/").replace(".jpg", ".txt")
    box = [float(v) for v in open(lbl).readline().split()[1:5]]
    im = Image.open(p).convert("RGB"); W, H = im.size
    cx, cy, bw, bh = box[0]*W, box[1]*H, box[2]*W, box[3]*H
    side = max(bw, bh) * 1.3
    x0, y0 = cx - side/2, cy - side/2
    crop = im.crop((round(x0), round(y0), round(x0+side), round(y0+side))).resize((224, 224))
    with torch.no_grad():
        kp = w(TF.to_tensor(crop).unsqueeze(0))[0].view(NUM_KPTS, 2).numpy()
    dr = ImageDraw.Draw(crop)
    for a, b in EDGES:
        dr.line([kp[a,0]*224, kp[a,1]*224, kp[b,0]*224, kp[b,1]*224], fill=(0,255,255), width=2)
    for x, y in kp:
        dr.ellipse([x*224-3, y*224-3, x*224+3, y*224+3], fill=(255,180,0))
    crop.save("landmark_check.png")
    print("saved landmark_check.png")


if __name__ == "__main__":
    main()
