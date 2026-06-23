#!/usr/bin/env python3
"""Verify converted MobileFaceNet TFLite vs source ONNX for face recognition.

A) Numerical parity on ONE real aligned face crop (onnx vs tflite float32).
B) Discrimination: same-image, same-person, different-person cosine.
C) Alignment template + landmark order report.
"""
import json
import os
import glob
import sys
import numpy as np
import cv2

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import onnxruntime as ort
import tensorflow as tf
from insightface.app import FaceAnalysis
from insightface.utils import face_align

ROOT = "/home/max/2026/ml-models"
ONNX = "/home/max/.insightface/models/buffalo_s/w600k_mbf.onnx"
TFLITE = f"{ROOT}/runs/face_recog/w600k_mbf_float32.tflite"
IMGDIR = f"{ROOT}/datasets/hagrid_raw/HaGRIDv2_dataset_512/three2"
ANN = f"{ROOT}/datasets/hagrid_raw/annotations/train/three2.json"


def l2n(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + 1e-12)


def cos(a, b):
    return float(np.dot(l2n(a), l2n(b)))


# ---------- ONNX runner ----------
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
onnx_in = sess.get_inputs()[0]
onnx_out = sess.get_outputs()[0].name
print("ONNX input:", onnx_in.name, onnx_in.shape, onnx_in.type)


def onnx_embed(aligned_rgb_uint8):
    # aligned_rgb_uint8: HWC RGB uint8, 112x112
    x = aligned_rgb_uint8.astype(np.float32)
    x = (x - 127.5) / 127.5            # [-1,1]
    x = np.transpose(x, (2, 0, 1))[None]  # NCHW
    out = sess.run([onnx_out], {onnx_in.name: x})[0]
    return out[0]


# ---------- TFLite runner ----------
interp = tf.lite.Interpreter(model_path=TFLITE)
interp.allocate_tensors()
tin = interp.get_input_details()[0]
tout = interp.get_output_details()[0]
print("TFLite input:", tin["shape"], tin["dtype"])
print("TFLite output:", tout["shape"], tout["dtype"])
TF_NHWC = list(tin["shape"]) == [1, 112, 112, 3]
print("TFLite layout NHWC:", TF_NHWC)


def tflite_embed(aligned_rgb_uint8):
    x = aligned_rgb_uint8.astype(np.float32)
    x = (x - 127.5) / 127.5
    if TF_NHWC:
        x = x[None]                       # [1,112,112,3]
    else:
        x = np.transpose(x, (2, 0, 1))[None]  # [1,3,112,112]
    interp.set_tensor(tin["index"], x.astype(tin["dtype"]))
    interp.invoke()
    return interp.get_tensor(tout["index"])[0]


# ---------- Face detector for alignment ----------
app = FaceAnalysis(name="buffalo_s",
                   providers=["CPUExecutionProvider"],
                   allowed_modules=["detection"])
app.prepare(ctx_id=-1, det_size=(640, 640))


def aligned_crop(img_path):
    """Detect largest face, return 112x112 RGB aligned crop or None."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None
    faces = app.get(bgr)
    if not faces:
        return None
    # largest face by box area
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    kps = faces[0].kps  # (5,2) in [le, re, nose, lm, rm] order
    aligned_bgr = face_align.norm_crop(bgr, kps, image_size=112)  # BGR 112x112
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    return aligned_rgb, kps


# ===================================================================
# Build a pool of images that have a detectable face, grouped by user_id
# ===================================================================
ann = json.load(open(ANN))
img_ids = [os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(IMGDIR + "/*.jpg")]
img_ids = [i for i in img_ids if i in ann]

# group by user
from collections import defaultdict
by_user = defaultdict(list)
for i in img_ids:
    by_user[ann[i]["user_id"]].append(i)
multi = sorted([(u, ids) for u, ids in by_user.items() if len(ids) >= 2],
               key=lambda x: -len(x[1]))

print("\nusers with >=2 imgs:", len(multi))

# Cache embeddings/crops; only compute on demand. Find faces that detect.
emb_cache = {}
crop_cache = {}


def get(img_id):
    if img_id in emb_cache:
        return emb_cache[img_id]
    res = aligned_crop(f"{IMGDIR}/{img_id}.jpg")
    if res is None:
        emb_cache[img_id] = None
        return None
    crop, kps = res
    crop_cache[img_id] = (crop, kps)
    emb = tflite_embed(crop)
    emb_cache[img_id] = emb
    return emb


# ---------------- A) PARITY ----------------
print("\n========== A) NUMERICAL PARITY ==========")
parity_id = None
for u, ids in multi:
    for i in ids:
        res = aligned_crop(f"{IMGDIR}/{i}.jpg")
        if res is not None:
            parity_id = i
            parity_crop, parity_kps = res
            break
    if parity_id:
        break
print("parity image:", parity_id)
e_onnx = onnx_embed(parity_crop)
e_tf = tflite_embed(parity_crop)
parity_cos = cos(e_onnx, e_tf)
# mean abs diff on raw (un-normalized) outputs
parity_mad = float(np.mean(np.abs(np.asarray(e_onnx).ravel() -
                                  np.asarray(e_tf).ravel())))
# also mad on L2-normalized
parity_mad_norm = float(np.mean(np.abs(l2n(e_onnx) - l2n(e_tf))))
print(f"onnx emb shape {np.asarray(e_onnx).shape}  tflite emb shape {np.asarray(e_tf).shape}")
print(f"parity cosine = {parity_cos:.8f}")
print(f"parity mean-abs-diff (raw) = {parity_mad:.8e}")
print(f"parity mean-abs-diff (L2norm) = {parity_mad_norm:.8e}")
print(f"onnx norm = {np.linalg.norm(e_onnx):.4f}  tflite norm = {np.linalg.norm(e_tf):.4f}")

# ---------------- B) DISCRIMINATION ----------------
print("\n========== B) DISCRIMINATION ==========")
# same image vs itself
same_img_cos = cos(tflite_embed(parity_crop), tflite_embed(parity_crop))
print(f"same-image (self) cosine = {same_img_cos:.6f}")

# same-person pairs (different images, same user)
same_pairs = []
for u, ids in multi:
    embs = []
    used = []
    for i in ids:
        if len(embs) >= 2:
            break
        e = get(i)
        if e is not None:
            embs.append(e)
            used.append(i)
    if len(embs) >= 2:
        same_pairs.append((u, used[0], used[1], cos(embs[0], embs[1])))
    if len(same_pairs) >= 8:
        break
same_vals = [p[3] for p in same_pairs]
print(f"\nSAME-PERSON pairs ({len(same_pairs)}):")
for u, a, b, c in same_pairs:
    print(f"  {c:.4f}  user={u[:10]}")
if same_vals:
    print(f"  same-person cosine range: {min(same_vals):.4f} .. {max(same_vals):.4f}  mean={np.mean(same_vals):.4f}")

# different-person pairs: take one good image from distinct users
diff_reps = []
for u, ids in multi:
    for i in ids:
        e = get(i)
        if e is not None:
            diff_reps.append((u, i, e))
            break
    if len(diff_reps) >= 8:
        break
diff_pairs = []
for a in range(len(diff_reps)):
    for b in range(a + 1, len(diff_reps)):
        diff_pairs.append((diff_reps[a][0], diff_reps[b][0],
                           cos(diff_reps[a][2], diff_reps[b][2])))
diff_vals = [p[2] for p in diff_pairs]
print(f"\nDIFFERENT-PERSON pairs ({len(diff_pairs)}) from {len(diff_reps)} distinct users:")
for ua, ub, c in diff_pairs:
    print(f"  {c:.4f}  {ua[:8]} vs {ub[:8]}")
if diff_vals:
    print(f"  different-person cosine range: {min(diff_vals):.4f} .. {max(diff_vals):.4f}  mean={np.mean(diff_vals):.4f}")

# threshold recommendation: midpoint between max(diff) and min(same)
if same_vals and diff_vals:
    thr = (max(diff_vals) + min(same_vals)) / 2.0
    print(f"\nseparation: max(diff)={max(diff_vals):.4f}  min(same)={min(same_vals):.4f}")
    print(f"recommended threshold ~ {thr:.3f}")

# ---------------- C) TEMPLATE ----------------
print("\n========== C) ALIGNMENT TEMPLATE ==========")
print("arcface_dst (112x112):")
print(face_align.arcface_dst)
print("landmark order: [left_eye, right_eye, nose, left_mouth, right_mouth]")
print("detector kps order sample (parity img):")
print(parity_kps)

# dump machine-readable summary
summary = dict(
    parity_cos=parity_cos,
    parity_mad=parity_mad,
    parity_mad_norm=parity_mad_norm,
    same_img_cos=same_img_cos,
    same_min=float(min(same_vals)) if same_vals else None,
    same_max=float(max(same_vals)) if same_vals else None,
    same_mean=float(np.mean(same_vals)) if same_vals else None,
    diff_min=float(min(diff_vals)) if diff_vals else None,
    diff_max=float(max(diff_vals)) if diff_vals else None,
    diff_mean=float(np.mean(diff_vals)) if diff_vals else None,
    n_same=len(same_pairs),
    n_diff=len(diff_pairs),
)
print("\nSUMMARY_JSON=" + json.dumps(summary))
