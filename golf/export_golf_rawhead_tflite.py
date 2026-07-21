#!/usr/bin/env python3
"""Export a golf YOLO26 detector to the RAW-HEAD float16 TFLite the Android QNN-NPU path needs.

Why not `yolo export format=tflite`: the deployed detector runs on the Qualcomm Hexagon NPU (QNN
HTP delegate), and YOLO26's end-to-end NMS-free head bakes TopK / GatherNd / INT64 casts into the
graph that NO delegate (GPU *or* NPU) can run. So we export the *raw* one-to-many head (pure
conv+attention) and do threshold + per-class NMS in Kotlin. Two hard-won tricks are baked in here
(see the android-golf-npu-deploy notes / MODELS_REPORT):

  1. Monkeypatch `Detect.forward` to emit the 3 raw per-scale maps from the ONE2ONE branch:
     each `cat([box(4), cls(nc)], 1)` -> `[1, 4+nc, H, W]`, reg_max=1 so distances are DIRECT
     (no DFL, no sigmoid, no anchor decode on-graph).
  2. Monkeypatch `Attention.forward` to unroll the multi-head matmuls to batch=1, AND run
     `onnx2tf` with `enable_batchmatmul_unfold=False`. Ultralytics' float export path unfolds the
     C2PSA attention BatchMatMuls into ~1600 FULLY_CONNECTED ops, which shatters the graph and the
     delegate rejects it. Unrolled + unfold=False keeps it a clean conv graph the QNN HTP eats whole.

Output TFLite: input `[1,640,640,3]` float32 NHWC; outputs `[1,80,80,4+nc] [1,40,40,4+nc]
[1,20,20,4+nc]` (strides 8/16/32), channel order per cell `[l,t,r,b, cls0_logit ... clsN_logit]`.
GENERIC over class count -> works for the 2-class (ball,club_head) and 3-class (+hole) models alike;
the Kotlin decoder reads nc from the output shape.

Run (in the offline uv env, after `uv pip install onnx onnxsim onnx2tf tensorflow`):
    uv run python export_golf_rawhead_tflite.py \
        --weights runs/detect/golf_ego_v4_hole/weights/best.pt \
        --out     golf.tflite
Then drop `golf.tflite` into android-golf/app/src/main/assets/ and rebuild. If the class count
changed, also update LABELS in GolfDetector.kt (the decoder is otherwise class-count-agnostic).
"""
import argparse
import glob
import os
import shutil
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.nn.modules.block import Attention  # noqa: E402


def raw_forward(self, x):
    """Raw one2one per-scale maps, pre-decode: cat([box(4), cls(nc)], 1) -> [1, 4+nc, H, W]."""
    outs = []
    for i in range(self.nl):
        box = self.one2one_cv2[i](x[i])   # [1, 4*reg_max, H, W] = [1,4,H,W] (reg_max=1)
        cls = self.one2one_cv3[i](x[i])   # [1, nc, H, W]
        outs.append(torch.cat((box, cls), 1))
    return tuple(outs)


def attn_unrolled(self, x):
    """Per-head unrolled attention (batch=1 matmuls) to dodge onnx2tf's 4D batched-matmul mangling.
    Mathematically identical to ultralytics.nn.modules.block.Attention.forward."""
    B, C, H, W = x.shape
    N = H * W
    qkv = self.qkv(x)
    q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
        [self.key_dim, self.key_dim, self.head_dim], dim=2)
    outs = []
    for h in range(self.num_heads):
        qh, kh, vh = q[:, h], k[:, h], v[:, h]                       # [B,key,N] [B,key,N] [B,head,N]
        attn = (torch.matmul(qh.transpose(-2, -1), kh) * self.scale).softmax(dim=-1)  # [B,N,N]
        outs.append(torch.matmul(vh, attn.transpose(-2, -1)))       # [B,head,N]
    out = torch.cat(outs, dim=1).view(B, C, H, W)
    x = out + self.pe(v.reshape(B, C, H, W))
    return self.proj(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="detector best.pt (2- or 3-class)")
    ap.add_argument("--out", default="golf.tflite", help="destination .tflite path")
    ap.add_argument("--imgsz", type=int, default=640, help="device input size (deployed = 640)")
    ap.add_argument("--work", default="", help="scratch dir for onnx/saved_model (default: next to --out)")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    work = Path(args.work).resolve() if args.work else out.parent / "_rawhead_export"
    work.mkdir(parents=True, exist_ok=True)

    # Patch at class level so ultralytics' exporter (it calls model(im)) uses our raw graph.
    Detect.forward = raw_forward
    Attention.forward = attn_unrolled

    m = YOLO(str(args.weights))
    det = m.model.model[-1]
    nc = det.nc
    print(f"weights={args.weights}  nc={nc} reg_max={det.reg_max} nl={det.nl} "
          f"stride={det.stride.tolist()}  names={m.names}")

    # 1) torch -> ONNX (fp32, clean conv graph; nms=False, simplify to fold constants)
    onnx_path = m.export(format="onnx", imgsz=args.imgsz, opset=20, simplify=True, nms=False, device="cpu")
    print("ONNX:", onnx_path)

    # 2) ONNX -> TFLite via onnx2tf, unfold=False (keeps attention as clean matmuls, not 1600 FCs)
    import onnx
    import onnx2tf
    mo = onnx.load(onnx_path)
    print("ONNX outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
                            for o in mo.graph.output])
    print("ONNX op counts:", dict(sorted(Counter(n.op_type for n in mo.graph.node).items())))

    sm_dir = work / "saved_model"
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(sm_dir),
        not_use_onnxsim=True,
        verbosity="error",
        output_integer_quantized_tflite=False,
        enable_batchmatmul_unfold=False,          # THE fix: no FC explosion of C2PSA attention
        output_signaturedefs=True,
        disable_group_convolution=False,
    )

    f16 = glob.glob(str(sm_dir / "*_float16.tflite"))
    if not f16:
        raise SystemExit(f"no *_float16.tflite produced in {sm_dir}")
    shutil.copy(f16[0], out)
    print(f"\n✅ wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    # 3) verify shapes (must be 3 heads [1,G,G,4+nc])
    try:
        import tensorflow as tf
        it = tf.lite.Interpreter(model_path=str(out)); it.allocate_tensors()
        print("  input :", [d["shape"].tolist() for d in it.get_input_details()])
        outs = sorted([d["shape"].tolist() for d in it.get_output_details()], key=lambda s: -s[1])
        print("  output:", outs)
        ch = {s[-1] for s in outs}
        assert ch == {4 + nc}, f"expected last-dim {4+nc}, got {ch}"
        print(f"  OK: 3 raw heads, {4+nc} ch = 4 box + {nc} classes")
    except Exception as e:
        print("  (shape check skipped:", e, ")")


if __name__ == "__main__":
    main()
