#!/usr/bin/env python3
"""Reference Florence-2-base-ft ONNX caption pipeline — pure onnxruntime + tokenizers.

This is the spec the Android app (FlorenceCaptioner.kt) mirrors exactly. Verified to reproduce
transformers' captions on CPU at ~2-4 s/image. The KV-cache decoder export has a static-16
`inputs_embeds` bug, so we decode WITHOUT a cache: the vision encoder runs once and the small BART
decoder is re-run on the growing sequence (fine for short captions).

    uv run --with onnxruntime --with tokenizers python florence_onnx_demo.py <image.jpg>

Models (downloaded once to models/florence2_onnx) from onnx-community/Florence-2-base-ft:
    onnx/{vision_encoder,embed_tokens,encoder_model,decoder_model}.onnx + tokenizer.json
"""
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

D = "../models/florence2_onnx/"
PROMPTS = {
    "<CAPTION>": "What does the image describe?",
    "<DETAILED_CAPTION>": "Describe in detail what is shown in the image.",
    "<MORE_DETAILED_CAPTION>": "Describe with a paragraph what is shown in the image.",
}


def main():
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = {n: ort.InferenceSession(D + f"onnx/{n}.onnx", so, providers=["CPUExecutionProvider"])
            for n in ("vision_encoder", "embed_tokens", "encoder_model", "decoder_model")}
    tok = Tokenizer.from_file(D + "tokenizer.json")

    def embed(ids):
        return sess["embed_tokens"].run(None, {"input_ids": np.array([ids], np.int64)})[0]

    def preprocess(img):
        img = img.convert("RGB").resize((768, 768), Image.BICUBIC)
        x = (np.asarray(img, np.float32) / 255 - np.array([0.485, 0.456, 0.406], np.float32)) \
            / np.array([0.229, 0.224, 0.225], np.float32)
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def caption(img, task, max_new=64):
        t0 = time.time()
        feat = sess["vision_encoder"].run(None, {"pixel_values": preprocess(img)})[0]   # [1, nImg, 768]
        inp = np.concatenate([feat, embed(tok.encode(PROMPTS[task]).ids)], axis=1).astype(np.float32)
        amask = np.ones((1, inp.shape[1]), np.int64)
        ehs = sess["encoder_model"].run(None, {"inputs_embeds": inp, "attention_mask": amask})[0]
        dec_ids, out = [2], []                                                            # start token = 2
        for _ in range(max_new):
            logits = sess["decoder_model"].run(["logits"], {
                "encoder_attention_mask": amask, "encoder_hidden_states": ehs,
                "inputs_embeds": embed(dec_ids)})[0]
            nxt = int(logits[0, -1].argmax())
            if nxt == 2:                                                                  # eos
                break
            out.append(nxt)
            dec_ids.append(nxt)
        return tok.decode(out, skip_special_tokens=True).strip(), time.time() - t0

    img = Image.open(sys.argv[1])
    for task in ("<CAPTION>", "<DETAILED_CAPTION>"):
        text, dt = caption(img, task)
        print(f"[{task} {dt:.1f}s] {text}")


if __name__ == "__main__":
    main()
