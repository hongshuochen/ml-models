# Florence Caption — on-device image captioning (Android)

A standalone test app that runs **Florence-2-base-ft** (0.23 B, Microsoft) fully on-device with
**ONNX Runtime** to caption a picked image. Built to evaluate small on-device captioners; in a local
head-to-head Florence-2 beat **SmolVLM-256M** on both quality *and* speed (2–4 s vs 14–21 s on CPU,
fewer hallucinations), so this app ships Florence-2.

## What it does
**Take a photo** (system camera, full-res) or **pick an image** → choose a task (**Caption** /
**Detailed** / **More detailed**) → get an on-device caption + latency. No network at inference time
(only the one-time model download). Camera uses the system camera app via an intent (no CAMERA
permission); shots go to `cacheDir/captures` through a `FileProvider`.

## How it works
Pure ONNX Runtime, four components, **no KV cache** (the with-past export has a static-length bug;
the vision encoder runs once and the tiny BART decoder re-runs on the growing sequence — fine for
short captions). Pipeline (verified bit-for-bit against transformers in `florence_onnx_demo.py`):

1. **preprocess** — resize 768², ImageNet mean/std, CHW float32.
2. **vision_encoder** → image features `[1, 577, 768]`.
3. **embed_tokens** on the (hardcoded, pre-tokenized) task prompt → prompt embeds; prepend image
   features → `inputs_embeds`.
4. **encoder_model** → encoder hidden states.
5. **decoder_model** greedy loop (start token `2`, stop at `2`, ≤64 tokens), argmax last logit.
6. **decode** ids → text via GPT-2 byte-level scheme + `vocab.json` (no native tokenizer needed —
   DJL's isn't built for Android ABIs; decode parity verified against HF `tokenizers`).

| file | role |
|------|------|
| `MainActivity.kt` | UI: load/download model, pick image, run, show caption + latency |
| `FlorenceCaptioner.kt` | the 4-session ORT pipeline + no-cache greedy decode |
| `BartTokenizer.kt` | hardcoded task prompt ids + byte-level decode from `vocab.json` |
| `ModelStore.kt` | one-time download of the int8 ONNX + `vocab.json` to `filesDir` |
| `florence_onnx_demo.py` | the reference pipeline this app mirrors (run it to sanity-check) |

## Models (auto-downloaded on first launch, ~261 MB int8)
From `onnx-community/Florence-2-base-ft` → `filesDir/florence/`:
`vision_encoder_int8.onnx`, `embed_tokens_int8.onnx`, `encoder_model_int8.onnx`,
`decoder_model_int8.onnx`, `vocab.json`. (Swap the `_int8` suffix for `_fp16` in `ModelStore.kt`
for higher quality at ~518 MB.)

## Build & run
1. Open **`android-caption/`** in Android Studio (JDK 17, Android SDK 34); let it sync + generate
   the Gradle wrapper.
2. Run ▶ on a device with internet (first launch downloads the models; subsequent launches are
   offline). Pick an image.

> Not yet device-tested (built on a headless box). Expect to iterate on ORT tensor/IO details in
> Android Studio — the pipeline itself is verified in `florence_onnx_demo.py`.

## Notes
- int8 is faster on ARM (phones) than the x86 numbers above; expect a few seconds/caption.
- Latency is dominated by the vision encoder (once) + the no-cache decode (re-runs the small
  decoder per token). A correct KV-cache export would speed up decode further.
