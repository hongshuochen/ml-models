# Face + Hand — Android (CameraX + TFLite)

Native Kotlin Android app that runs the **compact face+hand detector** on the live
phone camera, in real time, fully on-device. A **top-right** button toggles landmarks
(21-pt hand skeleton + 5-pt face points, run as a stage-2 crop→regress); a **bottom-right**
button flips front/back camera. Both are icon-only.

**Framing gesture:** when **both hands** make the "L" (thumb-index) pose, the app auto-detects
it (a rotation/flip-invariant landmark MLP, `l_gesture.tflite`) and draws a **quadrilateral**
from the two hands' thumb+index tips, **darkening everything outside** it. No button — it's
gesture-triggered.

**Face recognition (passive):** every detected face is recognized on-device and labelled live.
Detect → IoU-track → (on a new / lost-and-found / stale track) 5-point align → embed → match.
A face that's already on screen carries its identity forward via tracking, so it's embedded only
once per appearance (re-confirmed ~every 3 s), not every frame. Unknown faces are **auto-enrolled**
("Person N"). The **top-left** button opens a **gallery** of everyone seen (aligned crops; tap =
rename, ✕ / long-press = delete), with the **two most-recently-seen faces** shown as thumbnails
under it. The gallery persists to `filesDir/gallery/` across restarts.

- **Detector:** `app/src/main/assets/face_hand.tflite` — Pico-P4P5 + HaGRID, int8
  dynamic-range (~0.8 MB). NMS-free YOLO26: input `[1,640,640,3]` f32, output
  `[1,300,6]` = `[x1,y1,x2,y2,conf,cls]` (cls 0=face, 1=hand).
- **Face embedder:** `app/src/main/assets/face_embed.tflite` — ArcFace **MobileFaceNet**
  (InsightFace `w600k_mbf`) converted to TFLite, **fp32 (~13 MB)**. Input is an aligned
  `[1,112,112,3]` NHWC RGB face normalized `(x-127.5)/127.5`; output `[1,512]` raw → L2-normalize
  → cosine, recognition threshold **0.3**. Conversion was numerically verified bit-exact to the
  source ONNX (parity cosine 1.0); see `verify_mbf_tflite.py`. float16 (~6.8 MB, parity 0.999992)
  and dynamic-int8 (~3.6 MB, parity 0.9885) variants build to `runs/face_recog/` as drop-ins.
- **Landmark regressors:** `hand_landmark.tflite` (21-pt, ~0.25 MB) / `face_landmark.tflite`
  (5-pt, ~0.24 MB), MobileNetV3-small-025 with a trimmed **head-256** conv, **int8 dynamic-range**.
  The 5-pt model feeds both the landmark overlay and the ArcFace alignment. (int8 dyn-range needs
  `tensorflow-lite` ≥ 2.17.0 for its hybrid `FULLY_CONNECTED` v12 op — see CLAUDE.md.)
- **Gesture MLP:** `l_gesture.tflite` (~19 KB), int8 dynamic-range.
- **Camera:** CameraX `ImageAnalysis` (`KEEP_ONLY_LATEST`, RGBA_8888) on a background
  thread; `PreviewView` (cover-fit) with an `OverlayView` drawing the boxes
  (face = cyan, hand = red) and recognized names. Front camera by default, with a **Flip camera**
  button to switch front/back (the overlay mirror follows automatically).

## Build & run
1. Open the **`android/`** folder in **Android Studio** (let it sync Gradle / generate
   the wrapper). Requires JDK 17, Android SDK 34.
2. Plug in a device (or start an emulator with a camera) and **Run ▶**.
3. Grant the camera permission when prompted.

CLI alternative (after Android Studio has created the Gradle wrapper):
```bash
cd android && ./gradlew installDebug
```

## Files
```
app/src/main/
  AndroidManifest.xml                     camera permission + Main/Gallery activities
  assets/                                 bundled models: face_hand, face_embed,
                                          hand_landmark, face_landmark, l_gesture (.tflite)
  java/com/example/facehand/
    MainActivity.kt                       CameraX setup, permissions, per-frame analyze loop
    FaceHandDetector.kt                   detector TFLite load + preprocess + [1,300,6] decode
    LandmarkRegressor.kt                  generic 21/5-pt keypoint regressor
    GestureClassifier.kt                  rotation-invariant "L" gesture MLP
    FaceTracker.kt                        IoU + size-gate face tracking (when to (re-)embed)
    FaceAligner.kt                        5-pt similarity transform -> ArcFace 112x112 crop
    FaceEmbedder.kt                       MobileFaceNet 112x112 -> 512-d embedding (L2-normed)
    FaceGallery.kt                        persistent identity store (cosine match / auto-enroll)
    GalleryActivity.kt                    grid of enrolled faces (rename / delete)
    OverlayView.kt                        draws boxes, names, landmarks, framing (cover-fit + mirror)
  res/layout/activity_main.xml            PreviewView + OverlayView + buttons
```

## Swapping the model
Drop a different exported `.tflite` in `assets/` and update `FaceHandDetector.MODEL_ASSET`.
The float16 build (`pico_hagrid_float16.tflite`, ~1.3 MB) is a drop-in if you prefer it
over int8. Keep input 640² and the `[1,300,6]` output layout, or adjust the constants in
`FaceHandDetector`.

## Notes
- `androidResources { noCompress += "tflite" }` (in `app/build.gradle.kts`) keeps the
  models uncompressed so they can be memory-mapped from the APK.
- int8 dynamic-range runs fine here (Android TFLite/XNNPACK supports hybrid ops and gets an ARM
  speedup — unlike the browser's tfjs-tflite WASM, which needs float16). Its hybrid
  `FULLY_CONNECTED` is op v12, so the runtime is pinned to **2.17.0** (2.16.1 was too old).
- The 13.6 MB fp32 embedder loads on the analysis thread (not the UI thread) so it doesn't lengthen
  cold start; face recognition simply starts a beat after the camera. To shrink the APK, swap
  `face_embed.tflite` for the float16 build (~6.8 MB, indistinguishable accuracy).
- minSdk 24, targetSdk 34, Kotlin, CameraX 1.3.4, tensorflow-lite 2.17.0.
