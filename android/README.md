# Face + Hand — Android (CameraX + TFLite)

Native Kotlin Android app that runs the **compact face+hand detector** on the live
phone camera, in real time, fully on-device. (Detector only — no landmark stage yet.)

- **Model:** `app/src/main/assets/face_hand.tflite` — Pico-P4P5 + HaGRID, int8
  dynamic-range (~0.8 MB). NMS-free YOLO26: input `[1,640,640,3]` f32, output
  `[1,300,6]` = `[x1,y1,x2,y2,conf,cls]` (cls 0=face, 1=hand).
- **Camera:** CameraX `ImageAnalysis` (`KEEP_ONLY_LATEST`, RGBA_8888) on a background
  thread; `PreviewView` (cover-fit) with an `OverlayView` drawing the boxes
  (face = cyan, hand = red). Front camera by default (overlay mirrored to match).

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
  AndroidManifest.xml                     camera permission + launcher activity
  assets/face_hand.tflite                 the bundled detector model
  java/com/example/facehand/
    MainActivity.kt                       CameraX setup, permissions, per-frame analyze loop
    FaceHandDetector.kt                   TFLite load + preprocess + [1,300,6] decode
    OverlayView.kt                        draws boxes (cover-fit + front-camera mirror)
  res/layout/activity_main.xml            PreviewView + OverlayView
```

## Swapping the model
Drop a different exported `.tflite` in `assets/` and update `FaceHandDetector.MODEL_ASSET`.
The float16 build (`pico_hagrid_float16.tflite`, ~1.3 MB) is a drop-in if you prefer it
over int8. Keep input 640² and the `[1,300,6]` output layout, or adjust the constants in
`FaceHandDetector`.

## Notes
- `androidResources { noCompress += "tflite" }` (in `app/build.gradle.kts`) keeps the
  model uncompressed so it can be memory-mapped from the APK.
- int8 dynamic-range runs fine here (Android TFLite/XNNPACK supports hybrid ops and gets
  an ARM speedup — unlike the browser's tfjs-tflite WASM, which needs float16).
- minSdk 24, targetSdk 34, Kotlin, CameraX 1.3.4, tensorflow-lite 2.16.1.
