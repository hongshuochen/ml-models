# android-golf

Standalone Android app: the **Golf hit counter** (putts + full swings), split out from the
face/hand app in `../android/` so the two are fully separate (own `applicationId`
`com.example.golf`, own APK, install side by side).

- **What it does / how it works:** see `../GOLF_APP.md`.
- **The detector model:** see `../GOLF_YOLO.md` (`app/src/main/assets/golf.tflite`, f16, 18.9 MB).

## Build

```bash
JAVA_HOME=/opt/android-studio/jbr ./gradlew assembleDebug
# -> app/build/outputs/apk/debug/app-debug.apk  (~40 MB)
```
Needs Android SDK platform-34 (`local.properties` → `sdk.dir=...`).

## Source (`app/src/main/java/com/example/golf/`)

| file | role |
|---|---|
| `GolfActivity.kt` | CameraX → detector → counter → UI (the only launcher) |
| `GolfDetector.kt` | golf TFLite detector (ball + club_head) + `Detection` |
| `HitCounter.kt` | ego-compensated v3 hit algorithm (online port) |
| `GlobalMotion.kt` | local block-match camera-motion estimate (~2 ms) |
| `OverlayView.kt` | draws boxes (ball cyan / club_head amber) + HUD |

Not yet field-tested on device.
