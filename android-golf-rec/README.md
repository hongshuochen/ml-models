# android-golf-rec

Standalone Android app: a **golf swing auto-recorder**. You (e.g. wearing AR glasses / holding a
phone) watch a friend; when the app sees them get into the **address** posture with a club, it
prompts **"record?"** — on yes it records the swing and saves it.

Separate app (`applicationId com.example.golfrec`), independent of `../android-golf/`.

## How it works

```
CameraX (rear cam) ─┬─ ML Kit Pose (33 landmarks of the friend)  ─┐
                    └─ ClubDetector (our club_head, third-person) ─┘
                         → AddressDetector (hands together + low + arms down + still ~1s + club at hands)
                         → prompt "record?" → CameraX VideoCapture → Movies/GolfRec
```

- **Subject is third-person** → uses the **public-trained** club model (`golf_detect_s_v3`, int8),
  NOT the egocentric `golf_ego_v2` — measured: ego detects 0 club heads in third-person, public 4/6.
  `app/src/main/assets/club.tflite` is that model.
- **AddressDetector.kt** — pure geometry on the pose landmarks (body-relative thresholds), so it
  holds across how big the person appears. Tune on-device.
- **Recording** = ask-before (prompt on address → record on yes); auto-stops after 12 s.

## Build

```bash
JAVA_HOME=/opt/android-studio/jbr ./gradlew assembleDebug
# -> app/build/outputs/apk/debug/app-debug.apk
```
Needs Android SDK platform-34. Not yet field-tested — the pose thresholds (`AddressDetector`
constructor) and `CLUB_NEAR` will likely need tuning against real swings.

## Source (`app/src/main/java/com/example/golfrec/`)

| file | role |
|---|---|
| `RecActivity.kt` | CameraX preview+analysis+video, ML Kit pose, prompt→record→save |
| `AddressDetector.kt` | pose geometry → address-posture trigger (still + hands-low + club) |
| `ClubDetector.kt` | our club_head TFLite detector (public third-person model) |
| `OverlayView.kt` | draws the detected boxes |
