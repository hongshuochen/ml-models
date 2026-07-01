# Golf CV on AR Glasses — Unified Execution Plan
*(ball / club-head / club-shaft detection + impact + swing/shot analysis, egocentric moving camera, YOLO26 / uv / RTX 3080)*

All four reports agree on the dominant fact: **every public golf dataset is third-person (broadcast / tripod, face-on or down-the-line). None is egocentric.** Public data buys *appearance priors + analytics/event supervision*; your ~2 h of AR-glasses footage is the only thing that closes the viewpoint gap — the exact synthetic-QR-collapses-on-real-photos lesson already in CLAUDE.md.

---

## 0. Locked constraints (confirmed 2026-06-30) — READ FIRST

These override any conflicting recommendation below.

- **Camera = custom AR glasses @ 30 fps; glasses only capture and STREAM video to a PHONE that does all compute.** The **phone camera is the immediate demo device** — we can start collecting + prototyping NOW, no glasses hardware needed to begin. (Caveat: phone-held ≠ head-mounted egocentric; some real head-mounted data still needed before final deploy.)
- **Model input assumed 640×640** (flexible). Fine for `club_head` + `club_shaft` (both large in-view). Tight for the ball → see ball note.
- **30 fps KILLS ball-in-flight tracking outright** (ball moves >½ frame per frame; even worse than the 60 fps analysis below). → **Ball scope = address (static) + putting (slow, trackable) + at most 1 post-impact frame.** Do NOT promise ball-flight tracking, trajectory, carry, or apex.
- **No ground-truth gear** (no launch monitor, no 240 fps phone). → ball-speed / launch-angle become *unvalidated estimates* — de-prioritize. **Ship impact-frame + swing tempo**: both self-labelable, need zero external ground truth. Impact GT = manual from 30 fps video (±1 frame ≈ ±33 ms ceiling).
- **Research / personal use** → all datasets usable, incl. GolfDB (CC BY-NC) and gated/unlicensed sets; no license blocker.
- **Compute target = phone** (not a tiny glasses NPU) → model budget is generous: YOLO26n/s (even m) via TFLite GPU/NNAPI. The "1280 + SAHI" advice below applies to the **offline teacher / auto-labeler**, not the on-phone student.

**Net effect on scope:** the product is **club tracking (head + shaft) + impact detection + tempo + putt tracking**, with the ball as a static/putt-only object — NOT a ball-flight analyzer. Everything below still holds with that framing.

---

## A. TL;DR (recommended end-to-end)

- **Three objects, three representations, not one uniform box:** `ball` → detect (tiny/blurred, high-res); `club_head` → detect (1 class, don't split by club type); `club_shaft` → **pose keypoints (2 endpoints: grip-butt + hosel)**, because a thin diagonal segment is a line, not a box.
- **Two student models, one shared later:** (A) YOLO26-pose club (shaft kpts + head) @640–960; (B) YOLO26-detect ball+head @**1280 + P2 head / SAHI**. Prototype first with a single 3-class YOLO26-detect @1280 to get a baseline and mine hard frames.
- **Teacher→student:** YOLO26x-pose + YOLO26x-detect @1280 (+SAHI), trained on public data, used to **auto-pseudo-label your footage**; distill to YOLO26n/s, export **fp16 TFLite** (int8 pose is non-viable per repo; ship fp16).
- **The ball is physically un-trackable in flight** on consumer glasses (jumps ¼–½ frame/frame at 30–60 fps, shrinks to ~1 px in meters). Treat ball as reliable only **at address (static)** and **1–2 launch frames + putts**. Adopt **BlurBall center-of-streak labeling**.
- **Impact detection = audio-anchored, video-refined.** The mic transient gives sub-frame impact timing; video (min club-head↔ball distance + ball present→absent) confirms. Do **ego-motion compensation (IMU + background flow) first** — a static ball looks moving when the head turns.
- **Audio + IMU are first-class sensors**, not extras. They carry impact timing and swing phase that 30–60 fps video cannot.
- **Ship the robust wins:** impact frame, swing **tempo** (backswing:downswing ratio — camera-invariant, best single deliverable), reduced event set, coarse launch **direction/angle**. **Don't over-promise** spin, face angle, carry — those are radar/1000 fps territory, physically out of reach.
- **Capture 1080p @ 60 fps in bright light**, imgsz ≥ 960–1280, ~6–8k real labeled frames incl. ~20% negatives; always eval on a **real held-out set**.

---

## B. Model & Task Choice

**Teacher (offline, RTX 3080 — auto-label + analysis):**
- YOLO26x-pose @1280 for club keypoints; YOLO26x-detect @1280 **+ SAHI** (overlapping 640/832 tiles, ~0.2 overlap, +12–14 AP on small objects) for ball/head.
- Pseudo-labels your footage; human-verify a subset (especially blurred-ball / impact frames).

**Student (on-device, fp16 TFLite/ONNX):**

| Class | Task / representation | imgsz | Why (one line) |
|---|---|---|---|
| `ball` | detect, **P2 (stride-4) head**, BlurBall center-of-streak label + optional blur len/angle regression | **1280** (or high-res crop / temporal heatmap) | Few-pixel blurred object dies at 640 (repo's small-QR gotcha); accept ~4× latency here — ball is the hard part |
| `club_head` | detect, **single class** (no driver/iron/wedge split) | 960–1280 | Coarse classes = less label cost, more data/class; gate head detections by proximity to predicted hosel kpt for temporal stability |
| `club_shaft` | **pose, 2 kpts (grip-butt + hosel)**, visibility flags; optional 3–4 (add midpoint) | 640–960 | Two endpoints give position+angle+**length**, no OBB angle-wrap, occlusion-robust; hosel kpt = where head attaches (geometric prior). Beats OBB (◐) and AABB (✗, ~90% empty box) |

**Model scale reference (COCO detect @640):** n = 40.9 mAP / 1.7 ms T4 / 2.4 M params; s = 48.6 / 2.5 ms / 9.5 M; x = 57.5 / 11.8 ms. Latency ≈ (imgsz/640)² → n@1280 ≈ ~7 ms T4. Start **n@1280 for ball/head**, **n/s@640–960 for club-pose**; s buys +7–8 mAP at ~1.5× latency.

**Advanced option for the ball:** a **TrackNet/WASB-style multi-frame heatmap head** (3 stacked ego-motion-compensated frames → Gaussian heatmap) is SOTA for tiny/fast/blurred sports balls (97%+ on tennis/shuttle). Adopt the *architecture + loss*, retrain on golf — do **not** reuse tennis frames. Keep the pure-YOLO high-imgsz+SAHI detector as teacher/baseline.

YOLO26 is **NMS-free / end-to-end**: detect `(1,300,6)=[x1,y1,x2,y2,conf,cls]`, pose `(1,300,69)`. Keep post-processing consistent. fp16/fp32-consistent → fits repo's "no int8 hybrid on web/edge" rule.

---

## C. Public Datasets to Use NOW

**Ball-class pretrain (merge + dedupe — the Roboflow sets overlap):**
- Roboflow `golfball` yolo-qpdqh, ~31k train / ~38k total — https://universe.roboflow.com/yolo-qpdqh/golfball-3g99x
- Roboflow `golf-ball-tracker`, 3,944 — https://universe.roboflow.com/golf-balls/golf-ball-tracker-sksye
- `rucv/golf_ball`, 2,169 broadcast tiny-ball (unlicensed → research only) — https://github.com/rucv/golf_ball · https://arxiv.org/abs/2012.09393

**Club-head + our 3-class layout (closest schema match — warm-start):**
- Roboflow `golf-driver-tracker` (salo-levy), 2,646, **classes ball / club-handle / club-head** — https://universe.roboflow.com/salo-levy-nlqrn/golf-driver-tracker
- Roboflow `golf-club-tracking` v2, 6,750 club-head — https://universe.roboflow.com/club-head-tracking/golf-club-tracking/dataset/2

**Shaft class (no clean box dataset exists — bootstrap only):**
- `golf_club_pose` keypoints (shaft/grip/head) — https://universe.roboflow.com/golfswing-e1qwd/golf_club_pose → convert line to endpoints/OBB
- `pronisi` segmentation (only source giving shaft as pixels) — https://universe.roboflow.com/pronisi/golf-club-detection-1hgid
- **GolfPose/GolfSwing** (ICPR 2024), 13,782 imgs, 5 club kpts incl. shaft — **gated, email author** — https://github.com/MingHanLee/GolfPose

**Impact / event + analytics supervision (not detection labels):**
- **GolfDB / SwingNet**: 1,400 videos, 8 events incl. Impact; **CC BY-NC 4.0, YouTube-sourced (non-commercial)**; bbox is ONE per-video box, not per-frame — https://github.com/wmcnally/golfdb · https://arxiv.org/abs/1903.06528
- **CaddieSet** (2025): 924 shots, pose+events+**launch-monitor ball metrics** → pose→outcome analytics — https://arxiv.org/html/2508.20491v1 · https://github.com/damilab/CaddieSet

**Fast-ball method (not data):** TrackNetV2/V3 heatmap+temporal — https://github.com/nickluo/TrackNetV3 · https://arxiv.org/abs/1907.03698

**License hygiene:** GolfDB non-commercial; rucv/GolfPose gated/unlicensed; Roboflow licenses are **per-dataset — verify each page** before shipping.

**GAPS only your own data fills:** (1) egocentric moving-camera viewpoint; (2) shaft as a first-class target; (3) per-frame *synchronized* ball+head+shaft labels through impact; (4) egocentric impact event (camera jerks, ball vanishes in 1–2 frames); (5) head-cam motion-blur/rolling-shutter; (6) eye-height course diversity (white ball on bright turf); (7) egocentric launch geometry for shot analysis.

---

## D. Data-Collection Plan (answering the 4 questions)

**The physics that reshapes everything:** driver ball ~67–76 m/s; at 60 fps ball moves ~1.1 m (~530 px, 27% of frame)/frame, at 30 fps >½ frame; shrinks to ~1 px within meters. Ball ~20 px (1080p) / ~30 px (3K) at address. **→ ball reliable only at address + 1–2 launch frames + putts.** Ray-Ban Meta auto-exposes → **bright light is your only blur lever.**

**Direct answers:**
- **People: 4** (min 3, stretch 6) — P1 RH mid-handicap; P2 RH fast (worst-case blur); **P3 left-handed** (mirror geometry, ~absent from public data); P4 slow/high-handicap+mishits+different body type. Egocentric diversity = swing speed (blur), handedness (geometry), height (framing) — not faces.
- **Club types: 6** — **Driver, 7-iron, wedge, putter** (four corners of head-shape space, all High priority) + 3-wood + hybrid/4-iron (interpolate, optional). Shafts look near-identical → few needed for `club_shaft`.
- **Ball types: 4** — **glossy white** (90% of play, hardest: blows out on white mat/sky — over-collect this), yellow/optic (easy anchor), striped range ball, scuffed/dirty white.
- **Environments: 4 × 5 lightings** — indoor net/sim, driving range, outdoor grass, putting green; cover **bright sun / overcast / open shade / indoor artificial / dusk**. Deliberately capture hard background×ball contrasts (white-on-white-mat, white-on-sky, ball-on-grass, ball-on-dark-net).

**2-hour budget (120 min raw capture):**

| Person | Indoor net/sim | Range | Outdoor grass | Putting | Subtotal |
|---|---|---|---|---|---|
| P1 RH mid | 8 | 10 | 6 | 4 | 28 |
| P2 RH fast | 6 | 12 | 4 | 2 | 24 |
| P3 **LH** | 6 | 8 | 6 | 4 | 24 |
| P4 slow/mishits | 6 | 6 | 8 | 4 | 24 |
| Shared negatives/static-holds | 5 | 5 | 5 | 5 | 20 |
| **Total** | 31 | 41 | 29 | 19 | **120** |

Range gets the most (widest lighting + full ball speeds); putting green small but essential (only place the ball is frame-to-frame trackable + putter shape). Include **mishits** (topped/chunked/shank/whiff — high value) and **~20% negatives** (practice swing no-ball, empty mat/sky/net, white distractors like tees/markers — this repo already burned on hallucinated detections).

**Capture spec:** **primary 1080p @ 60 fps** (temporal > spatial for impact/club tracking; 20 px ball still detectable at imgsz ≥960); **secondary 3K @ 30 fps** for address/putt detail crops — record both in pilot and compare. Native 100° FOV, tilt head so ball sits lower-center. Favor **bright sun** (forces ~1/1000–1/2000 s shutter → ~16–30 px blur vs a smeared indoor ball). Rolling-shutter caveat: prefer raw/unstabilized; a Project Aria unit (global shutter) is an optional cleaner supplement, not the deployment target. Storage: 1080p60 ~5.4 GB/hr, 32 GB onboard fits 2 h but offload every 20–30 min; watch ~3-min per-clip cap.

**Sampling (dense-at-impact):** drop idle/walking (~40–50% of raw); address holds ~2–5 fps; **impact burst ±0.3–0.5 s at full 60 fps** (audio-click timestamps the window); backswing/follow-through ~5–10 fps deduped.

**Yield:** ~200 swings → ~25–40 useful frames/swing → **~6,000 positive + ~1,500 negative labeled frames** (roughly doubles the public sets). Ball will be under-represented (~3,500 instances vs ~5,000 head/shaft) → mitigate with static-hold/putt blocks, public ball data, and copy-paste ball augmentation.

**Annotation:** teacher pre-labels → **CVAT (Ultralytics/SAM auto-annotate) or Roboflow (SAM label-assist)** human correction → active-learning ordering (review low-confidence / impact / blur / white-on-white first; auto-accept high-confidence statics).

---

## E. Hit-Detection + Analysis

**Impact algorithm (audio-anchored, video-refined, multi-cue fusion):**
1. **Audio onset (primary, sub-frame):** spectral-flux / high-freq energy of the mic transient → `τ_hit` to ~1 ms. The strike is a sharp broadband "leading-edge click" (impact-sound patents US6149532, US10478700).
2. **Video evidence (confirm / fallback):** after **ego-motion compensation** (subtract background flow / IMU rotation), use (A) min club-head↔ball distance, (B) ball present→absent / velocity-onset discontinuity, (C) club-head speed peak + direction reversal.
3. **Small temporal head** (1D CNN / bi-LSTM, SwingNet-style) over per-frame features `[ball, head, distance, Δball, Δhead, shaft angle, confidences, IMU |ω|, audio energy]` → P(impact|t).
4. **Fuse:** `t_hit = argmax P_model(t)·gaussian(t − t_audio)`, report sub-frame via audio phase.

Ego-motion compensation is **mandatory** — impact is a non-smooth discontinuity that survives after subtracting smooth head-induced flow. Audio degrades outdoors/windy/nearby players → keep the video model strong enough to stand alone (GolfDB shows Impact is the *easiest* event third-person, PCE ~98%; egocentrically it's harder visually but audio equalizes).

**Swing events:** don't force all 8 (Toe-up / Top / Mid-backswing are Low/None egocentrically — club leaves FOV above the head). Train an **egocentric SwingNet variant on detector-feature + IMU sequences** targeting a **reduced reliable set {Address, Takeaway, Mid-downswing, Impact, Follow-through, Finish}**, mask out-of-FOV events. Fuse IMU heavily (single-IMU swing-phase work: arXiv 2506.17505).

**Shot analysis — what's feasible monocular:**
- **FEASIBLE:** impact timing (High); **swing tempo & total time** (High — pure timestamps, camera-invariant, *best single deliverable*); launch **direction** (Medium — first 1–3 ball positions vs aim line + IMU); **vertical launch angle** (Medium — precedent ±0.74° from blur streak + known ball size + gravity/IMU); ball **speed** (Low-Med — pixel displacement or streak-length÷exposure, scale off 42.67 mm ball, ~5–15% error); club-head speed & swing path (Low-Med, coarse/relative).
- **NOT feasible from glasses:** spin rate/axis (needs >>1000 fps or radar), club-face angle/dynamic loft (small/blurred/occluded), precise 3D trajectory/carry/apex (monocular depth ambiguous, ball exits FOV). Curve/shot-shape follows spin → also out.
- **Scale trick:** use the **42.67 mm ball** (and known club length) as an always-present in-frame ruler for every metric.

**Labels to collect:** (1) impact-frame index (audio-seeded, verified vs 240 fps phone/launch monitor on a subset); (2) reduced 8-event indices, out-of-FOV masked; (3) per-frame ball/head/shaft boxes densified at impact; (4) post-impact ball centroids + blur-streak endpoints/length; (5) **co-recorded launch-monitor ground truth** (ball speed, launch angle/direction, spin, club speed, smash) — both regression targets and the honest measuring stick; (6) time-synced IMU (angular rate + gravity); (7) per-swing metadata (club, handedness, indoor/outdoor, ball type, **fps + exposure**, lighting). **Priority: always-on audio + IMU + logged exposure; a launch monitor for a substantial fraction; a 240 fps phone for a smaller sub-frame-GT subset.**

---

## F. Phased Roadmap (aligned to uv / Ultralytics / RTX 3080)

| Phase | Effort | Actions | Exit criterion |
|---|---|---|---|
| **0. Pilot** | ~1 h | Record 5–10 min, one golfer. Validate: ball ≥15 px at address? impact in FOV with natural head-tilt? blur acceptable? 1080p60 vs 3K30? teacher fires on frames? | Settings locked only if all pass |
| **1. Bootstrap teacher (public)** | 2–3 d | `uv run yolo` — pull ball sets + golf-driver-tracker + club-tracking + club-pose/seg; **remap all to 3 classes** (ball / club_head / club_shaft, shaft from kpts/masks); train **YOLO26x teacher @1280 (+SAHI)**; new `runs/` per model, config + TRAINING.md command committed | Teacher usable as pre-labeler |
| **2. Collect** | 1–2 d | Full 2 h per §D table (4 people / 6 clubs / 4 balls / 4 envs / 5 lightings + mishits + negatives); always-on audio + IMU; offload every 20–30 min | ~120 min captured |
| **3. Auto-label + correct** | 3–4 d | Teacher pre-labels → CVAT/Roboflow human correction; active-learning order; audio-click impact indexing | ~6–8k clean frames + **real held-out test split** |
| **4. Train students** | 1–2 d | Baseline: single YOLO26n/s-detect 3-class @1280. Then split: (A) YOLO26-pose club @640–960, (B) detect/heatmap ball+head @1280+P2/SAHI. Heavy aug (copy-paste ball, motion-blur, brightness); distill from teacher (feature+logit KD); **export fp16 TFLite** | Converges, exports clean |
| **5. Eval on REAL set** | 1 d | Per-class mAP on real held-out (never trust train/synthetic metrics — repo rule); focus **ball recall**, mishits, white-on-white, LH, blurred-ball frames; impact PCE (δ=1 frame) + absolute ms error | Ball recall + club mAP meet bar |
| **6. Analytics heads** | 2–3 d | Temporal impact head (audio+IMU+detector features); reduced-set event model; tempo/launch-direction estimators; validate vs launch-monitor GT | Impact + tempo shippable |
| **7. Iterate** | ongoing | Mine failures (small/fast/specular ball, shade, mishits), targeted re-collection, retrain | Failure modes closed |

**Repo ties:** eval on real held-out always; imgsz ≥960–1280 or SAHI for the ball (640 misses small objects); YOLO26 NMS-free post-processing; fp16 not int8 for pose; spawn fresh worker per model in web app if deployed there; separate `runs/` per model, commit configs + exact TRAINING.md commands.

---

## G. Open Decisions / Risks to Confirm

1. **AR-glasses fps/shutter — the hard physical limit.** Ray-Ban Meta = 1080p@60 / 3K@30, rolling shutter, auto-exposure — **cannot freeze the ball** (1–2 in-flight samples, streaks). Confirm your exact device/firmware and whether you can force short exposure or access a global-shutter unit (Project Aria). This gates everything downstream.
2. **On-device real-time vs offline split.** Recommended: lightweight per-frame detection on-device; trajectory + ego-motion comp + physics + analytics **offline/teacher**. Confirm the deployment latency target (≥60 fps NPU?) and whether analytics can be offline.
3. **Indoor vs outdoor first.** Indoor net/sim gives controlled lighting + easy impact audio (fewer nearby hitters) → recommend piloting there; confirm.
4. **Do you have a launch monitor + 240 fps phone?** Ground truth for §E error bars and impact sub-frame GT — without them, ball-speed/launch-angle claims stay unvalidated. Confirm availability.
5. **IMU + raw audio access on the glasses.** The whole impact + ego-motion design assumes time-synced IMU (angular rate + gravity) and raw mic with known offset to video PTS. Confirm the SDK exposes these; if not, impact falls back to video-only.
6. **Commercial intent.** GolfDB is CC BY-NC; several Roboflow/GolfPose sets are gated/unlicensed. If this ships commercially, verify each license before including — non-commercial sets are teacher-only.
7. **Scope honesty on shot metrics.** Confirm stakeholders accept that spin, face angle, and true carry are **out of reach** on glasses — promise impact/tempo/events/coarse-launch, not launch-monitor parity.

*Where reports differed:* Report 4 leaned "single 3-class detector"; Reports 2/3 argued shaft-as-pose + split heads — I took **split (pose shaft + detect ball/head)** because a near-1-D shaft is genuinely better as endpoints (no angle-wrap, occlusion flags) and the ball needs a different imgsz/temporal treatment than the club, but kept Report 4's single-detector as the fast Phase-4 baseline.