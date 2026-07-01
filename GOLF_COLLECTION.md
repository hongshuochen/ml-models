# Egocentric Golf Data-Collection Plan — ball + club_head (2-class)

**Goal:** close the first-person viewpoint gap. Third-person YOLO26x teacher auto-labels; we correct. Split by **session**, never by frame.

## Diversity budget (cover the axes)
| Axis | Count | Values |
|---|---|---|
| **People** | **6** (incl. ≥1 **left-handed**; mix skill / height / build) | egocentric diversity = swing speed, handedness, framing height — not faces |
| **Clubs** | **6** — driver, 3-wood, hybrid, 7-iron, wedge, **putter** | span club-head shapes; putter gives the most usable ball frames |
| **Balls** | **4** | glossy **white** (over-collect — hardest), optic-yellow, striped range, scuffed/dirty white |
| **Environments** | **4** | bright range, putting green, indoor mat/net, outdoor grass/backyard |
| **Lighting** | 3–4 across the 4 envs | midday sun, overcast, open shade, indoor — always brightest available |

**Spread the axes, don't cross them.** 6×6×4×4 = 576 cells is impossible in 2 h — instead distribute so every
value appears **≥3–4 times across the whole set**: each person does a **rotating subset** of clubs/balls/envs, and
every club / ball / env is hit by **several different people**. Priority pairings to force: **white ball × every
environment**, and **left-hander × ≥2 environments**.

## Capture settings
- **1080p30, H.264** (pin it; not 4K/60/ProRes). Mount = **cap-brim / collar clip at eye-line**, tilted **down 30–45°** so the ball (4–6 ft, near feet) stays in frame. **No tripods, no chest mount, no hand-hold during swings.**
- **Framing check** (per new person/env): at address, ball + club_head both fully in frame, ball ≈ fingernail-sized (**≥12–16 px in 640 space** ≈ ≥24–30 px at 1080p). Do a **10-s test-record + playback** first.
- **Exposure:** manual app, **shutter ≥1/1000 s, ISO floats**. Sun **behind camera**; keep **your body-shadow off the ball** (orient shadow to the side). Indoor/overcast = static/slow only.
- Record **long continuous takes (2–5 min)** per (club,ball); split later. **Voice-slate each take** ("Person 2, driver, yellow, range").

## Shot menu (natural, per person)
| Type | Notes |
|---|---|
| **Address holds** | ball + head sharp — highest value |
| **Putts** | trackable through stroke + roll (build in retrieval time) |
| **Full swings** (range/net only) | record continuously; keep natural **mishits** (tops/chunks) — don't stage whiffs |
| **Grip / waggle / walk-up** | free club_head motion |
| **Negatives (10–15% of frames)** | leave rolling between shots; add **hard negatives**: ball buckets, sprinkler heads, daisies, yardage markers, alignment sticks, bag handles |

Log **lie** per block: rubber tee/mat, real tee/turf, ball on green, ball on mat.

## Metadata (per BLOCK, voice-slated; reconcile to CSV later)
`block_id, person, handedness, device, mount, club, ball_color, environment, lighting, lie`
Auto-derive clip_id/fps/res/timestamp from file.

## Sampling + annotation
- **Address holds/putts:** dedup (perceptual-hash) to **2–3 distinct frames per hold**. Swings: 2 fps, but **only sharp frames survive** (assume 40–60% of swing frames unusable). Keep ±0.3 s impact window on disk for later hand-marking; push only **2–3 sharp frames** into the label set.
- **Pilot first:** hand-label ~200–300 egocentric frames; confirm teacher's "club head" = head-only (hosel→toe, no shaft); measure ball vs club_head recall → set real correction budget.
- **Correct:** ~5–10 s/frame (teacher is blind here; many boxes from scratch). Follow a **one-page labeling guide + crops** (ball = full visible disk only; blurred club_head = tight blur-envelope box). QA re-review random 5%.
- **Target ≥2–3k ball instances**; report final club_head:ball ratio.

## Budget (rolling capture; wall-clock ≈ 1 day of sessions)
| Item | Amount |
|---|---|
| Rolling capture | **~2 h min, ~3 h recommended** for 6 people (≈ 20–30 min/person) |
| Per person | ~20–30 min: 60% address/putt, 40% swings; a **rotating 2–3 clubs + 1–2 balls** in whatever env they're in |
| Usable frames | ~4–6k (address/putt dominate) |
| Storage | ~25–35 GB raw + backup |
| Auto-label pass | minutes (3080) |
| Human correction | ~15–25 h |

> With 6 people the axes are covered **across people**, so 2 h is enough for good coverage but thin per cell; budget
> **~3 h** if you want each club/ball well-represented. Don't make one person grind all 6 clubs — rotate.

## After capture
1. Offload + verify, immutable backup of **RAW** + CSV (needed for re-sampling/impact marks). 2. Auto-label → correct → export YOLO (ball=0, club_head=1). 3. **Held-out split disjoint on person AND environment AND club.** 4. Fine-tune student on teacher + our frames with **egocentric upsampled** so ~3–4k ours isn't drowned by 6350 third-person; compare vs teacher on the egocentric split.

---
*Full rationale + the third-person teacher setup: see `GOLF_PLAN.md`. This one-pager reflects the narrowed 2-class {ball, club_head} scope, 30 fps capture, phone compute, and no ground-truth gear.*
