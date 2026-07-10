"""Golf HIT detector v3 ("ego-fusion") — counts hits (putts AND full swings) from per-frame
{ball, club_head} boxes, with optional per-frame camera-motion affines for ego compensation.

Why v3: the v2 raw-image-motion detector fired on WALK-AWAYS (head turns over a world-static
ball -> the ball's image track drifts out of the frame, indistinguishable from launch+vanish in
raw pixels) and missed SHORT PUTTS (a 2-4 m practice putt stops at the hole ~10 D away, never
reaching a safe raw-motion confirm distance). Both problems share one root: image motion mixes
BALL motion with CAMERA motion. v3 subtracts the camera (background optical-flow affines,
golf/cam_affine.py): a world-static ball measures ~0.05 D/frame TRUE speed under a walking head
vs ~1.05 for a real putt roll, so a low 5 D TRUE-displacement confirm is safe — that one change
kills the walk-away false fires AND recovers short-putt recall.

v3 merges the two winners of an adversarial 5-design eval (each 12/12 on the dev oracle, each
6/8 on 4 blind held-out clips with complementary misses; this fusion targets the 8/8 union):

  TRACKING   gated primary-ball track: association jump gate while tracked; while LOST the last
             position is PROPAGATED by the camera affine and re-acquired ONLY near that spot
             (never teleports to basket/background balls); short flicker gaps are bridged.
  ADDRESS    TRUE (ego-compensated) per-frame speed < REST_THR for REST_RUN_S, with a club_head
             seen near the ball recently (club-at-ball MEMORY: the club head is often motion-blur
             invisible AT impact, so it is never required at the fire itself). The anchor is
             PROPAGATED through every frame's affine so it stays glued to the ground spot.
  FIRE       (a) LAUNCH: TRUE displacement from the propagated anchor >= LAUNCH_D within
                 LAUNCH_WIN_S of leaving rest (putt/roll; safe only because it is ego-true);
             (b) VANISH-IN-PLACE: track lost >= GONE_S with the last sighting still AT the
                 anchor (full swing / tap-in: ball gone within 1-2 frames), suppressed when the
                 post-loss camera drift says the ball left the VIEW, not the SPOT (exit-ray to
                 the border; bottom-exit with downward drift is allowed = head lifting at the
                 strike). A bounded look-ahead veto window follows both paths.
             (c) DEPARTED-THEN-LOST: the ball had already left the anchor (>= VANISH_NEAR_D,
                 moving) when the track died — a mid-roll loss confirms the roll; NO exit-ray
                 test (a panning head follows a real putt, which looks exactly like an exit).
  VETOES     continuous-track return (the SAME tracked ball comes back to the anchor => practice
             putt / reposition; a DISCRETE reappearance after full absence = re-teeing, no veto;
             sparse in-cup sightings of a holed ball do not veto), persistent-reappearance check
             on vanish, no-grow guard (a struck ball recedes; a picked-up ball approaches the
             camera), cooldown. Fire times are BACK-DATED to the departure frame (~contact).

cams=None (no optical flow, e.g. the Android port before it grows a global-motion estimator)
degrades to the validated raw preset: stricter stillness (walking never holds a ball image still
0.5 s), and the pure-launch path requires a far/vanish confirm instead of 5 D. Short-putt recall
drops; walk-away rejection then rests on the exit/return guards alone.

All thresholds are in ball diameters (D = 2*radius) and seconds, converted per-clip via fps;
pixel sizes scale with frame width. Offline reference (bounded look-ahead <= ~2 s); an online
port defers the count by the same window (see android/.../PuttCounter.kt).
"""
import math
import os

_DBG = bool(os.environ.get("HIT_DBG"))

# ---- tunables (ego preset; physical units) ----
RMAX_FRAC   = 58 / 2048.0  # of frame width: a real address ball is never bigger (phone/hand/basket)
RMIN_CAND   = 5.0    # px: smaller ball boxes are detector noise
RMIN        = 6.0    # px: floor on the radius used as a divisor
DEDUP_D     = 1.0    # D: candidate centers closer than this are duplicate boxes of one ball
GATE_STEP   = 5.0    # D/frame: max association jump while tracked (a putt rolls ~1 D/fr; more = teleport)
REACQ_D     = 2.5    # D: while LOST, re-acquire only this close to the ego-propagated last position
GAP_MAX_S   = 0.13   # s: detection gaps up to this are flicker (bridged; rest run survives)
LOST_MAX_S  = 2.0    # s: LOST longer than this -> the track dies (a fresh track may then seed anywhere)
REST_THR    = 0.25   # D/frame TRUE speed under which the ball is at rest (world-static measures 0.03-0.11)
REST_RUN_S  = 0.17   # s of consecutive TRUE rest to arm an address
ADDR_HOLD_S = 1.5    # s: an armed address stays live this long after the last rest frame
CLUB_NEAR_D = 3.5    # D: club_head within this of the ball = "club at the ball"
CLUB_MEM_S  = 3.0    # s: club-at-ball memory (covers backswing + blur-invisible impact)
LAUNCH_D    = 5.0    # D: TRUE displacement from the propagated anchor that fires the launch path
LAUNCH_WIN_S = 1.0   # s: the displacement must be reached within this after leaving rest
GROW_MAX    = 1.3    # x: ball grown past this multiple of its address radius = picked up, not struck
GONE_S      = 0.27   # s (~8 fr @30): absence that turns a loss into a vanish candidate (absorbs flicker)
VANISH_NEAR_D = 3.0  # D: last-seen within this of the propagated anchor = vanished IN PLACE (swing/tap-in)
DEPART_DPF  = 0.30   # D/frame: min recent raw speed for a departed-then-lost roll confirm (occlusion of a
                     #   static ball fails this; a rolling putt is >= ~0.6)
VANISH_HOLD_S = 1.0  # s: vanish look-ahead; persistent reappearance at the anchor within it vetoes
RET_FRAC    = 0.25   # of the hold window: reappearance is "persistent" above this fraction of frames
                     #   (sparse in-cup sightings of a holed tap-in stay far below it)
RET_D       = 3.0    # D: back within this of the propagated anchor counts as a return
RAW_RET_D   = 3.5    # D: wider in raw mode — the unpropagated image-space anchor drifts under head motion
RET_LOOK_S  = 2.0    # s: launch-path veto window for a continuous-track return (practice putt)
EDGE_D      = 1.5    # D: last-seen closer than this to a border = left the view
EXIT_FRAC   = 0.22   # x min(W,H): ray-to-border along the post-loss camera drift below this = view exit
DRIFT_MIN_D = 1.0    # D: post-loss drift below this = camera steady, exit tests don't apply
# NOTE a rescue-watch for exit-suppressed vanishes (fire anyway if a ball reappears DEPARTED from
# the anchor) was tried and REVERTED: in multi-ball scenes (practice green, range) other balls in
# view during the pan fake the departure evidence -> 4 new false fires across dev+held-out. The
# one event it would rescue (putt whose detection drops at contact while the head pans, with a
# second ball lying at the spot) is structurally ambiguous with a practice putt; accept the miss.
COOLDOWN_S  = 2.5    # s: min separation between fired hits

# cams=None degradation (validated as the box-only design): raw stillness must be strict —
# a walking head never holds a ball image below 0.42 D/fr for 0.5 s — and a bare 5 D raw
# displacement is NOT safe to fire on (walking produces it), so launch needs a far confirm.
RAW_REST_THR   = 0.42   # D/frame raw rest
RAW_REST_RUN_S = 0.50   # s raw rest run
RAW_FAR_D      = 14.0   # D: raw displacement that confirms an in-frame roll
RAW_ROLL_DPF   = 0.73   # D/frame: min mean speed since departure (putt ~1.05; walk drift 0.4-0.7)
RAW_MAXSTEP_D  = 15.0   # D: bigger per-frame jump at confirm = tracker teleport, reject


def _apply(M, x, y):
    """Apply a [a,b,tx,c,d,ty] camera affine; identity when None."""
    if M is None:
        return x, y
    return M[0] * x + M[1] * y + M[2], M[3] * x + M[4] * y + M[5]


def _cands(frame, rmax):
    """Size-gated, deduped ball candidates as (x, y, r)."""
    out = []
    for b in frame.get("b") or []:
        r = (b[2] - b[0] + b[3] - b[1]) / 4.0
        if r > rmax or r < RMIN_CAND:
            continue
        x, y = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        if any(math.hypot(x - px, y - py) < DEDUP_D * 2 * max(pr, r) for px, py, pr in out):
            continue
        out.append((x, y, max(r, RMIN)))
    return out


def track_balls(frames, fps=30.0, cams=None, size=(2048, 1536)):
    """Primary-ball track (gated association, ego-propagated LOST re-acquisition).
    Returns tr[i] = (x, y, r) or None — used by the annotator for the trajectory trail."""
    W, _ = size
    rmax = RMAX_FRAC * W
    n = len(frames)
    tr = [None] * n
    tk = tk_pred = None
    lost = 0
    lost_max = int(LOST_MAX_S * fps)
    for i in range(n):
        cands = _cands(frames[i], rmax)
        M = cams[i - 1] if (cams is not None and 0 <= i - 1 < len(cams)) else None
        if tk is not None and lost == 0:
            px, py = _apply(M, tk[0], tk[1])
        elif tk_pred is not None:
            px, py = _apply(M, tk_pred[0], tk_pred[1])
            tk_pred = (px, py, tk_pred[2])
        else:
            px = py = None
        assoc = None
        if px is not None and cands:
            D = 2 * (tk[2] if lost == 0 else tk_pred[2])
            gate = (GATE_STEP if lost == 0 else REACQ_D) * D
            best, bd = None, gate
            for c in cands:
                d = math.hypot(c[0] - px, c[1] - py)
                if d < bd:
                    best, bd = c, d
            assoc = best
        if assoc is not None:
            tk, tk_pred, lost = assoc, None, 0
        elif tk is not None or tk_pred is not None:
            if lost == 0:
                tk_pred = (px, py, tk[2]) if px is not None else tk
            lost += 1
            if lost > lost_max:
                tk = tk_pred = None
                lost = 0
        elif cands:
            tk = max(cands, key=lambda c: c[2])   # fresh seed: largest = the near/addressed ball
            tk_pred, lost = None, 0
        tr[i] = tk if (tk is not None and lost == 0) else None
    return tr


def _ray_to_border(x, y, vx, vy, W, H):
    """Distance from (x,y) along (vx,vy) to the frame border + which border ('l','r','t','b')."""
    m = math.hypot(vx, vy)
    if m < 1e-9:
        return float("inf"), None
    ux, uy = vx / m, vy / m
    best, edge = float("inf"), None
    if ux > 1e-9 and (W - x) / ux < best:
        best, edge = (W - x) / ux, "r"
    if ux < -1e-9 and -x / ux < best:
        best, edge = -x / ux, "l"
    if uy > 1e-9 and (H - y) / uy < best:
        best, edge = (H - y) / uy, "b"
    if uy < -1e-9 and -y / uy < best:
        best, edge = -y / uy, "t"
    return best, edge


def detect_hits(frames, fps, cams=None, size=(2048, 1536)):
    """Return (hit_frames, armed).
       frames[i] = {"b": [[x1,y1,x2,y2],..], "c": [[x1,y1,x2,y2,conf],..]} (full-res pixels)
       cams      = per-frame camera affines (len n-1, entries may be None) from golf/cam_affine.py,
                   or None -> raw degradation preset.
       hit_frames = sorted fire frame indices (back-dated to ~contact).
       armed[i]   = True while an address is armed (for PREPARE status rendering)."""
    n = len(frames)
    if n == 0:
        return [], []
    ego = cams is not None
    W, H = size
    # orientation robustness: if boxes exceed the stated bounds, swap W/H
    mx = my = 0.0
    for f in frames[: min(n, 400)]:
        for b in (f.get("b") or []) + [c[:4] for c in (f.get("c") or [])]:
            mx, my = max(mx, b[2]), max(my, b[3])
    if mx > W or my > H:
        W, H = (max(W, H), min(W, H)) if mx >= my else (min(W, H), max(W, H))
    rmax = RMAX_FRAC * W

    rest_thr   = REST_THR if ego else RAW_REST_THR
    rest_run_n = max(2, int(round((REST_RUN_S if ego else RAW_REST_RUN_S) * fps)))
    gap_max    = max(1, int(round(GAP_MAX_S * fps)))
    lost_max   = int(LOST_MAX_S * fps)
    addr_hold  = int(ADDR_HOLD_S * fps)
    club_mem   = int(CLUB_MEM_S * fps)
    launch_win = int(LAUNCH_WIN_S * fps)
    gone_n     = max(2, int(round(GONE_S * fps)))
    vanish_hold = int(VANISH_HOLD_S * fps)
    ret_look   = int(RET_LOOK_S * fps)
    cooldown   = int(COOLDOWN_S * fps)

    def cam(i):
        if cams is not None and 0 <= i < len(cams):
            return cams[i]
        return None

    def dbg(i, *a):
        if _DBG:
            print(f"[{i / fps:7.2f}s]", *a)

    # ---- state ----
    tk = tk_pred = None          # tracked ball / ego-propagated position while LOST
    lost = 0
    lost_pos = None              # last-seen position when the current loss began
    trail = []                   # (frame, x, y) recent tracked positions (departure back-dating)
    rest_run = 0
    last_rest = -10 ** 9
    anchor = None                # (x, y, r), propagated every frame
    last_club = -10 ** 9
    launch_pend = None           # {"fire", "anchor", "gapmax"}
    vanish_pend = None           # {"fire", "anchor", "seen", "frames"}
    last_fire = -10 ** 9
    hits = []
    armed_arr = [False] * n

    ret_d = RET_D if ego else RAW_RET_D

    def reset_address():
        nonlocal anchor, last_rest, rest_run
        anchor, last_rest, rest_run = None, -10 ** 9, 0

    def mark_track_break():
        """The primary track died or re-seeded onto a different ball: any later ball near the
        anchor is NOT a continuous return (fixes the practice-putt-veto killing re-tracked hits)."""
        if launch_pend is not None:
            launch_pend["gapmax"] = 10 ** 9

    for i in range(n):
        cands = _cands(frames[i], rmax)
        M = cam(i - 1)   # camera motion frame i-1 -> i

        # -- propagate ego-anchored state into frame i --
        if anchor is not None:
            ax, ay = _apply(M, anchor[0], anchor[1])
            anchor = (ax, ay, anchor[2])
        for p in (launch_pend, vanish_pend):
            if p is not None:
                ax, ay = _apply(M, p["anchor"][0], p["anchor"][1])
                p["anchor"] = (ax, ay, p["anchor"][2])
        # trail positions are propagated too, so departure_frame compares in current coords
        if M is not None:
            trail = [(k, *_apply(M, x, y)) for k, x, y in trail]
        if lost == 0 and tk is not None:
            px, py = _apply(M, tk[0], tk[1])
        elif tk_pred is not None:
            px, py = _apply(M, tk_pred[0], tk_pred[1])
            tk_pred = (px, py, tk_pred[2])
        else:
            px = py = None

        # -- associate --
        assoc = None
        if px is not None and cands:
            D = 2 * (tk[2] if lost == 0 else tk_pred[2])
            gate = (GATE_STEP if lost == 0 else REACQ_D) * D
            best, bd = None, gate
            for c in cands:
                d = math.hypot(c[0] - px, c[1] - py)
                if d < bd:
                    best, bd = c, d
            assoc = best

        true_step = None
        raw_step = None
        if assoc is not None:
            if lost > 0 and launch_pend is not None:
                launch_pend["gapmax"] = max(launch_pend["gapmax"], lost)
            if px is not None and lost == 0 and tk is not None:
                true_step = math.hypot(assoc[0] - px, assoc[1] - py) / (2 * assoc[2])
                raw_step = math.hypot(assoc[0] - tk[0], assoc[1] - tk[1]) / (2 * assoc[2])
            if lost > gap_max:
                rest_run = 0          # back after more than a flicker: rest history is void
            tk, tk_pred, lost = assoc, None, 0
            trail.append((i, tk[0], tk[1]))
            if len(trail) > 4 * ret_look:
                trail = trail[-4 * ret_look:]
        elif tk is not None or tk_pred is not None:
            if lost == 0:
                tk_pred = (px, py, tk[2]) if px is not None else tk
                lost_pos = (tk[0], tk[1])
            lost += 1
            if lost > lost_max:
                tk = tk_pred = None
                lost = 0
                rest_run = 0
                mark_track_break()
        elif cands:
            tk = max(cands, key=lambda c: c[2])
            tk_pred, lost, rest_run = None, 0, 0
            trail.append((i, tk[0], tk[1]))
            mark_track_break()

        # -- rest / address arming (TRUE speed when cams given, raw otherwise) --
        step_for_rest = true_step if ego else raw_step
        if step_for_rest is not None:
            if step_for_rest < rest_thr:
                rest_run += 1
                if rest_run >= rest_run_n:
                    last_rest = i
                    anchor = (tk[0], tk[1], tk[2])
            else:
                rest_run = 0

        # -- club-at-ball memory --
        if tk is not None and lost == 0 and frames[i].get("c"):
            D = 2 * tk[2]
            for cb in frames[i]["c"]:
                if math.hypot((cb[0] + cb[2]) / 2 - tk[0], (cb[1] + cb[3]) / 2 - tk[1]) < CLUB_NEAR_D * D:
                    last_club = i
                    break

        armed = anchor is not None and (i - last_rest) <= addr_hold
        armed_arr[i] = armed
        club_ok = (i - last_club) <= club_mem

        # ---- trigger 1: LAUNCH (tracked displacement from the propagated anchor) ----
        if (launch_pend is None and vanish_pend is None and armed and club_ok
                and tk is not None and lost == 0 and (i - last_rest) <= launch_win
                and (i - last_fire) >= cooldown):
            D = 2 * anchor[2]
            disp = math.hypot(tk[0] - anchor[0], tk[1] - anchor[1]) / max(D, 1e-6)
            if disp >= LAUNCH_D and tk[2] <= GROW_MAX * anchor[2]:
                # fire time = the moment stillness ended (~contact); armed guarantees it is recent
                if ego:
                    launch_pend = {"fire": last_rest, "anchor": anchor, "gapmax": 0}
                    dbg(i, f"LAUNCH pend disp={disp:.1f}D fire@{last_rest / fps:.2f}s")
                else:
                    # raw mode: a bare 5 D raw displacement can be pure camera motion -> demand a
                    # far confirm at real roll speed before pending the veto window
                    mean_spd = disp / max(1, i - last_rest)
                    lstep = raw_step if raw_step is not None else 0.0
                    if disp >= RAW_FAR_D and mean_spd >= RAW_ROLL_DPF and lstep <= RAW_MAXSTEP_D \
                            and tk[2] <= GROW_MAX * anchor[2]:
                        launch_pend = {"fire": last_rest, "anchor": anchor, "gapmax": 0}
                        dbg(i, f"LAUNCH(raw far) pend disp={disp:.1f}D")

        # ---- trigger 2: VANISH (track lost from an armed address) ----
        if (vanish_pend is None and launch_pend is None and armed and club_ok
                and lost == gone_n and tk_pred is not None and lost_pos is not None
                and (i - last_fire) >= cooldown):
            D = 2 * anchor[2]
            last_disp = math.hypot(lost_pos[0] - anchor[0], lost_pos[1] - anchor[1])  # coords drift apart
            # recompute against the anchor as it was at loss start: approximate with current
            # propagated anchor vs propagated last position (both carried to frame i)
            last_disp = math.hypot(tk_pred[0] - anchor[0], tk_pred[1] - anchor[1]) / max(D, 1e-6)
            # recent raw speed just before the loss (from the trail)
            spd = 0.0
            pts = [(k, x, y) for k, x, y in trail if k >= i - lost - 3]
            if len(pts) >= 2:
                (k0, x0, y0), (k1, x1, y1) = pts[0], pts[-1]
                if k1 > k0:
                    spd = math.hypot(x1 - x0, y1 - y0) / (k1 - k0) / max(D, 1e-6)
            if last_disp <= VANISH_NEAR_D:
                # vanished IN PLACE -> full swing / tap-in, unless the ball left the VIEW
                dvx, dvy = tk_pred[0] - lost_pos[0], tk_pred[1] - lost_pos[1]
                drift = math.hypot(dvx, dvy)
                exit_like = False
                if drift >= DRIFT_MIN_D * D:
                    ray, edge = _ray_to_border(lost_pos[0], lost_pos[1], dvx, dvy, W, H)
                    near_border = min(lost_pos[0], W - lost_pos[0], lost_pos[1], H - lost_pos[1]) < EDGE_D * D
                    exit_like = ray < EXIT_FRAC * min(W, H) or near_border
                    if exit_like and edge == "b" and dvy > abs(dvx):
                        exit_like = False   # bottom exit w/ downward drift = head lifting at the strike
                if not exit_like:
                    ff = i - gone_n + 1
                    vanish_pend = {"fire": ff, "anchor": anchor, "seen": 0, "frames": 0}
                    dbg(i, f"VANISH pend @{ff / fps:.2f}s last_disp={last_disp:.1f}D")
                else:
                    dbg(i, f"vanish suppressed: exit-like drift=({dvx:.0f},{dvy:.0f})")
                    reset_address()
            elif spd >= DEPART_DPF:
                # DEPARTED (>= VANISH_NEAR_D) and moving when lost -> mid-roll loss confirms the
                # roll; no exit-ray test (a panning head follows a real putt exactly like an exit)
                vanish_pend = {"fire": last_rest, "anchor": anchor, "seen": 0, "frames": 0}
                dbg(i, f"ROLL-LOSS pend @{last_rest / fps:.2f}s disp={last_disp:.1f}D spd={spd:.2f}D/fr")

        # ---- pending resolution ----
        if launch_pend is not None:
            D = 2 * launch_pend["anchor"][2]
            vetoed = False
            if tk is not None and lost == 0 and launch_pend["gapmax"] <= gap_max:
                # the SAME continuously-tracked ball back at the spot = practice putt/reposition
                if math.hypot(tk[0] - launch_pend["anchor"][0], tk[1] - launch_pend["anchor"][1]) < ret_d * D:
                    vetoed = True
            if vetoed:
                dbg(i, "launch VETO (continuous return)")
                launch_pend = None
                reset_address()
            elif i - launch_pend["fire"] >= ret_look:
                if (launch_pend["fire"] - last_fire) >= cooldown:
                    hits.append(launch_pend["fire"])
                    last_fire = launch_pend["fire"]
                    dbg(i, f"FIRE launch @{launch_pend['fire'] / fps:.2f}s")
                launch_pend = None
                reset_address()

        if vanish_pend is not None:
            D = 2 * vanish_pend["anchor"][2]
            vanish_pend["frames"] += 1
            if any(math.hypot(c[0] - vanish_pend["anchor"][0], c[1] - vanish_pend["anchor"][1]) < ret_d * D
                   for c in cands):
                vanish_pend["seen"] += 1
            if vanish_pend["frames"] >= vanish_hold:
                # persistent reappearance at the spot = the ball is still there (waggle/drop-out);
                # sparse sightings (a holed ball glimpsed in the cup) stay under RET_FRAC
                if vanish_pend["seen"] < RET_FRAC * vanish_pend["frames"]:
                    if (vanish_pend["fire"] - last_fire) >= cooldown:
                        hits.append(vanish_pend["fire"])
                        last_fire = vanish_pend["fire"]
                        dbg(i, f"FIRE vanish @{vanish_pend['fire'] / fps:.2f}s")
                else:
                    dbg(i, f"vanish VETO (reappeared {vanish_pend['seen']}/{vanish_pend['frames']} frames)")
                vanish_pend = None
                reset_address()

    return sorted(hits), armed_arr
