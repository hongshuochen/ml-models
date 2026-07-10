"""REFERENCE: box-only golf hit detector — {ball, club_head} BOXES ONLY, NO camera affines.

Provenance: 2nd winner of the 2026-07-10 adversarial hit-algorithm eval (12/12 on the 4-clip dev
oracle, 6/8 on 4 blind held-out clips, 0 false fires). The DEPLOYED offline detector is
golf/hit_detector.py (ego-compensated; better short-putt recall). Keep THIS file as the blueprint
for a no-optical-flow port (e.g. PuttCounter.kt before the app grows a global-motion estimator);
its known cost is missing short putts that stop <14 D away (raw 5 D displacement can be pure
camera motion, so it demands a far/vanish confirm). detect(frames, fps, cams=None, W, H) is the
API; __main__ scores the dev oracle given a directory of <clip>_fr_det.json (golf/cache_dets.py).

Device-cheapest floor: the Android app may never get optical flow, so every signal here is
ego-invariant or ego-robust by construction:

  * address gate     : ball ~still (raw) with a club_head seen NEAR it (relative distance —
                       ego-invariant) — the one reliable pre-hit signal (club is often invisible
                       AT impact due to motion blur, so club-sweep is never required).
  * launch triggers  : (a) displacement from the address anchor, (b) ball-club SEPARATION RATE
                       (relative distance rate — ego-invariant putt trigger).
  * coherence veto   : if ball and club image velocities agree over a short window, that shared
                       motion is CAMERA motion, not ball motion -> suppress launch triggers.
  * vanish guards    : a real struck ball is last seen AT the anchor and gone in 1-2 frames (or
                       leaves at high speed); a world-static ball under a walking head drifts
                       SLOWLY + SMOOTHLY to a frame EDGE before "vanishing" -> reject slow,
                       far-from-anchor, edge-exiting vanishes.
  * far-confirm      : an in-frame roll must average real roll speed (a putt rolls ~1 D/fr TRUE;
                       walk-drift of a static ball is only 0.4-0.7 D/fr) — a soft ego guard.
  * return veto      : a ball that comes back to the anchor via a CONTINUOUS track (practice putt)
                       vetoes; a DISCRETE reappearance after >= ~0.7 s of full absence near the
                       anchor (re-teeing a new ball) does not.

Offline reference: uses bounded look-ahead (vanish confirm 0.6 s, return veto 3.0 s). An online
port fires with that delay. One parameter set for all clips; thresholds in ball-diameters (D),
D/second, or seconds, converted per-clip via fps.
"""
import json
import math
import os

DEBUG = bool(os.environ.get("HIT3_DEBUG"))

# ---------------- tunables (physical units; converted to frames/pixels via fps and D) ----------
RMIN = 6.0            # px: floor on ball radius (avoid divide-by-tiny)
RMAX = 58.0           # px: a real address ball is never bigger @2048w -> drop phone/hand/basket blobs
GATE_DPS = 120.0      # D/s (=4 D/fr @30): max association jump per frame of gap; beyond = teleport, drop
GATE_MAX_D = 8.0      # D: hard cap on the association gate regardless of gap (stops stray-chaining)
RESEED_S = 1.0        # s: track lost this long -> next ball is a NEW ball (re-seed; enables re-tee)
REST_DPS = 12.6       # D/s (=0.42 D/fr @30): raw per-frame move still counted as "at rest" (address)
REST_RUN_S = 0.5      # s: consecutive still time to arm an address. THE key walk-away killer:
                      #   a walking head never holds the ball image <0.42 D/fr for 0.5 s (measured
                      #   max run ~0.37 s on golf_039), while a golfer at address is still for seconds.
ADDR_HOLD_S = 1.5     # s: an armed address stays live this long after the last still frame
CLUB_NEAR_D = 6.0     # D: club_head counts as "at the ball" below this relative distance
CLUB_HITS = 2         # frames: min frames with club at the ball inside CLUB_WIN_S (club det flickers:
                      #   the 051 driver address shows only 2 club-on-ball frames in 2.5 s)
CLUB_WIN_S = 2.5      # s: look-back window for club-at-ball before the fire (address gate; covers a
                      #   full backswing+downswing, during which the club is away from the ball)
BACKSWING_LO = 3.0    # D: club must start this close to the ball ...
BACKSWING_HI = 6.0    # D: ... and visibly move this far away within BACKSWING_S to count as a
BACKSWING_S = 1.5     # s: backswing signature (gates the vanish-without-displacement trigger)
COH_WIN_S = 0.27      # s (~8 fr @30): window for the ball/club velocity-coherence test
COH_TOL_DPS = 9.0     # D/s (=0.3 D/fr): |v_ball - v_club| below this => shared motion = camera motion
COH_MOVE_DPS = 6.0    # D/s (=0.2 D/fr): coherence only counts if things are actually moving
COH_MIN_FRAC = 0.6    # fraction of window frames needing both ball+club velocity samples
LAUNCH_D = 5.0        # D: displacement from the rest anchor that opens a PENDING hit
SEP_NEAR_D = 3.0      # D: ball-club relative distance counted as "in contact position" (putt trigger)
SEP_RATE_DPS = 15.0   # D/s (=0.5 D/fr): separation rate leaving SEP_NEAR that opens a PENDING hit
SEP_RUN_S = 0.08      # s: separation rate must be sustained this long (~2-3 fr)
CONFIRM_S = 1.0       # s: PENDING must confirm (vanish or far) within this window
FAR_D = 14.0          # D: cumulative move from anchor confirming an in-frame launch/roll
FAR_MAXSTEP_D = 15.0  # D: per-frame jump beyond this at far-confirm = tracker teleport (reject)
RECEDE = 1.2          # x: a struck ball never GROWS past this multiple of its address radius
ROLL_MIN_DPS = 22.0   # D/s (~0.73 D/fr): min MEAN speed since departure for far-confirm
                      #   (real putt ~1.05 D/fr TRUE; walking drift of a static ball 0.4-0.7 D/fr raw)
VAN_WIN_S = 0.4       # s: vanish look-ahead; absorbs 1-3 frame detection flicker + brief strays
                      #   (kept short so an unrelated ball rolling by ~0.5 s later can't resurrect it)
VAN_SEEN_S = 0.12     # s (~3 fr @30): max time a ball may be seen NEAR the lost track inside VAN_WIN
CONT_ZONE_D = 6.0     # D: a candidate within this of the last-seen position = the track continuing
                      #   (impact strays reappear 7-10 D away for <=2 frames -> outside this zone)
VANISH_NEAR_D = 3.5   # D: last-seen within this of the anchor = "vanished in place" (real swing)
FAST_DPS = 45.0       # D/s (=1.5 D/fr): OR last-seen recent speed at least this (ball caught launching)
DEPART_DPS = 9.0      # D/s (=0.3 D/fr): OR ball measurably moving when it vanished (rolling putt lost
                      #   mid-roll). A gappy/static track that just drops out (occlusion) fails this.
EDGE_OVR_S = 1.2      # s: edge-exit veto is waived if the ball was still (at address) this recently —
                      #   a swing's head-sweep can carry the ball image out an edge AT impact, but a
                      #   walk-off drift needs longer than this after the last true stillness
EDGE_FRAC = 0.06      # of min(W,H): last-seen this close to a border can be a walk-off exit
EDGE_CONSIST = 0.80   # mean-unit-velocity norm over the last seen frames: drift this SMOOTH,
                      #   pointed at that border, at sub-launch speed => walk-off, reject vanish
EDGE_SLOW_DPS = 36.0  # D/s (=1.2 D/fr): edge-exit veto only applies below this speed
RET_D = 3.5           # D: the tracked ball back within this of the anchor = a return (practice putt).
                      #   Must stay >= ~3 D: the image-space anchor drifts under head motion (no
                      #   affines to propagate it), so the return lands off the exact address pixel.
RET_LOOK_S = 3.0      # s: look-ahead window for the return veto
ABSENT_MIN_S = 0.7    # s: primary-track gap this long => any later ball is a NEW ball (re-tee),
                      #   not a return (a practice putt is tracked back with gaps of only 1-3 frames)
COOLDOWN_S = 2.5      # s: min separation between fired hits


def _ball_cands(frame):
    out = []
    for b in frame.get("b") or []:
        r = (b[2] - b[0] + b[3] - b[1]) / 4.0
        if r <= RMAX:
            out.append(((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, max(r, RMIN)))
    return out


def _track_balls(frames, fps):
    """Primary-ball track with a teleport gate and >=RESEED_S re-seeding. tr[i]=(x,y,r) or None."""
    n = len(frames)
    tr = [None] * n
    last = None
    last_i = -10 ** 9
    gate_dpf = GATE_DPS / fps
    reseed_f = int(RESEED_S * fps)
    for i in range(n):
        cands = _ball_cands(frames[i])
        if not cands:
            continue
        gap = i - last_i
        if last is None or gap > reseed_f:
            c = max(cands, key=lambda p: p[2])      # fresh seed: largest = the near/addressed ball
        else:
            c = min(cands, key=lambda p: (p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2)
            d = math.hypot(c[0] - last[0], c[1] - last[1])
            gate = min(gate_dpf * gap, GATE_MAX_D) * (2 * last[2])
            if d > gate:                            # teleport to a far ball -> keep track lost
                continue
        tr[i] = c
        last = c
        last_i = i
    return tr


def detect(frames, fps, cams=None, W=2048, H=1536):
    """frames[i] = {"b": [[x1,y1,x2,y2],..], "c": [[x1,y1,x2,y2,conf],..]}. cams IGNORED (box-only).
    Returns sorted list of fire FRAME indices."""
    n = len(frames)
    if n == 0:
        return []
    # defensive: derive effective frame bounds from the data (portrait/landscape mixups happen)
    for f in frames:
        for b in (f.get("b") or []) + [cc[:4] for cc in (f.get("c") or [])]:
            W = max(W, b[2])
            H = max(H, b[3])
    tr = _track_balls(frames, fps)

    # nearest club_head to the tracked ball, per frame
    club = [None] * n           # (x, y) of nearest club to ball
    club_dist = [float("inf")] * n
    for i in range(n):
        c = tr[i]
        if c is None or not frames[i].get("c"):
            continue
        best, bd = None, float("inf")
        for cb in frames[i]["c"]:
            cx, cy = (cb[0] + cb[2]) / 2.0, (cb[1] + cb[3]) / 2.0
            d = math.hypot(cx - c[0], cy - c[1])
            if d < bd:
                bd, best = d, (cx, cy)
        club[i], club_dist[i] = best, bd

    # per-frame velocities (only across single-frame gaps)
    vb = [None] * n
    vc = [None] * n
    for i in range(1, n):
        if tr[i] and tr[i - 1]:
            vb[i] = (tr[i][0] - tr[i - 1][0], tr[i][1] - tr[i - 1][1])
        if club[i] and club[i - 1]:
            vc[i] = (club[i][0] - club[i - 1][0], club[i][1] - club[i - 1][1])

    # frame-unit conversions
    rest_thr_dpf = REST_DPS / fps
    rest_run = max(2, int(round(REST_RUN_S * fps)))
    addr_hold = int(ADDR_HOLD_S * fps)
    club_win = int(CLUB_WIN_S * fps)
    coh_win = max(3, int(round(COH_WIN_S * fps)))
    coh_tol_dpf = COH_TOL_DPS / fps
    coh_move_dpf = COH_MOVE_DPS / fps
    sep_rate_dpf = SEP_RATE_DPS / fps
    sep_run = max(2, int(round(SEP_RUN_S * fps)))
    confirm_w = int(CONFIRM_S * fps)
    roll_min_dpf = ROLL_MIN_DPS / fps
    van_win = int(VAN_WIN_S * fps)
    van_seen_max = int(round(VAN_SEEN_S * fps))
    fast_dpf = FAST_DPS / fps
    depart_dpf = DEPART_DPS / fps
    edge_px = EDGE_FRAC * min(W, H)
    edge_slow_dpf = EDGE_SLOW_DPS / fps
    ret_look = int(RET_LOOK_S * fps)
    absent_min = int(ABSENT_MIN_S * fps)
    cooldown = int(COOLDOWN_S * fps)

    def coherent(i, D):
        """Ball & club image velocities agree over the recent window -> shared motion = CAMERA."""
        diffs, moves, have = [], [], 0
        for j in range(max(1, i - coh_win + 1), i + 1):
            if vb[j] is not None and vc[j] is not None:
                have += 1
                diffs.append(math.hypot(vb[j][0] - vc[j][0], vb[j][1] - vc[j][1]))
                moves.append(math.hypot(vb[j][0], vb[j][1]))
        if have < max(2, int(COH_MIN_FRAC * coh_win)):
            return False
        return (sum(diffs) / len(diffs) <= coh_tol_dpf * D
                and sum(moves) / len(moves) >= coh_move_dpf * D)

    def club_at_ball(fire_frame, D):
        """>= CLUB_HITS frames with the club at the ball in the CLUB_WIN_S look-back.
        Count-based, not run-based: club detection flickers at address."""
        lo = max(0, fire_frame - club_win)
        thr = CLUB_NEAR_D * D
        return sum(1 for k in range(lo, fire_frame + 1) if club_dist[k] < thr) >= CLUB_HITS

    def backswing_seen(i, D):
        """Club VISIBLY left the ball (<=BACKSWING_LO -> >=BACKSWING_HI) within BACKSWING_S."""
        lo = max(0, i - int(BACKSWING_S * fps))
        seen_close = False
        for k in range(lo, i + 1):
            if club_dist[k] == float("inf"):
                continue
            d = club_dist[k] / D
            if d <= BACKSWING_LO:
                seen_close = True
            elif seen_close and d >= BACKSWING_HI:
                return True
        return False

    def recent_speed(i, D):
        """Mean per-frame ball speed (D/fr) over the last few velocity samples before i inclusive."""
        v = [math.hypot(*vb[j]) for j in range(max(1, i - 2), i + 1) if vb[j] is not None]
        return (sum(v) / len(v) / D) if v else 0.0

    def edge_exit(i, D):
        """Last-seen near a border, drifting smoothly toward it at sub-launch speed = walk-off."""
        c = tr[i]
        if c is None:
            return False
        near_l, near_r = c[0] < edge_px, c[0] > W - edge_px
        near_t, near_b = c[1] < edge_px, c[1] > H - edge_px
        if not (near_l or near_r or near_t or near_b):
            return False
        ux = uy = 0.0
        mags = []
        cnt = 0
        for j in range(max(1, i - coh_win + 1), i + 1):
            if vb[j] is None:
                continue
            m = math.hypot(*vb[j])
            if m < 1e-6:
                continue
            ux += vb[j][0] / m
            uy += vb[j][1] / m
            mags.append(m)
            cnt += 1
        if cnt < 3:
            return False
        consist = math.hypot(ux, uy) / cnt
        speed = sum(mags) / len(mags) / D
        toward = ((near_l and ux < 0) or (near_r and ux > 0)
                  or (near_t and uy < 0) or (near_b and uy > 0))
        return consist >= EDGE_CONSIST and toward and speed <= edge_slow_dpf

    def vanish_confirmed(i, lx, ly, D):
        """The TRACK is gone: over VAN_WIN_S after i, a ball is seen near the last-seen position
        (within CONT_ZONE_D) in at most VAN_SEEN_S of frames. Balls elsewhere in the frame
        (other range balls, impact strays 7-10 D off) do not resurrect the track."""
        hi = min(n, i + van_win)
        seen = 0
        for j in range(i, hi):
            for (x, y, _r) in _ball_cands(frames[j]):
                if math.hypot(x - lx, y - ly) <= CONT_ZONE_D * D:
                    seen += 1
                    break
        return seen <= van_seen_max

    def returns_continuously(ax, ay, D, cf):
        """Practice-putt veto: the PRIMARY TRACKED ball comes back within RET_D of the anchor with
        no track gap >= ABSENT_MIN_S on the way. A ball that shows up near the anchor after a full
        absence (re-teeing a new ball) or an unrelated ball elsewhere does NOT veto — in multi-ball
        scenes an any-candidate-near-anchor test false-vetoes real hits (stale image-space anchor)."""
        gap = 0
        for k in range(cf + 1, min(n, cf + 1 + ret_look)):
            if tr[k] is None:
                gap += 1
                if gap >= absent_min:
                    return False    # track fully lost -> anything later is a NEW ball
                continue
            gap = 0
            if math.hypot(tr[k][0] - ax, tr[k][1] - ay) < RET_D * D:
                return True
        return False

    # ---------------- state machine ----------------
    still_run = 0
    last_still = -10 ** 9
    anchor = None            # (x, y, r) address ball
    sep_run_cnt = 0
    prev_cd = float("inf")   # previous frame ball-club distance
    pend = None              # {"anchor":..., "open":..., "expire":..., "depart":...}
    last_fire = -10 ** 9
    hits = []

    for i in range(n):
        c = tr[i]

        # -- address arming (raw stillness; coherent camera motion does NOT refresh the anchor,
        #    it just isn't treated as ball motion)
        if c is not None and i > 0 and tr[i - 1] is not None:
            D = 2 * c[2]
            mv = math.hypot(c[0] - tr[i - 1][0], c[1] - tr[i - 1][1])
            if mv < rest_thr_dpf * D:
                still_run += 1
                if still_run >= rest_run:
                    last_still = i
                    anchor = (c[0], c[1], c[2])
            else:
                still_run = 0
        elif c is not None:
            still_run = 1

        armed = anchor is not None and (i - last_still) <= addr_hold

        # -- separation-rate bookkeeping (ego-invariant putt trigger)
        cd = club_dist[i] / (2 * c[2]) if (c is not None and club_dist[i] < float("inf")) else float("inf")
        sep_trig = False
        if cd < float("inf") and prev_cd < float("inf"):
            rate = cd - prev_cd                       # D/frame separation rate
            ball_moving = (anchor is not None and c is not None
                           and math.hypot(c[0] - anchor[0], c[1] - anchor[1]) >= 1.0 * 2 * anchor[2])
            if prev_cd <= SEP_NEAR_D + 1.5 and rate >= sep_rate_dpf and ball_moving:
                sep_run_cnt += 1
                if sep_run_cnt >= sep_run:
                    sep_trig = True
            else:
                sep_run_cnt = 0
        else:
            sep_run_cnt = 0
        prev_cd = cd

        # -- open PENDING
        if pend is None and armed:
            D = 2 * anchor[2]
            trig = False
            if c is not None:
                disp = math.hypot(c[0] - anchor[0], c[1] - anchor[1]) / max(D, 1e-6)
                if (disp >= LAUNCH_D or sep_trig) and not coherent(i, D):
                    trig = True
            elif i > 0 and tr[i - 1] is not None and backswing_seen(i, D):
                trig = True   # armed ball vanished outright right after a visible backswing
                              # (still-head full swing; the backswing gate rejects plain dropouts)
            if trig:
                if DEBUG:
                    print(f"  [dbg] t={i/fps:.2f} PEND open sep={sep_trig} anchor=({anchor[0]:.0f},{anchor[1]:.0f})")
                # departure start = last frame the ball was within 1.5 D of the anchor
                dep = i
                for j in range(i, max(0, i - confirm_w) - 1, -1):
                    if tr[j] is not None and \
                       math.hypot(tr[j][0] - anchor[0], tr[j][1] - anchor[1]) <= 1.5 * D:
                        dep = j
                        break
                pend = {"anchor": anchor, "open": i, "expire": i + confirm_w, "depart": dep}

        # -- confirm / expire PENDING
        if pend is not None:
            ax, ay, ar = pend["anchor"]
            D = 2 * ar
            confirmed = False

            fire = None
            if c is None and i > 0 and tr[i - 1] is not None:
                # ball just vanished: real swing vanishes IN PLACE (or at launch speed);
                # a walk-off drifts slowly, far from anchor, out an edge.
                ls = i - 1
                last_disp = math.hypot(tr[ls][0] - ax, tr[ls][1] - ay) / max(D, 1e-6)
                spd = recent_speed(ls, D)
                vconf = vanish_confirmed(i, tr[ls][0], tr[ls][1], D)
                edge = edge_exit(ls, D) and (i - last_still) > int(EDGE_OVR_S * fps)
                if DEBUG:
                    print(f"  [dbg] t={i/fps:.2f} VANISH? conf={vconf} last_disp={last_disp:.1f}D "
                          f"spd={spd:.2f}D/fr edge={edge} last=({tr[ls][0]:.0f},{tr[ls][1]:.0f})")
                if vconf:
                    in_place = last_disp <= VANISH_NEAR_D          # gone AT the address = full swing
                    fast = spd >= fast_dpf                          # caught leaving at launch speed
                    departing = spd >= depart_dpf                   # rolling when the track died
                    if (in_place or fast or departing) and not edge:
                        confirmed = True
                        fire = i          # vanish frame ~= impact for a full swing

            if not confirmed and c is not None and i > 0 and tr[i - 1] is not None:
                # far-confirm: sustained in-frame roll at real roll speed, receding, no teleport
                disp = math.hypot(c[0] - ax, c[1] - ay) / max(D, 1e-6)
                step = math.hypot(c[0] - tr[i - 1][0], c[1] - tr[i - 1][1]) / max(D, 1e-6)
                dt = max(1, i - pend["depart"])
                mean_spd = disp / dt                   # D/frame since departure
                if (disp >= FAR_D and c[2] <= RECEDE * ar and step <= FAR_MAXSTEP_D
                        and mean_spd >= roll_min_dpf and not coherent(i, D)):
                    confirmed = True
                    fire = i              # far-confirm frame (a putt reaches 14 D ~0.4 s after contact)

            if confirmed:
                ff = fire
                if DEBUG:
                    print(f"  [dbg] t={i/fps:.2f} CONFIRM ff={ff/fps:.2f} club={club_at_ball(ff, D)} "
                          f"ret={returns_continuously(ax, ay, D, i)} cd_ok={(ff - last_fire) >= cooldown}")
                if (club_at_ball(ff, D)
                        and not returns_continuously(ax, ay, D, i)
                        and (ff - last_fire) >= cooldown):
                    hits.append(ff)
                    last_fire = ff
                pend = None
                anchor = None
                still_run = 0
                last_still = -10 ** 9
            elif i >= pend["expire"]:
                pend = None

    return sorted(hits)


# ---------------- oracle self-scoring harness ----------------
if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    ORACLE = {
        "golf_010": {"real": [32.8], "false": [], "certain_total": True},
        "golf_049": {"real": [27.4, 35.2], "false": [14.9, 18.7, 29.8], "certain_total": True},
        "golf_039": {"real": [30.0], "false": [6.0, 9.3, 39.2, 46.0, 53.0], "certain_total": True},
        "golf_051": {"real": [18.0, 33.0], "false": [3.7, 27.1], "certain_total": False},
    }
    TOL = 0.8
    score = 0
    for clip in ["golf_010", "golf_049", "golf_039", "golf_051"]:
        d = json.load(open(os.path.join(base, clip + "_fr_det.json")))
        fps = d["fps"]
        fires = detect(d["frames"], fps, cams=None, W=2048, H=1536)
        times = [f / fps for f in fires]
        o = ORACLE[clip]
        caught = set()
        marks = []
        for t in times:
            r = next((x for x in o["real"] if abs(t - x) <= TOL and x not in caught), None)
            if r is not None:
                caught.add(r)
                score += 2
                marks.append(f"{t:.2f} ok")
            elif any(abs(t - x) <= TOL for x in o["false"]):
                score -= 3
                marks.append(f"{t:.2f} FALSE")
            elif o["certain_total"]:
                score -= 2
                marks.append(f"{t:.2f} EXTRA")
            else:
                marks.append(f"{t:.2f} uncertain")
        missed = [x for x in o["real"] if x not in caught]
        print(f"{clip}: [{' | '.join(marks) if marks else 'no fires'}]"
              + (f"  MISSED real: {missed}" if missed else ""))
    print(f"self-score: {score} (perfect 12; baseline 3)")
