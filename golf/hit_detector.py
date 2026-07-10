"""Golf HIT detector — counts hits (putts AND full swings) from per-frame {ball, club_head} boxes.

Ego-invariant, single primary-ball tracker. The signature common to BOTH a putt and a full swing
(measured on real first-person AR-glasses footage; picked by an adversarial multi-algorithm eval that
scored 6/6 real anchors and 0/7 confirmed false events on golf_010/049/039/051):

    the ball sits ~stationary AT ADDRESS (present, low image-plane motion, a club_head seen nearby)
    THEN it LEAVES the address abruptly — it either travels far very fast (launch / roll) or
    DISAPPEARS and stays gone.

That "leaves and does not come back" is what catches a full swing (ball gone in one frame at 30 fps)
which a distance-only "ball separates while staying trackable" state machine structurally misses.

Guards (why the confirmed-false events don't fire): a club_head must have met the ball (rejects walking
with a club / reaching into a ball basket / phone-in-view); the departure must persist into a vanish/far
(rejects a lifted-then-carried ball, which stays trackable and never vanishes); a return-to-address veto
rejects practice waggles / repositions; oversized ball boxes (phone/hand/basket blob) are dropped.

All thresholds are in ball-diameter / second units so they generalize across resolution & fps. This is
the OFFLINE reference (it looks ahead to confirm vanish + veto returns); the on-device app defers the
count by the look-ahead so it stays causal (see PuttCounter.kt).
"""
import math

# ---- tunables (ball-diameters / seconds; converted to frames per-clip via fps) ----
REST_THR    = 0.42   # per-frame move (in ball diameters) still counted as "at rest" (address)
REST_RUN    = 3      # consecutive still frames to arm an address
ADDR_HOLD_S = 1.5    # s: an armed address stays "recent" this long after the last still frame
CLUB_NEAR_D = 6.0    # D: a club_head must pass this close to the ball (address or impact)
CLUB_WIN_S  = 1.5    # s: look-back window for the club-near-ball test (this stroke only)
RECEDE      = 1.2    # far-path: ball must not have grown past this × its address size (a struck ball recedes)
LAUNCH_DISP = 5.0    # D: cumulative move from the rest anchor that opens a PENDING hit
FAR_D       = 14.0   # D: cumulative move that confirms an in-frame launch/roll
FAR_MAXSTEP = 15.0   # D: a far-confirm frame whose per-frame jump exceeds this is a tracker teleport (walking)
GONE        = 8      # frames the ball must stay absent to confirm a vanish
CONFIRM_S   = 1.0    # s: window after PENDING within which it must confirm
COOLDOWN_S  = 2.5    # s: min separation between fired hits (also merges roll-aftermath)
RMIN        = 6.0    # px: floor on ball radius (avoid divide-by-tiny)
RMAX        = 58.0   # px: a real address ball is never bigger -> drop phone/hand/basket blobs (@~2048w frames)
RET_D       = 2.0    # D: a ball reappearing this close to the address = NOT a hit (reposition / waggle)
RET_LOOK_S  = 2.0    # s: look-ahead window for the return-to-address test


def track_balls(frames, rmax=RMAX):
    """Nearest-to-previous ball centroid + radius per frame; None when no usable ball box.
    Drops boxes bigger than rmax px radius (phone/hand/basket). frames[i]['b'] = list of [x1,y1,x2,y2]."""
    tr = []
    prev = None
    for f in frames:
        cents = [((b[0]+b[2])/2, (b[1]+b[3])/2, max((b[2]-b[0]+b[3]-b[1])/4, RMIN))
                 for b in f["b"] if (b[2]-b[0]+b[3]-b[1])/4 <= rmax]
        if not cents:
            tr.append(None)
            continue
        if prev is not None:
            c = min(cents, key=lambda p: (p[0]-prev[0])**2 + (p[1]-prev[1])**2)
        else:
            c = max(cents, key=lambda p: p[2])   # largest = usually the addressed/near ball
        tr.append(c)
        prev = (c[0], c[1])
    return tr


def detect_hits(frames, fps, rmax=RMAX):
    """Return (hit_frames, armed) for a clip.
       hit_frames = sorted list of frame indices where a hit is counted.
       armed[i]   = bool, True while an address is armed at frame i (for PREPARE status rendering)."""
    n = len(frames)
    tr = track_balls(frames, rmax)

    # tracked ball -> nearest club_head distance each frame (inf if either absent)
    club_dist = [float("inf")] * n
    for i in range(n):
        c = tr[i]
        if c is None or not frames[i]["c"]:
            continue
        club_dist[i] = min(math.hypot((cb[0]+cb[2])/2 - c[0], (cb[1]+cb[3])/2 - c[1]) for cb in frames[i]["c"])

    addr_hold = int(ADDR_HOLD_S * fps)
    club_win  = int(CLUB_WIN_S * fps)
    confirm_w = int(CONFIRM_S * fps)
    cooldown  = int(COOLDOWN_S * fps)
    look_ret  = int(RET_LOOK_S * fps)

    def club_at_ball(fire_frame, D):
        thr = CLUB_NEAR_D * D
        lo = max(0, fire_frame - club_win)
        return any(club_dist[k] < thr for k in range(lo, fire_frame + 1))

    def returns_to_address(ax, ay, D, cf):
        for k in range(cf + 1, min(n, cf + 1 + look_ret)):
            for b in frames[k]["b"]:
                if math.hypot((b[0]+b[2])/2 - ax, (b[1]+b[3])/2 - ay) < RET_D * D:
                    return True
        return False

    still_run = 0
    last_still = -10**9
    anchor = None                 # (x, y, radius) of the address ball
    pend = None                   # {frame, anchor, expire} once launch displacement crossed
    last_fire = -10**9
    hits = []
    armed_arr = [False] * n

    for i in range(n):
        c = tr[i]
        if c is not None and i > 0 and tr[i-1] is not None:
            D = 2 * c[2]
            mv = math.hypot(c[0]-tr[i-1][0], c[1]-tr[i-1][1])
            if mv < REST_THR * D:
                still_run += 1
                if still_run >= REST_RUN:
                    last_still = i
                    anchor = (c[0], c[1], c[2])
            else:
                still_run = 0
        elif c is not None:
            still_run = 1

        armed = anchor is not None and (i - last_still) <= addr_hold
        armed_arr[i] = armed

        if pend is None and armed and c is not None:
            D = 2 * anchor[2]
            if math.hypot(c[0]-anchor[0], c[1]-anchor[1]) / max(D, 1e-6) >= LAUNCH_DISP:
                pend = {"anchor": anchor, "expire": i + confirm_w}

        if pend is not None:
            D = 2 * pend["anchor"][2]
            confirmed = False
            fire_frame = i
            if c is None:                                  # (a) vanish
                j = i; gone = 0
                while j < n and tr[j] is None:
                    gone += 1; j += 1
                if gone >= GONE:
                    confirmed = True; fire_frame = i
            if not confirmed and c is not None:            # (b) far, receding, no teleport
                disp = math.hypot(c[0]-pend["anchor"][0], c[1]-pend["anchor"][1]) / max(D, 1e-6)
                step = math.hypot(c[0]-tr[i-1][0], c[1]-tr[i-1][1]) / max(D, 1e-6) \
                    if i > 0 and tr[i-1] is not None else 0.0
                if disp >= FAR_D and c[2] <= RECEDE * pend["anchor"][2] and step <= FAR_MAXSTEP:
                    confirmed = True; fire_frame = i

            if confirmed:
                ax, ay, ar = pend["anchor"]
                if club_at_ball(fire_frame, 2*ar) and not returns_to_address(ax, ay, 2*ar, fire_frame) \
                        and (fire_frame - last_fire) >= cooldown:
                    hits.append(fire_frame)
                    last_fire = fire_frame
                pend = None; anchor = None; still_run = 0; last_still = -10**9
            elif i >= pend["expire"]:
                pend = None

    return sorted(hits), armed_arr
