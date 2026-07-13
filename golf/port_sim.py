"""Port-fidelity simulator for the Android HitCounter (v3 ego-fusion, ONLINE).

Mirrors the planned Kotlin 1:1: TIME-based thresholds (app fps varies), 640-square coordinates,
TRANSLATION-ONLY camera compensation (phone block-matcher can't give rotation/scale), deferred
confirmation (fires ~1-2 s after contact). Input = dev/held-out caches subsampled to 15 fps with
coords scaled x(640/2048) and affines composed pairwise then reduced to their center translation.
"""
import json
import math
import sys

# ---- physical-unit tunables (D = ball diameter; seconds; D/second) ----
REST_SPEED   = 7.5    # D/s TRUE speed under which the ball is at rest (0.25 D/fr @30)
REST_RUN_S   = 0.17
ADDR_HOLD_S  = 1.5
CLUB_NEAR_D  = 3.5
CLUB_MEM_S   = 3.0
LAUNCH_D     = 5.0
LAUNCH_WIN_S = 1.0
GROW_MAX     = 1.3
GONE_S       = 0.27
VANISH_NEAR_D = 3.0
DEPART_SPEED = 9.0    # D/s (0.3 D/fr @30)
VANISH_HOLD_S = 1.0
RET_FRAC     = 0.25
RET_D        = 3.0
RET_LOOK_S   = 2.0
EDGE_D       = 1.5
EXIT_FRAC    = 0.22
DRIFT_MIN_D  = 1.0
COOLDOWN_S   = 2.5
GATE_SPEED   = 150.0  # D/s association gate while tracked (5 D/fr @30) ...
GATE_CAP_D   = 8.0    # ... hard cap in D (long dt must not open the gate to teleports)
REACQ_D      = 2.5
GAP_MAX_S    = 0.15   # flicker gap bridged (2 frames @15fps)
LOST_MAX_S   = 2.0
RMAX_FRAC    = 58 / 2048.0
RMIN_CAND_FRAC = 5 / 2048.0
DEDUP_D      = 1.0
DWELL_MIN_S  = 0.4   # in-place vanish needs this much club-at-ball dwell in the last 3 s ...
DWELL_WIN_S  = 3.0
TOUCH_D      = 0.8   # ... OR one club sighting this close to the ball (sparse driver address)


class OnlineHitCounter:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.rmax = RMAX_FRAC * W
        self.rmin = RMIN_CAND_FRAC * W
        self.count = 0
        self.fires = []          # fire times (s)
        # track
        self.tk = None           # (x, y, r) tracked ball
        self.tk_pred = None      # propagated position while lost
        self.last_seen_t = -1e9
        self.lost_pos = None
        self.prev_seen = None    # (x, y, t) previous association (for depart speed)
        self.speed_dps = 0.0
        self.vanish_armed_for_loss = False
        # address
        self.rest_time = 0.0
        self.last_rest_t = -1e9
        self.anchor = None       # (x, y, r), translation-propagated
        self.last_club_t = -1e9
        self.club_hist = []      # (t, dt, near_bool, min_dist_D) of tracked-ball frames
        # pendings
        self.launch = None       # {fire_t, ax, ay, ar, broke}
        self.vanish = None       # {fire_t, ax, ay, ar, seen_s, start_t}
        self.last_fire_t = -1e9
        self.state = "SEARCH"

    def _cands(self, balls):
        out = []
        for (x, y, r) in balls:
            if r > self.rmax or r < self.rmin:
                continue
            if any(math.hypot(x - px, y - py) < DEDUP_D * 2 * max(pr, r) for px, py, pr in out):
                continue
            out.append((x, y, r))
        return out

    def _reset_address(self):
        self.anchor = None
        self.last_rest_t = -1e9
        self.rest_time = 0.0

    def _break_track(self):
        if self.launch is not None:
            self.launch["broke"] = True

    def update(self, t, balls, clubs, cam_at):
        """t s; balls [(x,y,r)]; clubs [(x,y)]; cam_at(x,y) -> (dx,dy) local camera translation
        since the previous analyzed frame, sampled at the tracking reference point (the Kotlin
        port approximates this with a local-patch block match around the same point)."""
        if self.anchor is not None:
            rx, ry = self.anchor[0], self.anchor[1]
        elif self.tk_pred is not None:
            rx, ry = self.tk_pred[0], self.tk_pred[1]
        elif self.tk is not None:
            rx, ry = self.tk[0], self.tk[1]
        else:
            rx, ry = self.W / 2.0, self.H / 2.0
        dx, dy = cam_at(rx, ry)
        cands = self._cands(balls)

        # -- propagate everything by the camera translation --
        if self.anchor is not None:
            self.anchor = (self.anchor[0] + dx, self.anchor[1] + dy, self.anchor[2])
        for p in (self.launch, self.vanish):
            if p is not None:
                p["ax"] += dx; p["ay"] += dy
        if self.tk is not None and self.tk_pred is None:
            px, py = self.tk[0] + dx, self.tk[1] + dy
        elif self.tk_pred is not None:
            self.tk_pred = (self.tk_pred[0] + dx, self.tk_pred[1] + dy, self.tk_pred[2])
            px, py = self.tk_pred[0], self.tk_pred[1]
        else:
            px = py = None

        # -- associate --
        assoc = None
        if px is not None and cands:
            tracked = self.tk_pred is None
            D = 2 * (self.tk[2] if tracked else self.tk_pred[2])
            dt_seen = t - self.last_seen_t
            gate = (min(GATE_SPEED * dt_seen, GATE_CAP_D) if tracked else REACQ_D) * D
            best, bd = None, gate
            for c in cands:
                d = math.hypot(c[0] - px, c[1] - py)
                if d < bd:
                    best, bd = c, d
            assoc = best

        true_speed = None
        if assoc is not None:
            gap = t - self.last_seen_t
            if self.tk_pred is not None and self.launch is not None and gap > GAP_MAX_S:
                self.launch["broke"] = True
            if px is not None and self.tk_pred is None and gap > 0:
                true_speed = math.hypot(assoc[0] - px, assoc[1] - py) / (2 * assoc[2]) / gap
            if gap > GAP_MAX_S:
                self.rest_time = 0.0
            self.prev_seen = (self.tk[0], self.tk[1], self.last_seen_t) if self.tk else None
            if self.prev_seen and self.last_seen_t < t:
                self.speed_dps = (math.hypot(assoc[0] - self.prev_seen[0] - dx,
                                             assoc[1] - self.prev_seen[1] - dy)
                                  / (2 * assoc[2]) / (t - self.last_seen_t))
            self.tk, self.tk_pred = assoc, None
            self.last_seen_t = t
            self.vanish_armed_for_loss = False
        elif self.tk is not None or self.tk_pred is not None:
            if self.tk_pred is None:                       # loss begins
                self.tk_pred = (px, py, self.tk[2]) if px is not None else self.tk
                self.lost_pos = (self.tk[0], self.tk[1])
            if t - self.last_seen_t > LOST_MAX_S:          # track dies
                self.tk = self.tk_pred = None
                self.rest_time = 0.0
                self._break_track()
        elif cands:
            self.tk = max(cands, key=lambda c: c[2])
            self.tk_pred = None
            self.last_seen_t = t
            self.rest_time = 0.0
            self.vanish_armed_for_loss = False
            self._break_track()

        tracked = self.tk is not None and self.tk_pred is None

        # -- rest / address arming --
        if true_speed is not None:
            if true_speed < REST_SPEED:
                self.rest_time += (t - (self.prev_seen[2] if self.prev_seen else t)) or 0.0
                if self.rest_time >= REST_RUN_S:
                    self.last_rest_t = t
                    self.anchor = self.tk
            else:
                self.rest_time = 0.0

        # -- club-at-ball memory + dwell history --
        if tracked:
            D = 2 * self.tk[2]
            mind = 1e9
            for (cx, cy) in clubs:
                mind = min(mind, math.hypot(cx - self.tk[0], cy - self.tk[1]) / D)
            near = mind < CLUB_NEAR_D
            if near:
                self.last_club_t = t
            dt_c = t - self.club_hist[-1][0] if self.club_hist else 0.0
            self.club_hist.append((t, min(dt_c, 0.2), near, mind))
            while self.club_hist and t - self.club_hist[0][0] > DWELL_WIN_S:
                self.club_hist.pop(0)

        armed = self.anchor is not None and (t - self.last_rest_t) <= ADDR_HOLD_S
        club_ok = (t - self.last_club_t) <= CLUB_MEM_S

        # ---- trigger 1: LAUNCH ----
        if (self.launch is None and self.vanish is None and armed and club_ok and tracked
                and (t - self.last_rest_t) <= LAUNCH_WIN_S
                and (t - self.last_fire_t) >= COOLDOWN_S):
            D = 2 * self.anchor[2]
            disp = math.hypot(self.tk[0] - self.anchor[0], self.tk[1] - self.anchor[1]) / max(D, 1e-6)
            if disp >= LAUNCH_D and self.tk[2] <= GROW_MAX * self.anchor[2]:
                self.launch = {"fire_t": self.last_rest_t, "ax": self.anchor[0],
                               "ay": self.anchor[1], "ar": self.anchor[2], "broke": False}

        # ---- trigger 2: VANISH ----
        lost_for = t - self.last_seen_t if (self.tk_pred is not None) else 0.0
        if (self.vanish is None and self.launch is None and armed and club_ok
                and self.tk_pred is not None and not self.vanish_armed_for_loss
                and lost_for >= GONE_S and self.lost_pos is not None
                and (t - self.last_fire_t) >= COOLDOWN_S):
            self.vanish_armed_for_loss = True
            D = 2 * self.anchor[2]
            last_disp = math.hypot(self.tk_pred[0] - self.anchor[0],
                                   self.tk_pred[1] - self.anchor[1]) / max(D, 1e-6)
            dwell = sum(h[1] for h in self.club_hist if h[2])
            touch = any(h[3] <= TOUCH_D for h in self.club_hist)
            if last_disp <= VANISH_NEAR_D and (dwell >= DWELL_MIN_S or touch):
                # in-place vanish (exit test now, as offline: swing head is ~still over the loss
                # window; a walk-away is already mid-pan). The dwell/touch gate above is the
                # 15 fps walk-killer: a dangling club never SITS at the ball.
                dvx = self.tk_pred[0] - self.lost_pos[0]
                dvy = self.tk_pred[1] - self.lost_pos[1]
                drift = math.hypot(dvx, dvy)
                exit_like = False
                if drift >= DRIFT_MIN_D * D:
                    ray, edge = self._ray(self.lost_pos[0], self.lost_pos[1], dvx, dvy)
                    near_b = min(self.lost_pos[0], self.W - self.lost_pos[0],
                                 self.lost_pos[1], self.H - self.lost_pos[1]) < EDGE_D * D
                    exit_like = ray < EXIT_FRAC * min(self.W, self.H) or near_b
                    if exit_like and edge == "b" and dvy > abs(dvx):
                        exit_like = False
                if not exit_like:
                    self.vanish = {"fire_t": max(self.last_rest_t, self.last_seen_t),
                                   "ax": self.anchor[0], "ay": self.anchor[1],
                                   "ar": self.anchor[2], "seen_s": 0.0, "start_t": t, "prev_t": t}
                else:
                    self._reset_address()
            elif last_disp > VANISH_NEAR_D and self.speed_dps >= DEPART_SPEED:
                self.vanish = {"fire_t": self.last_rest_t, "ax": self.anchor[0],
                               "ay": self.anchor[1], "ar": self.anchor[2],
                               "seen_s": 0.0, "start_t": t, "prev_t": t}

        # ---- pending resolution ----
        if self.launch is not None:
            D = 2 * self.launch["ar"]
            vetoed = (tracked and not self.launch["broke"]
                      and math.hypot(self.tk[0] - self.launch["ax"],
                                     self.tk[1] - self.launch["ay"]) < RET_D * D)
            if vetoed:
                self.launch = None
                self._reset_address()
            elif t - self.launch["fire_t"] >= RET_LOOK_S:
                if self.launch["fire_t"] - self.last_fire_t >= COOLDOWN_S:
                    self.count += 1
                    self.fires.append(self.launch["fire_t"])
                    self.last_fire_t = self.launch["fire_t"]
                self.launch = None
                self._reset_address()

        if self.vanish is not None:
            D = 2 * self.vanish["ar"]
            dt_v = t - self.vanish["prev_t"]
            self.vanish["prev_t"] = t
            if any(math.hypot(x - self.vanish["ax"], y - self.vanish["ay"]) < RET_D * D
                   for (x, y, r) in cands):
                self.vanish["seen_s"] += dt_v
            total = t - self.vanish["start_t"]
            if total >= VANISH_HOLD_S:
                if self.vanish["seen_s"] < RET_FRAC * total:
                    if self.vanish["fire_t"] - self.last_fire_t >= COOLDOWN_S:
                        self.count += 1
                        self.fires.append(self.vanish["fire_t"])
                        self.last_fire_t = self.vanish["fire_t"]
                self.vanish = None
                self._reset_address()

        self.state = ("PEND" if (self.launch or self.vanish) else
                      "ADDRESS" if armed else "TRACK" if tracked else "SEARCH")

    def _ray(self, x, y, vx, vy):
        m = math.hypot(vx, vy)
        if m < 1e-9:
            return float("inf"), None
        ux, uy = vx / m, vy / m
        best, edge = float("inf"), None
        if ux > 1e-9 and (self.W - x) / ux < best: best, edge = (self.W - x) / ux, "r"
        if ux < -1e-9 and -x / ux < best: best, edge = -x / ux, "l"
        if uy > 1e-9 and (self.H - y) / uy < best: best, edge = (self.H - y) / uy, "b"
        if uy < -1e-9 and -y / uy < best: best, edge = -y / uy, "t"
        return best, edge


# ------------------------- harness: 15 fps, 640-space, translation-only -------------------------
SP = sys.argv[1] if len(sys.argv) > 1 else "."   # dir of <clip>_fr_det.json + <clip>_cam_affine.json
# (regenerate caches per video: golf/cache_dets.py then golf/cam_affine.py)
S = 640 / 2048.0
ORACLE = {
    "golf_010":   {"real": [32.8], "false": [], "closed": True},
    "golf_049":   {"real": [27.4, 34.2], "false": [14.9, 18.7, 29.8], "closed": True},
    "golf_039":   {"real": [30.0], "false": [6.0, 9.3, 39.2, 46.0, 53.0], "closed": True},
    "golf_051":   {"real": [18.0, 33.0], "false": [3.7, 27.1], "closed": False},
    "ho_0702_012": {"real": [11.9], "false": [], "closed": True},
    "ho_0703_011": {"real": [16.8, 22.0, 35.0], "false": [], "closed": True},
    "ho_0704_005": {"real": [16.9, 29.8, 33.0], "false": [], "closed": True},
    "ho_0705_012": {"real": [27.9], "false": [], "closed": True},
}
TOL = 1.0   # deferred-online port: allow slightly wider matching than offline (fires backdated)


def apply(M, x, y):
    if M is None:
        return x, y
    return M[0]*x + M[1]*y + M[2], M[3]*x + M[4]*y + M[5]


def run(name):
    d = json.load(open(f"{SP}/{name}_fr_det.json"))
    cams = json.load(open(f"{SP}/{name}_cam_affine.json"))
    fps = d["fps"]; fr = d["frames"]
    hc = OnlineHitCounter(640, 480)
    cx, cy = 1024.0, 768.0
    for k in range(0, len(fr), 2):           # 30 -> 15 fps
        t = k / fps
        balls = [(((b[0]+b[2])/2)*S, ((b[1]+b[3])/2)*S, ((b[2]-b[0]+b[3]-b[1])/4)*S) for b in fr[k]["b"]]
        clubs = [(((c[0]+c[2])/2)*S, ((c[1]+c[3])/2)*S) for c in fr[k]["c"]]
        # local camera translation = the composed affine displacement evaluated AT the point the
        # counter asks for (the Kotlin port approximates this with a local-patch block match)
        M1 = cams[k-2] if 2 <= k and k-2 < len(cams) else None
        M2 = cams[k-1] if 2 <= k and k-1 < len(cams) else None

        def cam_at(x, y, M1=M1, M2=M2):
            if k < 2:
                return 0.0, 0.0
            px, py = x / S, y / S
            qx, qy = apply(M1, px, py)
            qx, qy = apply(M2, qx, qy)
            return (qx - px) * S, (qy - py) * S
        hc.update(t, balls, clubs, cam_at)
    return hc.fires


total_real = total_caught = total_false = total_extra = 0
for name, o in ORACLE.items():
    fires = run(name)
    times = [round(f, 2) for f in fires]
    caught = set(); marks = []
    for tm in times:
        r = next((x for x in o["real"] if abs(tm - x) <= TOL and x not in caught), None)
        if r is not None: caught.add(r); marks.append(f"{tm} ok")
        elif any(abs(tm - x) <= TOL for x in o["false"]): marks.append(f"{tm} FALSE"); total_false += 1
        elif o["closed"]: marks.append(f"{tm} EXTRA"); total_extra += 1
        else: marks.append(f"{tm} uncertain")
    missed = [x for x in o["real"] if x not in caught]
    total_real += len(o["real"]); total_caught += len(caught)
    print(f"{name}: [{' | '.join(marks) or 'none'}]" + (f"  MISSED {missed}" if missed else ""))
print(f"\n=> 15fps/640/translation-only port: real {total_caught}/{total_real}, "
      f"false {total_false}, extra {total_extra}")
