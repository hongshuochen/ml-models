package com.example.facehand

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.min

/**
 * On-device golf HIT counter — the online Kotlin port of golf/hit_detector.py v3 ("ego-fusion").
 * Counts BOTH putts (ball rolls away) and full swings (ball vanishes from a stable address).
 *
 * Core idea (validated offline): subtract CAMERA motion. A world-static ball under a walking /
 * turning head moves in the image but its TRUE (ground-relative) speed is ~0, while a real putt
 * rolls at ~30 D/s — so a low 5-diameter TRUE-displacement confirm is safe. [GlobalMotion]
 * supplies a local translation estimate around the tracked ball / address anchor.
 *
 * Differences from the offline reference (all validated by the port simulator, port_sim.py, on
 * 8 real egocentric clips subsampled to 15 fps with translation-only compensation —
 * 12/14 real hits, 0 false fires):
 *  - TIME-based thresholds (the app's analysis rate varies, ~15 fps inference-bound);
 *  - deferred confirmation: a hit is COUNTED 1–2 s after contact (veto windows must elapse);
 *  - the in-place vanish path additionally needs club-at-ball DWELL (≥0.4 s in the last 3 s) or
 *    one ≤0.8 D club touch — at 15 fps with translation-only compensation this is what rejects
 *    walk-aways; the accepted cost is missing tap-ins (sparse club detections at a 30 cm nudge).
 *
 * All thresholds are in ball diameters (D), seconds, or D/second, so they hold at any fps and
 * resolution; pixel sizes scale with the input width.
 */
class HitCounter {

    private class Ball(val x: Float, val y: Float, val r: Float)
    private class Launch(val fireT: Double, var ax: Float, var ay: Float, val ar: Float, var broke: Boolean)
    private class Vanish(val fireT: Double, var ax: Float, var ay: Float, val ar: Float) {
        var seenS = 0.0
        var startT = 0.0
        var prevT = 0.0
    }

    private class ClubObs(val t: Double, val dt: Double, val near: Boolean, val minD: Float)

    var count = 0; private set
    var state = "SEARCH"; private set

    // track
    private var tk: Ball? = null            // tracked ball (last associated detection)
    private var tkPred: Ball? = null        // camera-propagated position while LOST
    private var lastSeenT = -1e9
    private var lostX = 0f; private var lostY = 0f
    private var hasLostPos = false
    private var speedDps = 0f               // recent compensated ball speed (D/s)
    private var vanishArmedForLoss = false
    // address
    private var restTime = 0.0
    private var lastRestT = -1e9
    private var anchor: Ball? = null        // translation-propagated address spot
    private var lastClubT = -1e9
    private val clubHist = ArrayDeque<ClubObs>()
    // pendings
    private var launch: Launch? = null
    private var vanish: Vanish? = null
    private var lastFireT = -1e9

    fun reset() {
        count = 0; state = "SEARCH"
        tk = null; tkPred = null; lastSeenT = -1e9; hasLostPos = false
        speedDps = 0f; vanishArmedForLoss = false
        restTime = 0.0; lastRestT = -1e9; anchor = null; lastClubT = -1e9; clubHist.clear()
        launch = null; vanish = null; lastFireT = -1e9
    }

    private fun resetAddress() { anchor = null; lastRestT = -1e9; restTime = 0.0 }

    private fun breakTrack() { launch?.broke = true }

    /**
     * Feed one analyzed frame. [dets] in normalized [0,1] coords (as [FaceHandDetector] emits),
     * [t] a monotonic timestamp in seconds, [motion] already fed this frame's bitmap.
     * Returns true exactly when a hit is counted (1–2 s after the actual contact).
     */
    fun update(dets: List<Detection>, t: Double, motion: GlobalMotion): Boolean {
        // -- local camera translation, sampled at the point that matters --
        val a0 = anchor; val tp0 = tkPred; val tk0 = tk
        val rx = a0?.x ?: tp0?.x ?: tk0?.x ?: (W / 2f)
        val ry = a0?.y ?: tp0?.y ?: tk0?.y ?: (W / 2f)
        val d = motion.at(rx, ry)
        val dx = d[0]; val dy = d[1]

        // -- candidates (640-space, size-gated, deduped) --
        val cands = ArrayList<Ball>(4)
        var clubs: ArrayList<Ball>? = null
        for (det in dets) {
            val x = (det.x1 + det.x2) * 0.5f * W
            val y = (det.y1 + det.y2) * 0.5f * W
            val r = ((det.x2 - det.x1) + (det.y2 - det.y1)) * 0.25f * W
            if (det.label == "ball") {
                if (r > RMAX || r < RMIN_CAND) continue
                if (cands.any { hypot(x - it.x, y - it.y) < DEDUP_D * 2 * maxOf(it.r, r) }) continue
                cands.add(Ball(x, y, r))
            } else if (det.label == "club_head") {
                (clubs ?: ArrayList<Ball>(2).also { clubs = it }).add(Ball(x, y, r))
            }
        }

        // -- propagate camera-anchored state --
        anchor?.let { anchor = Ball(it.x + dx, it.y + dy, it.r) }
        launch?.let { it.ax += dx; it.ay += dy }
        vanish?.let { it.ax += dx; it.ay += dy }
        var px = Float.NaN; var py = Float.NaN
        val wasTracked = tk != null && tkPred == null
        if (wasTracked) {
            px = tk!!.x + dx; py = tk!!.y + dy
        } else if (tkPred != null) {
            tkPred = Ball(tkPred!!.x + dx, tkPred!!.y + dy, tkPred!!.r)
            px = tkPred!!.x; py = tkPred!!.y
        }

        // -- associate --
        var assoc: Ball? = null
        if (!px.isNaN() && cands.isNotEmpty()) {
            val dia = 2 * (if (wasTracked) tk!!.r else tkPred!!.r)
            val dtSeen = (t - lastSeenT).toFloat()
            val gate = (if (wasTracked) min(GATE_SPEED * dtSeen, GATE_CAP_D) else REACQ_D) * dia
            var bd = gate
            for (c in cands) {
                val dd = hypot(c.x - px, c.y - py)
                if (dd < bd) { bd = dd; assoc = c }
            }
        }

        var trueSpeed = Float.NaN
        if (assoc != null) {
            val gap = t - lastSeenT
            if (tkPred != null && gap > GAP_MAX_S) launch?.broke = true
            if (wasTracked && gap > 0) trueSpeed = (hypot(assoc.x - px, assoc.y - py) / (2 * assoc.r) / gap).toFloat()
            if (gap > GAP_MAX_S) restTime = 0.0
            val prevT = lastSeenT
            val prev = tk
            if (prev != null && prevT < t) {
                speedDps = (hypot(assoc.x - prev.x - dx, assoc.y - prev.y - dy) / (2 * assoc.r) / (t - prevT)).toFloat()
            }
            tk = assoc; tkPred = null; lastSeenT = t; vanishArmedForLoss = false
            // rest arming on TRUE (compensated) speed; NaN (re-acquired after a bridged flicker)
            // neither accumulates nor resets — the gap>GAP_MAX_S reset above handles real gaps
            if (!trueSpeed.isNaN()) {
                if (trueSpeed < REST_SPEED) {
                    restTime += t - prevT
                    if (restTime >= REST_RUN_S) { lastRestT = t; anchor = tk }
                } else restTime = 0.0
            }
        } else if (tk != null || tkPred != null) {
            if (tkPred == null) {                          // loss begins
                tkPred = if (!px.isNaN()) Ball(px, py, tk!!.r) else tk
                lostX = tk!!.x; lostY = tk!!.y; hasLostPos = true
            }
            if (t - lastSeenT > LOST_MAX_S) {              // track dies
                tk = null; tkPred = null; restTime = 0.0; breakTrack()
            }
        } else if (cands.isNotEmpty()) {
            tk = cands.maxByOrNull { it.r }                // fresh seed: largest = the near ball
            tkPred = null; lastSeenT = t; restTime = 0.0
            vanishArmedForLoss = false; breakTrack()
        }

        val tracked = tk != null && tkPred == null

        // -- club-at-ball memory + dwell history --
        if (tracked) {
            val dia = 2 * tk!!.r
            var minD = Float.MAX_VALUE
            clubs?.forEach { c -> minD = minOf(minD, (hypot(c.x - tk!!.x, c.y - tk!!.y) / dia).toFloat()) }
            val near = minD < CLUB_NEAR_D
            if (near) lastClubT = t
            val dtc = if (clubHist.isEmpty()) 0.0 else min(t - clubHist.last().t, 0.2)
            clubHist.addLast(ClubObs(t, dtc, near, minD))
            while (clubHist.isNotEmpty() && t - clubHist.first().t > DWELL_WIN_S) clubHist.removeFirst()
        }

        val armed = anchor != null && (t - lastRestT) <= ADDR_HOLD_S
        val clubOk = (t - lastClubT) <= CLUB_MEM_S

        // ---- trigger 1: LAUNCH (TRUE displacement from the propagated anchor) ----
        if (launch == null && vanish == null && armed && clubOk && tracked
            && (t - lastRestT) <= LAUNCH_WIN_S && (t - lastFireT) >= COOLDOWN_S) {
            val an = anchor!!
            val dia = 2 * an.r
            val disp = hypot(tk!!.x - an.x, tk!!.y - an.y) / dia
            if (disp >= LAUNCH_D && tk!!.r <= GROW_MAX * an.r) {
                launch = Launch(lastRestT, an.x, an.y, an.r, false)
            }
        }

        // ---- trigger 2: VANISH (track lost from an armed address) ----
        if (vanish == null && launch == null && armed && clubOk && tkPred != null
            && !vanishArmedForLoss && (t - lastSeenT) >= GONE_S && hasLostPos
            && (t - lastFireT) >= COOLDOWN_S) {
            vanishArmedForLoss = true
            val an = anchor!!
            val dia = 2 * an.r
            val lastDisp = hypot(tkPred!!.x - an.x, tkPred!!.y - an.y) / dia
            var dwell = 0.0
            var touch = false
            for (h in clubHist) { if (h.near) dwell += h.dt; if (h.minD <= TOUCH_D) touch = true }
            if (lastDisp <= VANISH_NEAR_D && (dwell >= DWELL_MIN_S || touch)) {
                // in-place vanish (swing): reject if the ball left the VIEW, not the SPOT
                val dvx = tkPred!!.x - lostX
                val dvy = tkPred!!.y - lostY
                var exitLike = false
                if (hypot(dvx, dvy) >= DRIFT_MIN_D * dia) {
                    val (ray, edge) = rayToBorder(lostX, lostY, dvx, dvy)
                    val nearBorder = minOf(lostX, W - lostX, lostY, W - lostY) < EDGE_D * dia
                    exitLike = ray < EXIT_FRAC * W || nearBorder
                    if (exitLike && edge == 'b' && dvy > abs(dvx)) exitLike = false  // head lift at strike
                }
                if (!exitLike) {
                    vanish = Vanish(maxOf(lastRestT, lastSeenT), an.x, an.y, an.r)
                        .also { it.startT = t; it.prevT = t }
                } else resetAddress()
            } else if (lastDisp > VANISH_NEAR_D && speedDps >= DEPART_SPEED) {
                // departed then lost = mid-roll loss confirms the roll (no exit test: a panning
                // head following a real putt looks exactly like an exit)
                vanish = Vanish(lastRestT, an.x, an.y, an.r).also { it.startT = t; it.prevT = t }
            }
        }

        // ---- pending resolution (deferred confirmation) ----
        var fired = false
        launch?.let { p ->
            val dia = 2 * p.ar
            val vetoed = tracked && !p.broke && hypot(tk!!.x - p.ax, tk!!.y - p.ay) < RET_D * dia
            if (vetoed) {                                   // continuous return = practice putt
                launch = null; resetAddress()
            } else if (t - p.fireT >= RET_LOOK_S) {
                if (p.fireT - lastFireT >= COOLDOWN_S) { count++; fired = true; lastFireT = p.fireT }
                launch = null; resetAddress()
            }
        }
        vanish?.let { p ->
            val dia = 2 * p.ar
            val dtv = t - p.prevT
            p.prevT = t
            if (cands.any { hypot(it.x - p.ax, it.y - p.ay) < RET_D * dia }) p.seenS += dtv
            val total = t - p.startT
            if (total >= VANISH_HOLD_S) {
                if (p.seenS < RET_FRAC * total) {           // persistent reappearance vetoes
                    if (p.fireT - lastFireT >= COOLDOWN_S) { count++; fired = true; lastFireT = p.fireT }
                }
                vanish = null; resetAddress()
            }
        }

        state = when {
            launch != null || vanish != null -> "PEND"
            armed -> "ADDRESS"
            tracked -> "TRACK"
            else -> "SEARCH"
        }
        return fired
    }

    /** Distance from (x,y) along (vx,vy) to the frame border and which border it hits. */
    private fun rayToBorder(x: Float, y: Float, vx: Float, vy: Float): Pair<Float, Char> {
        val m = hypot(vx, vy)
        if (m < 1e-9f) return Float.MAX_VALUE to ' '
        val ux = vx / m; val uy = vy / m
        var best = Float.MAX_VALUE; var edge = ' '
        if (ux > 1e-9f && (W - x) / ux < best) { best = (W - x) / ux; edge = 'r' }
        if (ux < -1e-9f && -x / ux < best) { best = -x / ux; edge = 'l' }
        if (uy > 1e-9f && (W - y) / uy < best) { best = (W - y) / uy; edge = 'b' }
        if (uy < -1e-9f && -y / uy < best) { best = -y / uy; edge = 't' }
        return best to edge
    }

    companion object {
        private const val W = FaceHandDetector.INPUT.toFloat()   // 640-square working space
        // physical-unit tunables — keep in sync with golf/hit_detector.py v3 + port_sim.py
        private const val REST_SPEED = 7.5f      // D/s: TRUE speed below this = at rest
        private const val REST_RUN_S = 0.17      // s of rest to arm an address
        private const val ADDR_HOLD_S = 1.5      // s an address stays armed after the last rest
        private const val CLUB_NEAR_D = 3.5f     // D: club within this = "at the ball"
        private const val CLUB_MEM_S = 3.0       // s club-at-ball memory (blur-invisible at impact)
        private const val LAUNCH_D = 5.0f        // D TRUE displacement that fires the launch path
        private const val LAUNCH_WIN_S = 1.0     // s: displacement must be reached this fast
        private const val GROW_MAX = 1.3f        // x: ball grown past this = picked up, not struck
        private const val GONE_S = 0.27          // s absence that opens a vanish decision
        private const val VANISH_NEAR_D = 3.0f   // D: last seen within this of anchor = in place
        private const val DEPART_SPEED = 9.0f    // D/s min speed for a mid-roll loss confirm
        private const val VANISH_HOLD_S = 1.0    // s reappearance-veto window after a vanish
        private const val RET_FRAC = 0.25        // fraction of hold seen near anchor that vetoes
        private const val RET_D = 3.0f           // D: reappearance/return radius
        private const val RET_LOOK_S = 2.0       // s launch-path continuous-return veto window
        private const val EDGE_D = 1.5f          // D: last seen this close to a border = view exit
        private const val EXIT_FRAC = 0.22f      // x W: drift ray shorter than this = view exit
        private const val DRIFT_MIN_D = 1.0f     // D: drift below this = camera steady, no exit test
        private const val COOLDOWN_S = 2.5       // s min separation between hits
        private const val GATE_SPEED = 150f      // D/s association gate while tracked ...
        private const val GATE_CAP_D = 8.0f      // ... capped (long dt must not allow teleports)
        private const val REACQ_D = 2.5f         // D: LOST re-acquire only near the propagated spot
        private const val GAP_MAX_S = 0.15       // s: flicker gap that is bridged
        private const val LOST_MAX_S = 2.0       // s: lost this long -> track dies
        private const val DWELL_MIN_S = 0.4      // s club-at-ball dwell required by in-place vanish
        private const val DWELL_WIN_S = 3.0      // s dwell look-back window
        private const val TOUCH_D = 0.8f         // D: one club sighting this close also qualifies
        private const val RMAX = 58f / 2048f * 640f       // px: max real ball radius (18.1 @640)
        private const val RMIN_CAND = 5f / 2048f * 640f   // px: smaller = detector noise
        private const val DEDUP_D = 1.0f         // D: closer candidate centers are duplicates
    }
}
