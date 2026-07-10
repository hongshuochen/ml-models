package com.example.facehand

import kotlin.math.hypot

/**
 * On-device putt counter from per-frame ball + club_head detections.
 *
 * Signal = nearest **ball ↔ club_head distance**, normalized by the club_head's size. This is
 * EGO-MOTION INVARIANT: both objects live in the same frame, so a head/camera move shifts them
 * together and their distance is preserved — no optical-flow / IMU compensation needed (validated
 * offline on real first-person putting clips, TRAINING.md §11c notes).
 *
 * A putt is the pattern:  ADDRESS (putter at ball, distance low)  →  ball SEPARATES and STAYS
 * separated (ball rolls away / leaves frame). The "stays separated" confirmation is what
 * distinguishes a real putt from a practice back-stroke (there the putter swings away then comes
 * BACK to the ball, so distance returns to low).
 *
 * All thresholds are tunable — calibrate on-device against real putt counts.
 */
class PuttCounter(
    private val dLow: Float = 1.6f,        // ≤ this (club-diagonals) = putter at ball (address)
    private val dHigh: Float = 3.0f,       // ≥ this = ball clearly separated from putter
    private val addressFrames: Int = 4,    // frames of "low" needed to arm an address
    private val confirmFrames: Int = 8,    // frames the separation must persist to count a putt
    private val cooldownFrames: Int = 40,  // min frames between putts (~1.3 s @30fps)
    private val maxAddrAgeFrames: Int = 30, // no ball-at-club for > this -> stale address (walking) -> drop
    private val maxSepFrames: Int = 22,     // a separation must confirm within this -> else stale
) {
    enum class State { WAITING_ADDRESS, ADDRESSED, SEPARATING }

    var count = 0; private set
    var state = State.WAITING_ADDRESS; private set
    var lastDistance: Float = Float.NaN; private set   // for HUD / debugging

    private var frame = 0
    private var lowStreak = 0
    private var sepStart = 0
    private var lastPutt = -10_000
    private var lastAddr = -10_000  // last frame a ball was genuinely AT the club (d <= dLow)
    private var sawBall = false     // a real ball was seen AT the club this address cycle

    /** Feed one frame's detections. Returns true exactly on the frame a putt is counted. */
    fun update(detections: List<Detection>): Boolean {
        frame++
        val d = nearestDistance(detections)   // null = no club (unknown); BIG = ball gone/far
        lastDistance = d ?: Float.NaN
        if (d != null && d <= dLow) lastAddr = frame
        // stale address = no ball at the club for a while (walking / looking away) -> drop it, so the
        // club reappearing without a ball can't fire a false HIT. Skip mid-separation (the ball has
        // legitimately left the club — a rolling putt / a struck ball flying away).
        if (state != State.SEPARATING && frame - lastAddr > maxAddrAgeFrames) {
            state = State.WAITING_ADDRESS; lowStreak = 0; sawBall = false
        }
        // a separation that never confirms within maxSep is stale (a long gap kept it alive) -> discard
        if (state == State.SEPARATING && frame - sepStart > maxSepFrames) {
            state = State.WAITING_ADDRESS; lowStreak = 0; sawBall = false
        }
        if (d == null) return false            // no putter -> hold; staleness handled above

        var counted = false
        when (state) {
            State.WAITING_ADDRESS -> {
                if (d <= dLow) { if (++lowStreak >= addressFrames) { state = State.ADDRESSED; sawBall = true } }
                else lowStreak = 0
            }
            State.ADDRESSED -> {
                if (d <= dLow) sawBall = true                                     // a real ball is at the club
                if (d >= dHigh) { state = State.SEPARATING; sepStart = frame }    // ball leaving
            }
            State.SEPARATING -> {
                if (d <= dLow) {
                    state = State.ADDRESSED; sawBall = true                       // came back = back-stroke, not a putt
                } else if (frame - sepStart >= confirmFrames) {                    // stayed separated = real putt
                    // require a real ball-at-address this cycle (guards "putter visible, no ball")
                    if (sawBall && frame - lastPutt >= cooldownFrames) { count++; counted = true; lastPutt = frame }
                    state = State.WAITING_ADDRESS
                    lowStreak = 0; sawBall = false
                }
            }
        }
        return counted
    }

    fun reset() {
        count = 0; state = State.WAITING_ADDRESS; lowStreak = 0; frame = 0; lastPutt = -10_000
        lastAddr = -10_000; sawBall = false
    }

    /**
     * Nearest ball→club_head center distance / club_head diagonal.
     *  - no club_head            -> null   (unknown; caller holds state)
     *  - club_head but no ball   -> BIG    (ball absent = separated / rolled out of frame)
     *  - both                    -> normalized distance
     */
    private fun nearestDistance(dets: List<Detection>): Float? {
        var club: Detection? = null
        for (x in dets) if (x.label == "club_head" && (club == null || x.score > club!!.score)) club = x
        club ?: return null
        val cx = (club!!.x1 + club!!.x2) / 2f; val cy = (club!!.y1 + club!!.y2) / 2f
        val cd = hypot((club!!.x2 - club!!.x1), (club!!.y2 - club!!.y1)).coerceAtLeast(1e-4f)
        var best = Float.NaN
        for (x in dets) if (x.label == "ball") {
            val bx = (x.x1 + x.x2) / 2f; val by = (x.y1 + x.y2) / 2f
            val nd = hypot(cx - bx, cy - by) / cd
            if (best.isNaN() || nd < best) best = nd
        }
        return if (best.isNaN()) BIG else best        // club present, no ball = ball gone
    }

    companion object { private const val BIG = 99f }
}
