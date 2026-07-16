package com.example.golfrec

import kotlin.math.abs
import kotlin.math.hypot

/** A body landmark, normalized to [0,1] of the frame (null = not confidently visible). */
data class Pt(val x: Float, val y: Float)

/**
 * Detects the golf ADDRESS posture of the person being filmed, from ML Kit's 33-point pose plus our
 * club_head detector. Address = "standing over the ball, ready to swing", recognized by geometry
 * that is robust to viewpoint:
 *   - HANDS TOGETHER : the two wrists are close (both hands grip one club),
 *   - HANDS LOW      : the wrists are around/below hip height (reaching down to the ball),
 *   - ARMS DOWN      : the wrists hang well below the shoulders,
 *   - STILL          : the posture is held still for [holdS] (a golfer settles before swinging),
 *   - CLUB/BALL SEEN : a club_head OR ball was seen recently, anywhere in view — confirms it is
 *                      golf, not e.g. tying a shoe. Required for the fire (remembered clubMemS s).
 *
 * Fires ONCE per address episode (re-arms only after the posture is lost), so the recorder is
 * prompted once. All thresholds are in body-relative units (fractions of torso/shoulder length),
 * so they hold across how big the person appears — tune on-device.
 */
class AddressDetector(
    private val holdS: Double = 0.4,       // s the posture must be held still to fire
    private val stillTol: Float = 0.03f,   // max wrist drift (frac of frame) over the hold window = "still"
    private val handsTogetherK: Float = 0.7f,  // wrists closer than this × shoulder width
    private val handsLowK: Float = 0.15f,      // wrists at/below hipY − this × torso
    private val armsDownK: Float = 0.30f,      // wrists below shoulderY + this × torso
    private val clubMemS: Double = 1.5,    // s: a club sighting near the hands stays valid this long
    private val reArmLostS: Double = 0.6,  // s the posture must be absent to re-arm after a fire
) {
    enum class State { SEARCHING, HUMAN, POSTURE, PROMPT }

    var state = State.SEARCHING; private set
    var lastClubSeen = -1e9; private set

    private var postureSince = -1.0
    private var lostSince = -1.0
    private var fired = false
    private val wristHist = ArrayDeque<Triple<Double, Float, Float>>()  // (t, midWristX, midWristY)

    /** Call when a club_head OR ball is anywhere in view (golf context; remembered for [clubMemS]). */
    fun noteGolfEvidence(t: Double) { lastClubSeen = t }

    private fun mid(a: Pt?, b: Pt?): Pt? =
        if (a != null && b != null) Pt((a.x + b.x) / 2, (a.y + b.y) / 2) else (a ?: b)

    /**
     * Feed one frame's key landmarks (normalized [0,1], null if not visible). Returns true on the
     * single frame the address is confirmed (caller shows the "record?" prompt then).
     */
    fun update(
        t: Double,
        lWrist: Pt?, rWrist: Pt?, lElbow: Pt?, rElbow: Pt?,
        lShoulder: Pt?, rShoulder: Pt?, lHip: Pt?, rHip: Pt?,
    ): Boolean {
        val shoulder = mid(lShoulder, rShoulder)
        val hip = mid(lHip, rHip)
        val wrist = mid(lWrist, rWrist)
        val posture = isPosture(lWrist, rWrist, lShoulder, rShoulder, shoulder, hip, wrist)

        // stillness: keep ~holdS of mid-wrist positions, measure drift
        if (posture && wrist != null) {
            wristHist.addLast(Triple(t, wrist.x, wrist.y))
            while (wristHist.isNotEmpty() && t - wristHist.first().first > holdS) wristHist.removeFirst()
        } else {
            wristHist.clear()
        }
        val still = posture && wristHist.size >= 3 && drift() <= stillTol

        val clubOk = (t - lastClubSeen) <= clubMemS
        var confirmed = false
        when {
            posture -> {
                lostSince = -1.0
                if (postureSince < 0) postureSince = t
                if (!fired && still && (t - postureSince) >= holdS && clubOk) {
                    fired = true; confirmed = true
                }
                state = if (fired) State.PROMPT else State.POSTURE
            }
            else -> {                                   // no posture -> HUMAN if a person is visible, else SEARCHING
                postureSince = -1.0
                if (lostSince < 0) lostSince = t
                val reArmed = t - lostSince >= reArmLostS
                if (reArmed) fired = false               // re-arm for the next swing episode
                val human = shoulder != null             // a person's torso is in frame
                // brief debounce: hold POSTURE/PROMPT right after posture is lost, else reflect presence
                state = if (!reArmed && (state == State.POSTURE || state == State.PROMPT)) state
                        else if (human) State.HUMAN else State.SEARCHING
            }
        }
        return confirmed
    }

    /** Manually re-arm (e.g. after the user dismisses the prompt or a recording ends). */
    fun rearm() { fired = false; postureSince = -1.0; wristHist.clear(); state = State.SEARCHING }

    private fun isPosture(
        lw: Pt?, rw: Pt?, ls: Pt?, rs: Pt?, shoulder: Pt?, hip: Pt?, wrist: Pt?,
    ): Boolean {
        if (lw == null || rw == null || ls == null || rs == null || shoulder == null || hip == null || wrist == null)
            return false
        val shoulderW = hypot(ls.x - rs.x, ls.y - rs.y).coerceAtLeast(1e-4f)
        val torso = hypot(shoulder.x - hip.x, shoulder.y - hip.y).coerceAtLeast(1e-4f)
        val handsTogether = hypot(lw.x - rw.x, lw.y - rw.y) < handsTogetherK * shoulderW
        val handsLow = wrist.y > hip.y - handsLowK * torso        // y grows downward
        val armsDown = wrist.y > shoulder.y + armsDownK * torso
        return handsTogether && handsLow && armsDown
    }

    private fun drift(): Float {
        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for ((_, x, y) in wristHist) {
            minX = minOf(minX, x); maxX = maxOf(maxX, x); minY = minOf(minY, y); maxY = maxOf(maxY, y)
        }
        return maxOf(abs(maxX - minX), abs(maxY - minY))
    }
}
