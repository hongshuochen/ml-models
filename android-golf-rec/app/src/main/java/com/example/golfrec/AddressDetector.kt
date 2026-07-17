package com.example.golfrec

import kotlin.math.abs
import kotlin.math.max

/** A body landmark (or point), normalized to [0,1] of the frame (null = not confidently visible). */
data class Pt(val x: Float, val y: Float)

/**
 * Detects the golf ADDRESS of the person being filmed, robust to SIDE / down-the-line views (how
 * golf is almost always filmed) where the far arm is occluded so the two-hands geometry fails.
 * Instead of the hands it uses viewpoint-robust signals (validated on real third-person clips):
 *   - BENT     : the torso is tilted forward (shoulders forward of the hips),
 *   - CLUB LOW : a club_head OR ball is detected BELOW the hip line (i.e. down at the ball),
 *   - STILL    : the hip is held still for [holdS] (a golfer settles before swinging).
 *
 * Fires ONCE per address episode (re-arms only after the address is lost). Thresholds are in
 * body-relative units so they hold across how big the person appears — tune on-device.
 */
class AddressDetector(
    private val holdS: Double = 0.4,       // s the address must be held still to fire
    private val stillTol: Float = 0.04f,   // max hip drift (frac of frame) over the hold window = "still"
    private val tiltK: Float = 0.30f,      // torso tilt |dx|/|dy| over this = "bent forward"
    private val clubMemS: Double = 1.0,    // s a "club/ball low" sighting stays valid
    private val reArmLostS: Double = 0.6,  // s the address must be absent to re-arm after a fire
) {
    enum class State { SEARCHING, HUMAN, POSTURE, PROMPT }

    var state = State.SEARCHING; private set

    private var lastClubLow = -1e9
    private var postureSince = -1.0
    private var lostSince = -1.0
    private var fired = false
    private val hipHist = ArrayDeque<Triple<Double, Float, Float>>()  // (t, hipX, hipY)

    /**
     * Feed one frame. [shoulder]/[hip] are the normalized mid-points of the shoulders/hips (null if
     * the person isn't confidently visible); [clubLow] is true when a club_head or ball was detected
     * below the hip line this frame. Returns true on the single frame the address is confirmed.
     */
    fun update(t: Double, shoulder: Pt?, hip: Pt?, clubLow: Boolean): Boolean {
        if (clubLow) lastClubLow = t
        val clubOk = (t - lastClubLow) <= clubMemS
        val person = shoulder != null && hip != null
        val bent = if (shoulder != null && hip != null) {
            val dx = abs(shoulder.x - hip.x)
            val dy = max(abs(shoulder.y - hip.y), 1e-4f)
            dx / dy > tiltK                              // torso tilted forward from vertical
        } else false
        val address = person && bent && clubOk

        // stillness on the (stable) hip point
        if (address && hip != null) {
            hipHist.addLast(Triple(t, hip.x, hip.y))
            while (hipHist.isNotEmpty() && t - hipHist.first().first > holdS) hipHist.removeFirst()
        } else {
            hipHist.clear()
        }
        val still = address && hipHist.size >= 3 && drift() <= stillTol

        var confirmed = false
        when {
            address -> {
                lostSince = -1.0
                if (postureSince < 0) postureSince = t
                if (!fired && still && (t - postureSince) >= holdS) {
                    fired = true; confirmed = true
                }
                state = if (fired) State.PROMPT else State.POSTURE
            }
            else -> {                                   // not address -> HUMAN if a person is visible, else SEARCHING
                postureSince = -1.0
                if (lostSince < 0) lostSince = t
                val reArmed = t - lostSince >= reArmLostS
                if (reArmed) fired = false
                // brief debounce: hold POSTURE/PROMPT right after address is lost, else reflect presence
                if (!(!reArmed && (state == State.POSTURE || state == State.PROMPT)))
                    state = if (person) State.HUMAN else State.SEARCHING
            }
        }
        return confirmed
    }

    /** Manually re-arm (after the prompt is dismissed or a recording ends). */
    fun rearm() { fired = false; postureSince = -1.0; lostSince = -1.0; hipHist.clear(); state = State.SEARCHING }

    private fun drift(): Float {
        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for ((_, x, y) in hipHist) {
            minX = minOf(minX, x); maxX = maxOf(maxX, x); minY = minOf(minY, y); maxY = maxOf(maxY, y)
        }
        return maxOf(abs(maxX - minX), abs(maxY - minY))
    }
}
