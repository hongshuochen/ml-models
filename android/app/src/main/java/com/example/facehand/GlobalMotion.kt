package com.example.facehand

import android.graphics.Bitmap
import kotlin.math.abs

/**
 * Cheap LOCAL camera-motion estimator for the golf hit counter: block matching on a 160×160 luma
 * downsample of the 640×640 detector input. [at] returns the translation of static content from
 * the previous analyzed frame to the current one, sampled around a point of interest (the ball /
 * address anchor) — a phone-budget stand-in for the offline optical-flow affine
 * (golf/cam_affine.py). Translation-only is enough because the counter only compares positions
 * over short windows around that same point; validated offline at 15 fps (port_sim).
 *
 * Cost: one 160² downsample + ~300 SADs on a 48² patch ≈ 1–3 ms on a phone CPU.
 */
class GlobalMotion {
    private var prev: IntArray? = null
    private var cur: IntArray? = null
    private val pixels = IntArray(SIZE * SIZE)

    /** Feed this frame's 640×640 detector-input bitmap. Call once per frame, before [at]. */
    fun prepare(scaled640: Bitmap) {
        val small = Bitmap.createScaledBitmap(scaled640, SIZE, SIZE, true)
        small.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val luma = IntArray(SIZE * SIZE)
        for (i in pixels.indices) {
            val p = pixels[i]
            luma[i] = ((p shr 16 and 0xFF) + 2 * (p shr 8 and 0xFF) + (p and 0xFF)) shr 2
        }
        if (small != scaled640) small.recycle()
        prev = cur
        cur = luma
    }

    fun reset() { prev = null; cur = null }

    /**
     * Local (dx, dy) in 640-space of static content prev→cur around (x640, y640).
     * (0, 0) until two frames have been fed or when the patch is textureless.
     */
    fun at(x640: Float, y640: Float): FloatArray {
        val p = prev ?: return ZERO
        val c = cur ?: return ZERO
        val margin = PATCH / 2 + RANGE + 2
        val cx = (x640 / SCALE).toInt().coerceIn(margin, SIZE - margin - 1)
        val cy = (y640 / SCALE).toInt().coerceIn(margin, SIZE - margin - 1)
        var best = Long.MAX_VALUE
        var bx = 0
        var by = 0
        var dy = -RANGE
        while (dy <= RANGE) {                      // coarse pass, step 2
            var dx = -RANGE
            while (dx <= RANGE) {
                val s = sad(p, c, cx, cy, dx, dy, best)
                if (s < best) { best = s; bx = dx; by = dy }
                dx += 2
            }
            dy += 2
        }
        val fx = bx
        val fy = by
        for (ry in -1..1) for (rx in -1..1) {      // refine ±1 around the coarse best
            if (rx == 0 && ry == 0) continue
            val s = sad(p, c, cx, cy, fx + rx, fy + ry, best)
            if (s < best) { best = s; bx = fx + rx; by = fy + ry }
        }
        return floatArrayOf(bx * SCALE.toFloat(), by * SCALE.toFloat())
    }

    /** SAD of the PATCH² block at (cx,cy) in prev vs the (dx,dy)-shifted block in cur. */
    private fun sad(p: IntArray, c: IntArray, cx: Int, cy: Int, dx: Int, dy: Int, cut: Long): Long {
        var s = 0L
        val x0 = cx - PATCH / 2
        var yy = cy - PATCH / 2
        val yEnd = yy + PATCH
        while (yy < yEnd) {
            var i = yy * SIZE + x0
            var j = (yy + dy) * SIZE + x0 + dx
            repeat(PATCH) { s += abs(p[i] - c[j]); i++; j++ }
            if (s >= cut) return s                 // early out: already worse than the best
            yy++
        }
        return s
    }

    companion object {
        private const val SIZE = 160               // luma grid (640 / 4)
        private const val SCALE = FaceHandDetector.INPUT / SIZE
        private const val PATCH = 48               // matching block (≈192 px of the 640 frame)
        private const val RANGE = 16               // search ±16 (=±64 px/frame @640)
        private val ZERO = floatArrayOf(0f, 0f)
    }
}
