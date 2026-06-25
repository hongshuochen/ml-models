package com.example.facehand

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Region
import android.os.Build
import android.os.SystemClock
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

/**
 * Transparent overlay drawn on top of the camera preview. Maps detection boxes
 * (normalized to the upright camera frame) onto the view using the same cover-fit
 * crop the PreviewView uses (FILL_CENTER), mirroring horizontally for the front camera.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()
    private var srcW = 1
    private var srcH = 1
    private var mirror = false
    private var latencyMs = 0f

    private val facePaint = Paint().apply {
        color = Color.parseColor("#22d3ee") // cyan
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }
    private val handPaint = Paint().apply {
        color = Color.parseColor("#ef4444") // red
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }
    private val labelBg = Paint().apply { style = Paint.Style.FILL; isAntiAlias = true }
    private val labelText = Paint().apply {
        color = Color.parseColor("#0b0f17")
        textSize = 34f
        isFakeBoldText = true
        isAntiAlias = true
    }
    private val hudText = Paint().apply {
        color = Color.WHITE
        textSize = 34f
        isAntiAlias = true
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }
    private val kptDot = Paint().apply { color = Color.WHITE; style = Paint.Style.FILL; isAntiAlias = true }
    private val kptLine = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f; isAntiAlias = true }
    private val framePaint = Paint().apply { color = Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 5f; isAntiAlias = true }
    private var quad: FloatArray? = null // framing quad: 4 frame-normalized points (x,y)*4
    private var flashUntil = 0L          // shutter flash (capture feedback) end time

    /** Briefly flash the screen white to acknowledge a framing capture. */
    fun flashCapture() {
        flashUntil = SystemClock.elapsedRealtime() + FLASH_MS
        invalidate()
    }

    /**
     * @param dets boxes normalized to the upright frame
     * @param frameW/frameH upright camera-frame dimensions
     * @param mirrorX true for the front camera (preview is mirrored)
     * @param ms inference latency for the HUD
     */
    fun setResults(dets: List<Detection>, frameW: Int, frameH: Int, mirrorX: Boolean, ms: Float, quadPts: FloatArray? = null) {
        detections = dets
        srcW = frameW
        srcH = frameH
        mirror = mirrorX
        latencyMs = ms
        quad = quadPts
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (srcW <= 0 || srcH <= 0) return

        // Cover-fit: scale the frame to fill the view, cropping the overflow (matches FILL_CENTER).
        val scale = max(width.toFloat() / srcW, height.toFloat() / srcH)
        val dx = (width - srcW * scale) / 2f
        val dy = (height - srcH * scale) / 2f

        quad?.let { drawFraming(canvas, it, scale, dx, dy) }

        for (d in detections) {
            // Horizontal mirror for the front camera so boxes line up with the mirrored preview.
            val nx1 = if (mirror) 1f - d.x2 else d.x1
            val nx2 = if (mirror) 1f - d.x1 else d.x2
            val left = nx1 * srcW * scale + dx
            val right = nx2 * srcW * scale + dx
            val top = d.y1 * srcH * scale + dy
            val bottom = d.y2 * srcH * scale + dy

            val paint = if (d.label == "hand") handPaint else facePaint
            canvas.drawRect(left, top, right, bottom, paint)

            // Label chip — recognized name for known faces ("Alex 62%"), else class + conf.
            val text = if (d.name != null) "${d.name} ${(d.nameSim * 100).toInt()}%"
            else "${d.label} ${(d.score * 100).toInt()}%"
            val tw = labelText.measureText(text)
            val chipTop = if (top > 40f) top - 40f else top
            labelBg.color = paint.color
            canvas.drawRect(left, chipTop, left + tw + 16f, chipTop + 40f, labelBg)
            canvas.drawText(text, left + 8f, chipTop + 30f, labelText)

            // Stage-2 landmarks (when enabled): hand = connected skeleton, face = points only.
            d.keypoints?.let { kp ->
                drawKeypoints(canvas, kp, connect = d.label == "hand", color = paint.color, scale = scale, dx = dx, dy = dy)
            }
        }

        canvas.drawText(
            "${detections.size} objs  •  ${latencyMs.toInt()} ms",
            16f,
            height - 24f,
            hudText,
        )

        // Shutter flash on capture — fades out over FLASH_MS.
        val rem = flashUntil - SystemClock.elapsedRealtime()
        if (rem > 0) {
            val alpha = (rem.toFloat() / FLASH_MS * 200f).toInt().coerceIn(0, 200)
            canvas.drawColor(Color.argb(alpha, 255, 255, 255))
            invalidate() // keep animating the fade
        }
    }

    /** Draw frame-normalized keypoints; hand gets connecting edges, face just points. */
    private fun drawKeypoints(canvas: Canvas, kp: FloatArray, connect: Boolean, color: Int, scale: Float, dx: Float, dy: Float) {
        val n = kp.size / 2
        fun vx(i: Int) = (if (mirror) 1f - kp[i * 2] else kp[i * 2]) * srcW * scale + dx
        fun vy(i: Int) = kp[i * 2 + 1] * srcH * scale + dy
        if (connect) {
            kptLine.color = color
            for (e in HAND_EDGES) {
                if (e[0] >= n || e[1] >= n) continue
                canvas.drawLine(vx(e[0]), vy(e[0]), vx(e[1]), vy(e[1]), kptLine)
            }
        }
        for (i in 0 until n) canvas.drawCircle(vx(i), vy(i), 6f, kptDot)
    }

    /** Darken everything outside the 4-point framing quad (gesture-triggered). */
    private fun drawFraming(canvas: Canvas, q: FloatArray, scale: Float, dx: Float, dy: Float) {
        val pts = Array(4) { i ->
            val nx = if (mirror) 1f - q[i * 2] else q[i * 2]
            floatArrayOf(nx * srcW * scale + dx, q[i * 2 + 1] * srcH * scale + dy)
        }
        // order around the centroid so the polygon doesn't self-intersect (bowtie)
        val cxp = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4f
        val cyp = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4f
        val ordered = pts.sortedBy { Math.atan2((it[1] - cyp).toDouble(), (it[0] - cxp).toDouble()) }
        val path = Path()
        path.moveTo(ordered[0][0], ordered[0][1])
        for (i in 1 until 4) path.lineTo(ordered[i][0], ordered[i][1])
        path.close()
        canvas.save()
        if (Build.VERSION.SDK_INT >= 26) {
            canvas.clipOutPath(path)
        } else {
            @Suppress("DEPRECATION")
            canvas.clipPath(path, Region.Op.DIFFERENCE)
        }
        canvas.drawColor(0xB0000000.toInt()) // dim outside the frame
        canvas.restore()
        canvas.drawPath(path, framePaint) // outline the frame
    }

    companion object {
        private const val FLASH_MS = 220L // shutter-flash duration

        // MediaPipe 21-point hand topology, for connecting hand keypoints.
        private val HAND_EDGES = arrayOf(
            intArrayOf(0, 1), intArrayOf(1, 2), intArrayOf(2, 3), intArrayOf(3, 4),
            intArrayOf(0, 5), intArrayOf(5, 6), intArrayOf(6, 7), intArrayOf(7, 8),
            intArrayOf(5, 9), intArrayOf(9, 10), intArrayOf(10, 11), intArrayOf(11, 12),
            intArrayOf(9, 13), intArrayOf(13, 14), intArrayOf(14, 15), intArrayOf(15, 16),
            intArrayOf(13, 17), intArrayOf(17, 18), intArrayOf(18, 19), intArrayOf(19, 20),
            intArrayOf(0, 17),
        )
    }
}
