package com.example.golfrec

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View
import com.google.mlkit.vision.pose.PoseLandmark
import kotlin.math.max

/**
 * Transparent overlay on top of the camera preview. Maps golf detection boxes AND the ML Kit body
 * skeleton (both normalized to the upright camera frame) onto the view with the same cover-fit crop
 * the PreviewView uses (FILL_CENTER). ball = cyan, club_head = amber, pose = green.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()
    private var pose: Map<Int, PointF> = emptyMap()      // ML Kit landmarkType -> normalized point
    private var srcW = 1
    private var srcH = 1
    private var mirror = false
    private var latencyMs = 0f

    private val ballPaint = Paint().apply {
        color = Color.parseColor("#22d3ee"); style = Paint.Style.STROKE; strokeWidth = 6f; isAntiAlias = true
    }
    private val clubPaint = Paint().apply {
        color = Color.parseColor("#f59e0b"); style = Paint.Style.STROKE; strokeWidth = 6f; isAntiAlias = true
    }
    private val labelBg = Paint().apply { style = Paint.Style.FILL; isAntiAlias = true }
    private val labelText = Paint().apply {
        color = Color.parseColor("#0b0f17"); textSize = 34f; isFakeBoldText = true; isAntiAlias = true
    }
    private val posePaint = Paint().apply {
        color = Color.parseColor("#4ade80"); style = Paint.Style.STROKE; strokeWidth = 5f; isAntiAlias = true
    }
    private val poseDotPaint = Paint().apply {
        color = Color.parseColor("#4ade80"); style = Paint.Style.FILL; isAntiAlias = true
    }

    /**
     * @param dets boxes normalized to the upright frame
     * @param poseLandmarks ML Kit landmarkType -> normalized point (same upright frame)
     * @param frameW/frameH upright camera-frame dimensions
     * @param mirrorX true for the front camera (preview is mirrored)
     * @param ms inference latency (kept for the HUD; currently shown in the activity)
     */
    fun setResults(dets: List<Detection>, poseLandmarks: Map<Int, PointF>,
                   frameW: Int, frameH: Int, mirrorX: Boolean, ms: Float) {
        detections = dets
        pose = poseLandmarks
        srcW = frameW
        srcH = frameH
        mirror = mirrorX
        latencyMs = ms
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (srcW <= 0 || srcH <= 0) return

        // Cover-fit: scale the frame to fill the view, cropping the overflow (matches FILL_CENTER).
        val scale = max(width.toFloat() / srcW, height.toFloat() / srcH)
        val dx = (width - srcW * scale) / 2f
        val dy = (height - srcH * scale) / 2f
        fun mapX(nx: Float) = (if (mirror) 1f - nx else nx) * srcW * scale + dx
        fun mapY(ny: Float) = ny * srcH * scale + dy

        // --- ML Kit body skeleton (drawn first, under the boxes) ---
        for ((a, b) in POSE_CONNECTIONS) {
            val pa = pose[a] ?: continue
            val pb = pose[b] ?: continue
            canvas.drawLine(mapX(pa.x), mapY(pa.y), mapX(pb.x), mapY(pb.y), posePaint)
        }
        for (p in pose.values) canvas.drawCircle(mapX(p.x), mapY(p.y), 8f, poseDotPaint)

        // --- golf detections (ball + club_head) ---
        for (d in detections) {
            val left = mapX(if (mirror) d.x2 else d.x1)
            val right = mapX(if (mirror) d.x1 else d.x2)
            val top = mapY(d.y1)
            val bottom = mapY(d.y2)

            val paint = if (d.label == "club_head") clubPaint else ballPaint
            canvas.drawRect(left, top, right, bottom, paint)

            val text = "${d.label} ${(d.score * 100).toInt()}%"
            val tw = labelText.measureText(text)
            val chipTop = if (top > 40f) top - 40f else top
            labelBg.color = paint.color
            canvas.drawRect(left, chipTop, left + tw + 16f, chipTop + 40f, labelBg)
            canvas.drawText(text, left + 8f, chipTop + 30f, labelText)
        }
    }

    companion object {
        // Body skeleton connections (ML Kit PoseLandmark ids) — shoulders, arms, torso, legs.
        private val POSE_CONNECTIONS = listOf(
            PoseLandmark.LEFT_SHOULDER to PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_SHOULDER to PoseLandmark.LEFT_ELBOW,
            PoseLandmark.LEFT_ELBOW to PoseLandmark.LEFT_WRIST,
            PoseLandmark.RIGHT_SHOULDER to PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.RIGHT_ELBOW to PoseLandmark.RIGHT_WRIST,
            PoseLandmark.LEFT_SHOULDER to PoseLandmark.LEFT_HIP,
            PoseLandmark.RIGHT_SHOULDER to PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_HIP to PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_HIP to PoseLandmark.LEFT_KNEE,
            PoseLandmark.LEFT_KNEE to PoseLandmark.LEFT_ANKLE,
            PoseLandmark.RIGHT_HIP to PoseLandmark.RIGHT_KNEE,
            PoseLandmark.RIGHT_KNEE to PoseLandmark.RIGHT_ANKLE,
        )
    }
}
