package com.example.golf

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

/**
 * Transparent overlay on top of the camera preview. Maps golf detection boxes (normalized to the
 * upright camera frame) onto the view with the same cover-fit crop the PreviewView uses
 * (FILL_CENTER). ball = cyan, club_head = amber, hole = green; plus a small detection/latency HUD.
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

    private val ballPaint = Paint().apply {
        color = Color.parseColor("#22d3ee"); style = Paint.Style.STROKE; strokeWidth = 6f; isAntiAlias = true
    }
    private val clubPaint = Paint().apply {
        color = Color.parseColor("#f59e0b"); style = Paint.Style.STROKE; strokeWidth = 6f; isAntiAlias = true
    }
    private val holePaint = Paint().apply {   // 3-class model: hole = green
        color = Color.parseColor("#22c55e"); style = Paint.Style.STROKE; strokeWidth = 6f; isAntiAlias = true
    }
    private val labelBg = Paint().apply { style = Paint.Style.FILL; isAntiAlias = true }
    private val labelText = Paint().apply {
        color = Color.parseColor("#0b0f17"); textSize = 34f; isFakeBoldText = true; isAntiAlias = true
    }
    private val hudText = Paint().apply {
        color = Color.WHITE; textSize = 34f; isAntiAlias = true; setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }

    /**
     * @param dets boxes normalized to the upright frame
     * @param frameW/frameH upright camera-frame dimensions
     * @param mirrorX true for the front camera (preview is mirrored)
     * @param ms inference latency for the HUD
     */
    fun setResults(dets: List<Detection>, frameW: Int, frameH: Int, mirrorX: Boolean, ms: Float) {
        detections = dets
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

        for (d in detections) {
            val nx1 = if (mirror) 1f - d.x2 else d.x1
            val nx2 = if (mirror) 1f - d.x1 else d.x2
            val left = nx1 * srcW * scale + dx
            val right = nx2 * srcW * scale + dx
            val top = d.y1 * srcH * scale + dy
            val bottom = d.y2 * srcH * scale + dy

            val paint = when (d.label) { "club_head" -> clubPaint; "hole" -> holePaint; else -> ballPaint }
            canvas.drawRect(left, top, right, bottom, paint)

            val text = "${d.label} ${(d.score * 100).toInt()}%"
            val tw = labelText.measureText(text)
            val chipTop = if (top > 40f) top - 40f else top
            labelBg.color = paint.color
            canvas.drawRect(left, chipTop, left + tw + 16f, chipTop + 40f, labelBg)
            canvas.drawText(text, left + 8f, chipTop + 30f, labelText)
        }
        // (the latency / state read-out lives in the activity's HUD TextView — one ms is enough)
    }
}
