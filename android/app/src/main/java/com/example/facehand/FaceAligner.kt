package com.example.facehand

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint

/**
 * ArcFace face alignment. Warps a face into the canonical 112x112 ArcFace crop using the same
 * 5-point similarity transform InsightFace's `face_align.norm_crop` uses, so the embedding
 * model sees exactly what it was trained on.
 *
 * We fit a 4-DOF similarity transform (uniform scale + rotation + translation, no shear) from
 * the 5 detected landmarks to the fixed template by least squares — the closed-form Horn/Umeyama
 * 2D solution, which reproduces skimage's SimilarityTransform.estimate (what InsightFace uses).
 */
class FaceAligner {

    // ArcFace destination template for a 112x112 crop (insightface face_align.arcface_dst),
    // points in order: left_eye, right_eye, nose, left_mouth, right_mouth.
    private val dst = floatArrayOf(
        38.2946f, 51.6963f,
        73.5318f, 51.5014f,
        56.0252f, 71.7366f,
        41.5493f, 92.3655f,
        70.7299f, 92.2041f,
    )

    private val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)

    /**
     * @param upright the full upright frame bitmap
     * @param kp5 the 5 face landmarks, frame-normalized [x0,y0,x1,y1,...], SAME order as [dst]
     * @return a 112x112 aligned RGB bitmap, or null if the points are degenerate
     */
    fun align(upright: Bitmap, kp5: FloatArray): Bitmap? {
        if (kp5.size < N * 2) return null
        val w = upright.width
        val h = upright.height

        // Source landmarks in pixel coords of the upright frame.
        val sx = FloatArray(N)
        val sy = FloatArray(N)
        for (i in 0 until N) { sx[i] = kp5[i * 2] * w; sy[i] = kp5[i * 2 + 1] * h }

        // Centroids of src and dst.
        var mx = 0f; var my = 0f; var dmx = 0f; var dmy = 0f
        for (i in 0 until N) { mx += sx[i]; my += sy[i]; dmx += dst[i * 2]; dmy += dst[i * 2 + 1] }
        mx /= N; my /= N; dmx /= N; dmy /= N

        // Closed-form least-squares similarity:  X = a*x - b*y + tx ,  Y = b*x + a*y + ty
        var den = 0f; var numA = 0f; var numB = 0f
        for (i in 0 until N) {
            val xi = sx[i] - mx; val yi = sy[i] - my
            val xj = dst[i * 2] - dmx; val yj = dst[i * 2 + 1] - dmy
            den += xi * xi + yi * yi
            numA += xj * xi + yj * yi
            numB += yj * xi - xj * yi
        }
        if (den < 1e-6f) return null
        val a = numA / den
        val b = numB / den
        val tx = dmx - (a * mx - b * my)
        val ty = dmy - (b * mx + a * my)

        // Android Matrix maps (x,y) -> (m0*x + m1*y + m2, m3*x + m4*y + m5).
        val matrix = Matrix()
        matrix.setValues(floatArrayOf(a, -b, tx, b, a, ty, 0f, 0f, 1f))

        val out = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        Canvas(out).apply {
            drawColor(Color.BLACK) // matches InsightFace warpAffine borderValue=0
            drawBitmap(upright, matrix, paint)
        }
        return out
    }

    companion object {
        private const val N = 5
        const val SIZE = 112
    }
}
