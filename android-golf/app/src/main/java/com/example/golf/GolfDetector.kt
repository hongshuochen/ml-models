package com.example.golf

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/** One detected box. Coords are normalized [0,1] (corners) relative to the model input. */
data class Detection(
    val label: String,
    val score: Float,
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
)

/**
 * Golf detector — benchmark build. Loads a chosen `.tflite` [asset] on a chosen [backendPref]
 * ("CPU" / "GPU" / "NNAPI"), so the on-device model×backend picker can compare configs directly.
 *
 * Two output heads are supported and AUTO-DETECTED from the output tensor shape:
 *   - end-to-end  [1,300,6]  -> already-final rows [x1,y1,x2,y2,conf,cls] (NMS baked in)
 *   - raw         [1,6,8400] -> per grid point [cx,cy,w,h,cls0,cls1]; decode + NMS done here
 *     (the raw head drops the GPU-hostile INT64/TopK ops so the GPU delegate can run the convs).
 */
class GolfDetector(context: Context, val asset: String, val backendPref: String) {

    companion object {
        const val INPUT = 640
        const val SCORE_THRESHOLD = 0.5f
        const val NMS_IOU = 0.5f
        private const val TAG = "GolfDetector"
        private val LABELS = arrayOf("ball", "club_head")
    }

    private val interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null
    private var backendName = "CPU"

    /** Actual backend that loaded; "CPU(GPU✗)" etc. if the requested one failed and it fell back. */
    val backend: String get() = backendName

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(INPUT * INPUT * 3 * 4).order(ByteOrder.nativeOrder())
    private val pixels = IntArray(INPUT * INPUT)

    private val isRaw: Boolean
    private val outE2E: Array<Array<FloatArray>>?     // [1,300,6]
    private val outRaw: Array<Array<FloatArray>>?     // [1,6,8400]

    init {
        interpreter = buildInterpreter(context, backendPref)
        val oshape = interpreter.getOutputTensor(0).shape()   // [1,300,6] or [1,6,8400]
        isRaw = oshape.size == 3 && oshape[1] == 6
        outE2E = if (isRaw) null else Array(1) { Array(oshape[1]) { FloatArray(oshape[2]) } }
        outRaw = if (isRaw) Array(1) { Array(oshape[1]) { FloatArray(oshape[2]) } } else null
        Log.i(TAG, "asset=$asset backend=$backendName head=${if (isRaw) "raw" else "e2e"} out=${oshape.toList()}")
    }

    /** Force the requested backend; on failure fall back to CPU and mark it in [backendName]. */
    private fun buildInterpreter(context: Context, pref: String): Interpreter {
        try {
            when (pref) {
                "GPU" -> {
                    val d = GpuDelegate()
                    val it = Interpreter(loadModelFile(context), Interpreter.Options().addDelegate(d))
                    gpuDelegate = d; backendName = "GPU"; return it
                }
                "NNAPI" -> {
                    val it = Interpreter(loadModelFile(context),
                        Interpreter.Options().apply { setUseNNAPI(true); setNumThreads(4) })
                    backendName = "NNAPI"; return it
                }
            }
        } catch (e: Throwable) {
            Log.w(TAG, "$pref failed, using CPU", e); gpuDelegate?.close(); gpuDelegate = null
            backendName = "CPU($pref✗)"
            return Interpreter(loadModelFile(context), Interpreter.Options().apply { setNumThreads(4) })
        }
        backendName = "CPU"
        return Interpreter(loadModelFile(context), Interpreter.Options().apply { setNumThreads(4) })
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        context.assets.openFd(asset).use { fd ->
            FileInputStream(fd.fileDescriptor).use { input ->
                return input.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    /** Run inference on a frame already scaled to [INPUT]×[INPUT]. Boxes normalized to [0,1]. */
    fun detect(scaled: Bitmap): List<Detection> {
        scaled.getPixels(pixels, 0, INPUT, 0, 0, INPUT, INPUT)
        inputBuffer.rewind()
        for (p in pixels) {
            inputBuffer.putFloat(((p shr 16) and 0xFF) / 255f)
            inputBuffer.putFloat(((p shr 8) and 0xFF) / 255f)
            inputBuffer.putFloat((p and 0xFF) / 255f)
        }
        inputBuffer.rewind()
        return if (isRaw) decodeRaw() else decodeE2E()
    }

    private fun decodeE2E(): List<Detection> {
        interpreter.run(inputBuffer, outE2E)
        val rows = outE2E!![0]
        val out = ArrayList<Detection>()
        for (r in rows) {
            val score = r[4]
            if (score < SCORE_THRESHOLD) break            // rows sorted desc
            val x1 = r[0].coerceIn(0f, 1f); val y1 = r[1].coerceIn(0f, 1f)
            val x2 = r[2].coerceIn(0f, 1f); val y2 = r[3].coerceIn(0f, 1f)
            if (x2 > x1 && y2 > y1) out.add(Detection(LABELS[r[5].toInt().coerceIn(0, 1)], score, x1, y1, x2, y2))
        }
        return out
    }

    private fun decodeRaw(): List<Detection> {
        interpreter.run(inputBuffer, outRaw)
        val o = outRaw!![0]
        val cx = o[0]; val cy = o[1]; val w = o[2]; val h = o[3]; val s0 = o[4]; val s1 = o[5]
        val cand = ArrayList<Detection>()
        for (i in cx.indices) {
            val a = s0[i]; val b = s1[i]
            val score = if (a >= b) a else b
            if (score < SCORE_THRESHOLD) continue
            val hw = w[i] * 0.5f; val hh = h[i] * 0.5f
            val x1 = (cx[i] - hw).coerceIn(0f, 1f); val y1 = (cy[i] - hh).coerceIn(0f, 1f)
            val x2 = (cx[i] + hw).coerceIn(0f, 1f); val y2 = (cy[i] + hh).coerceIn(0f, 1f)
            if (x2 > x1 && y2 > y1) cand.add(Detection(LABELS[if (b > a) 1 else 0], score, x1, y1, x2, y2))
        }
        return nms(cand)
    }

    private fun nms(cand: ArrayList<Detection>): List<Detection> {
        if (cand.size <= 1) return cand
        cand.sortByDescending { it.score }
        val kept = ArrayList<Detection>(cand.size)
        val dead = BooleanArray(cand.size)
        for (i in cand.indices) {
            if (dead[i]) continue
            val a = cand[i]; kept.add(a)
            for (j in i + 1 until cand.size) {
                if (dead[j] || cand[j].label != a.label) continue
                if (iou(a, cand[j]) > NMS_IOU) dead[j] = true
            }
        }
        return kept
    }

    private fun iou(a: Detection, b: Detection): Float {
        val x1 = maxOf(a.x1, b.x1); val y1 = maxOf(a.y1, b.y1)
        val x2 = minOf(a.x2, b.x2); val y2 = minOf(a.y2, b.y2)
        val inter = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return if (ua <= 0f) 0f else inter / ua
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }
}
