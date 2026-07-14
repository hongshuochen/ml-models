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
 * Golf detector — the deployed 2-class YOLO26 model (ball + club_head), NMS-free, exported to
 * float16 TFLite (`golf.tflite`). Output is `(1, 300, 6)` = 300 rows of `[x1,y1,x2,y2,conf,cls]`,
 * already final (no NMS). See GOLF_YOLO.md for the model card.
 */
class GolfDetector(context: Context) {

    companion object {
        const val INPUT = 640            // square model input
        const val CHANNELS = 6           // per anchor: cx, cy, w, h, cls0, cls1 (all normalized [0,1])
        const val ANCHORS = 8400         // 80²+40²+20² grid points (anchor-free) at 640
        const val SCORE_THRESHOLD = 0.5f // keep an anchor above this class prob
        const val NMS_IOU = 0.5f
        private const val ASSET = "golf.tflite"
        private const val TAG = "GolfDetector"
        private val LABELS = arrayOf("ball", "club_head")
    }

    private val interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null
    private var backendName = "CPU"

    /** "GPU" / "NNAPI" / "CPU" — the accelerator that actually initialized (shown in the HUD). */
    val backend: String get() = backendName

    // Reusable buffers (avoid per-frame allocations).
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(INPUT * INPUT * 3 * 4).order(ByteOrder.nativeOrder())
    private val output = Array(1) { Array(CHANNELS) { FloatArray(ANCHORS) } }   // [1, 6, 8400]
    private val pixels = IntArray(INPUT * INPUT)

    init {
        interpreter = buildInterpreter(context)
        Log.i(TAG, "golf detector backend = $backendName")
    }

    /** Pick the faster of GPU vs CPU by timing a few inferences on each, then close the loser.
     *  (NNAPI is dropped: it is slow to compile at startup and, where tested, slower than CPU.)
     *  This runs on a BACKGROUND thread — see GolfActivity, which lazy-builds the detector on its
     *  analysis executor so the UI opens instantly. We don't gate GPU on CompatibilityList (often
     *  a false negative) — we just try it and fall back if it throws. */
    private fun buildInterpreter(context: Context): Interpreter {
        data class Cand(val name: String, val interp: Interpreter, val delegate: GpuDelegate?)
        val cands = ArrayList<Cand>()
        try {
            val d = GpuDelegate()
            cands.add(Cand("GPU", Interpreter(loadModelFile(context), Interpreter.Options().addDelegate(d)), d))
        } catch (e: Throwable) { Log.w(TAG, "GPU delegate unavailable", e) }
        cands.add(Cand("CPU", Interpreter(loadModelFile(context),
            Interpreter.Options().apply { setNumThreads(4) }), null))

        // if GPU didn't build there's nothing to compare — just use CPU (skip the benchmark)
        if (cands.size == 1) { backendName = "CPU"; return cands[0].interp }

        var best: Cand? = null
        var bestMs = Double.MAX_VALUE
        for (c in cands) {
            val ms = try {
                repeat(2) { c.interp.run(inputBuffer, output) }          // warmup
                val t0 = System.nanoTime()
                repeat(3) { c.interp.run(inputBuffer, output) }
                (System.nanoTime() - t0) / 3e6
            } catch (e: Throwable) { Log.w(TAG, "${c.name} inference failed", e); Double.MAX_VALUE }
            Log.i(TAG, "backend ${c.name}: ${ms.toInt()} ms")
            if (ms < bestMs) { bestMs = ms; best = c }
        }
        val winner = best ?: cands.last()
        for (c in cands) if (c !== winner) { c.interp.close(); c.delegate?.close() }  // free the loser
        backendName = winner.name
        gpuDelegate = winner.delegate
        return winner.interp
    }

    /** Memory-map the .tflite from assets (requires noCompress "tflite" in Gradle). */
    private fun loadModelFile(context: Context): MappedByteBuffer {
        context.assets.openFd(ASSET).use { fd ->
            FileInputStream(fd.fileDescriptor).use { input ->
                return input.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    /**
     * Run inference on a frame already scaled to [INPUT] x [INPUT] (ARGB_8888).
     * Returns boxes normalized to [0,1] of that square frame.
     */
    fun detect(scaled: Bitmap): List<Detection> {
        // --- preprocess: ARGB pixels -> NHWC RGB float32 in [0,1] ---
        scaled.getPixels(pixels, 0, INPUT, 0, 0, INPUT, INPUT)
        inputBuffer.rewind()
        for (p in pixels) {
            inputBuffer.putFloat(((p shr 16) and 0xFF) / 255f) // R
            inputBuffer.putFloat(((p shr 8) and 0xFF) / 255f)  // G
            inputBuffer.putFloat((p and 0xFF) / 255f)          // B
        }
        inputBuffer.rewind()

        // --- inference ---
        interpreter.run(inputBuffer, output)

        // --- decode raw [1,6,8400]: per anchor [cx,cy,w,h,cls0,cls1] normalized, then NMS ---
        // (this GPU-friendly raw head replaces the end-to-end NMS-free output, whose INT64/TopK
        //  ops the GPU delegate can't run; the top-k selection is done here instead.)
        val o = output[0]
        val cx = o[0]; val cy = o[1]; val w = o[2]; val h = o[3]; val s0 = o[4]; val s1 = o[5]
        val cand = ArrayList<Detection>()
        for (i in 0 until ANCHORS) {
            val a = s0[i]; val b = s1[i]
            val score = if (a >= b) a else b
            if (score < SCORE_THRESHOLD) continue
            val cls = if (b > a) 1 else 0
            val hw = w[i] * 0.5f; val hh = h[i] * 0.5f
            val x1 = (cx[i] - hw).coerceIn(0f, 1f)
            val y1 = (cy[i] - hh).coerceIn(0f, 1f)
            val x2 = (cx[i] + hw).coerceIn(0f, 1f)
            val y2 = (cy[i] + hh).coerceIn(0f, 1f)
            if (x2 > x1 && y2 > y1) cand.add(Detection(LABELS[cls], score, x1, y1, x2, y2))
        }
        return nms(cand)
    }

    /** Greedy per-class NMS (few candidates survive the 0.5 threshold, so this is cheap). */
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
