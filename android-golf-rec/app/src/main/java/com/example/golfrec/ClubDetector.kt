package com.example.golfrec

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
 * Golf detector — the deployed 2-class YOLO26 model (ball + club_head), end-to-end / NMS-free,
 * exported to int8 TFLite (`golf.tflite`; picked over fp16/fp32 and the raw head after an on-device
 * benchmark — int8 is ~2-3× faster on the ARM CPU, and raw≈e2e in speed so the simpler e2e head
 * with its baked-in selection wins). Output `(1,300,6)` = 300 rows `[x1,y1,x2,y2,conf,cls]`, final.
 * Backend is auto-picked (GPU vs CPU) at startup. See GOLF_YOLO.md.
 */
class ClubDetector(context: Context) {

    companion object {
        const val INPUT = 640            // square model input
        const val NUM_DET = 300          // fixed number of output rows
        const val STRIDE = 6             // values per row: x1,y1,x2,y2,conf,cls
        const val SCORE_THRESHOLD = 0.5f
        private const val ASSET = "club.tflite"
        private const val TAG = "ClubDetector"
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
    private val output = Array(1) { Array(NUM_DET) { FloatArray(STRIDE) } }
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

        // --- decode [1,300,6] (rows sorted by conf desc) ---
        val results = ArrayList<Detection>()
        val rows = output[0]
        for (i in 0 until NUM_DET) {
            val r = rows[i]
            val score = r[4]
            if (score < SCORE_THRESHOLD) break
            val x1 = r[0].coerceIn(0f, 1f)
            val y1 = r[1].coerceIn(0f, 1f)
            val x2 = r[2].coerceIn(0f, 1f)
            val y2 = r[3].coerceIn(0f, 1f)
            if (x2 <= x1 || y2 <= y1) continue
            val cls = r[5].toInt().coerceIn(0, LABELS.size - 1)
            results.add(Detection(LABELS[cls], score, x1, y1, x2, y2))
        }
        return results
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }
}
