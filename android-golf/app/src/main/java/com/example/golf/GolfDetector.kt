package com.example.golf

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
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
        const val NUM_DET = 300          // fixed number of output rows
        const val STRIDE = 6             // values per row: x1,y1,x2,y2,conf,cls
        const val SCORE_THRESHOLD = 0.5f
        private const val ASSET = "golf.tflite"
        private val LABELS = arrayOf("ball", "club_head")
    }

    private val interpreter: Interpreter

    // Reusable buffers (avoid per-frame allocations).
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(INPUT * INPUT * 3 * 4).order(ByteOrder.nativeOrder())
    private val output = Array(1) { Array(NUM_DET) { FloatArray(STRIDE) } }
    private val pixels = IntArray(INPUT * INPUT)

    init {
        val options = Interpreter.Options().apply { setNumThreads(4) }
        interpreter = Interpreter(loadModelFile(context), options)
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

    fun close() = interpreter.close()
}
