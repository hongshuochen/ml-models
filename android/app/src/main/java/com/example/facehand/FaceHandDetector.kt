package com.example.facehand

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
    // Optional stage-2 landmarks, frame-normalized: [x0,y0, x1,y1, ...]. null when disabled.
    val keypoints: FloatArray? = null,
    // True if this hand is classified as the "L" (thumb-index) pose.
    val isL: Boolean = false,
    // Recognized identity for a face (gallery label) + its cosine similarity. null = unknown.
    val name: String? = null,
    val nameSim: Float = 0f,
)

/**
 * A selectable detector model: its bundled TFLite asset, a display name for the picker, and the
 * class labels it emits (the model's class index maps into this array). All variants share the same
 * NMS-free YOLO26 Pico-P4P5 architecture / IO, so they're interchangeable at runtime.
 */
enum class DetectorModel(val asset: String, val displayName: String, val labels: Array<String>) {
    FACE_HAND("face_hand.tflite", "Face + Hand", arrayOf("face", "hand")),
    FACE_HAND_QR_BAR(
        "face_hand_qr_bar.tflite", "Face + Hand + QR + Barcode",
        arrayOf("face", "hand", "qr", "barcode"),
    );

    /** True if this model emits qr/barcode boxes worth handing to the barcode decoder. */
    val decodesCodes get() = labels.contains("qr") || labels.contains("barcode")
}

/**
 * Face + hand (+ optional QR/barcode) detector backed by our compact YOLO26 TFLite model
 * (Pico-P4P5 + HaGRID).
 *
 * The model is NMS-free / end-to-end: input is float32 NHWC [1, 640, 640, 3] with RGB
 * values in [0,1]; output is float32 [1, 300, 6] where each row is
 * [x1, y1, x2, y2, conf, cls] (corners normalized to [0,1]; cls indexes [model]'s labels).
 * Rows are sorted by confidence descending, so we can stop interpreting once scores drop.
 */
class FaceHandDetector(context: Context, val model: DetectorModel = DetectorModel.FACE_HAND) {

    private val labels = model.labels

    companion object {
        const val INPUT = 640            // square model input (shared by all variants)
        const val NUM_DET = 300          // fixed number of output rows
        const val STRIDE = 6             // values per row: x1,y1,x2,y2,conf,cls
        const val SCORE_THRESHOLD = 0.5f
    }

    private val interpreter: Interpreter

    // Reusable buffers (avoid per-frame allocations).
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(INPUT * INPUT * 3 * 4).order(ByteOrder.nativeOrder())
    private val output = Array(1) { Array(NUM_DET) { FloatArray(STRIDE) } }
    private val pixels = IntArray(INPUT * INPUT)

    init {
        val options = Interpreter.Options().apply { setNumThreads(4) }
        interpreter = Interpreter(loadModelFile(context, model.asset), options)
    }

    /** Memory-map the .tflite from assets (requires noCompress "tflite" in Gradle). */
    private fun loadModelFile(context: Context, asset: String): MappedByteBuffer {
        context.assets.openFd(asset).use { fd ->
            FileInputStream(fd.fileDescriptor).use { input ->
                return input.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fd.startOffset,
                    fd.declaredLength,
                )
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

        // --- decode [1,300,6] ---
        val results = ArrayList<Detection>()
        val rows = output[0]
        for (i in 0 until NUM_DET) {
            val r = rows[i]
            val score = r[4]
            if (score < SCORE_THRESHOLD) break // rows are sorted desc -> nothing better follows
            val x1 = r[0].coerceIn(0f, 1f)
            val y1 = r[1].coerceIn(0f, 1f)
            val x2 = r[2].coerceIn(0f, 1f)
            val y2 = r[3].coerceIn(0f, 1f)
            if (x2 <= x1 || y2 <= y1) continue
            val cls = r[5].toInt().coerceIn(0, labels.size - 1)
            results.add(Detection(labels[cls], score, x1, y1, x2, y2))
        }
        return results
    }

    fun close() = interpreter.close()
}
