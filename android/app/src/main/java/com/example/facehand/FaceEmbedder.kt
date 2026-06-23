package com.example.facehand

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * Face embedder backed by ArcFace MobileFaceNet (InsightFace w600k_mbf), converted to TFLite.
 *
 * Input is an ALIGNED 112x112 face (see [FaceAligner]) as NHWC RGB float32 normalized to
 * [-1,1] with (x-127.5)/127.5 — exactly InsightFace's preprocessing. Output is a 512-d
 * embedding which we L2-normalize so that cosine similarity is a plain dot product.
 *
 * The float32 TFLite is bit-exact to the source ONNX (verified parity cosine = 1.0).
 */
class FaceEmbedder(context: Context, modelAsset: String = MODEL_ASSET) {

    private val interpreter: Interpreter
    private val input = ByteBuffer.allocateDirect(SIZE * SIZE * 3 * 4).order(ByteOrder.nativeOrder())
    private val output = Array(1) { FloatArray(DIM) }
    private val pixels = IntArray(SIZE * SIZE)

    init {
        interpreter = Interpreter(
            loadModelFile(context, modelAsset),
            Interpreter.Options().apply { setNumThreads(2) },
        )
    }

    private fun loadModelFile(context: Context, asset: String): MappedByteBuffer {
        context.assets.openFd(asset).use { fd ->
            FileInputStream(fd.fileDescriptor).use {
                return it.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    /** Embed an aligned 112x112 face. Returns a unit-length 512-d vector. */
    fun embed(aligned: Bitmap): FloatArray {
        aligned.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        input.rewind()
        for (p in pixels) {
            input.putFloat((((p shr 16) and 0xFF) - 127.5f) / 127.5f) // R
            input.putFloat((((p shr 8) and 0xFF) - 127.5f) / 127.5f)  // G
            input.putFloat(((p and 0xFF) - 127.5f) / 127.5f)          // B
        }
        input.rewind()
        interpreter.run(input, output)
        return l2(output[0].copyOf())
    }

    private fun l2(v: FloatArray): FloatArray {
        var s = 0f
        for (x in v) s += x * x
        val inv = 1f / (sqrt(s) + 1e-9f)
        for (i in v.indices) v[i] *= inv
        return v
    }

    fun close() = interpreter.close()

    companion object {
        const val MODEL_ASSET = "face_embed.tflite"
        const val SIZE = 112
        const val DIM = 512
    }
}
