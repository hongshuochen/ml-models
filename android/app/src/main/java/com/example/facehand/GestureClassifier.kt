package com.example.facehand

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.hypot

/**
 * "L" (thumb-index) gesture classifier. Input = the 210 normalized pairwise distances
 * between a hand's 21 keypoints (rotation/translation/mirror-invariant by construction),
 * exactly matching train_gesture_mlp.py. The model outputs a logit -> we sigmoid it.
 */
class GestureClassifier(context: Context, modelAsset: String = "l_gesture.tflite") {

    private val interpreter: Interpreter
    private val pairs: Array<IntArray> = buildList {
        for (a in 0 until NK) for (b in a + 1 until NK) add(intArrayOf(a, b))
    }.toTypedArray() // 210 pairs, same order as itertools.combinations(range(21), 2)
    private val input = ByteBuffer.allocateDirect(pairs.size * 4).order(ByteOrder.nativeOrder())
    private val output = FloatArray(1) // model output shape is [1] (a single logit)

    init {
        interpreter = Interpreter(loadModelFile(context, modelAsset),
            Interpreter.Options().apply { setNumThreads(1) })
    }

    private fun loadModelFile(context: Context, asset: String): MappedByteBuffer {
        context.assets.openFd(asset).use { fd ->
            FileInputStream(fd.fileDescriptor).use {
                return it.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    /** cropPts = 42 crop-normalized hand keypoints (x0,y0,x1,y1,...). Returns P(L pose). */
    fun scoreL(cropPts: FloatArray): Float {
        val scale = hypot(cropPts[18] - cropPts[0], cropPts[19] - cropPts[1]) + 1e-6f // wrist->mid-MCP
        input.rewind()
        for (p in pairs) {
            val dx = cropPts[p[0] * 2] - cropPts[p[1] * 2]
            val dy = cropPts[p[0] * 2 + 1] - cropPts[p[1] * 2 + 1]
            input.putFloat(hypot(dx, dy) / scale)
        }
        input.rewind()
        interpreter.run(input, output)
        return 1f / (1f + exp(-output[0])) // sigmoid(logit)
    }

    fun close() = interpreter.close()

    companion object {
        const val NK = 21
        const val THRESHOLD = 0.6f
    }
}
