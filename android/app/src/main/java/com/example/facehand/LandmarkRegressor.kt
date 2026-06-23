package com.example.facehand

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Stage-2 keypoint regressor (MobileNetV3-small). Input is a 224x224 crop (NHWC, RGB
 * normalized to [0,1] — the ImageNet normalize + sigmoid are baked into the model), output
 * is `numKpts` (x,y) pairs in [0,1] of the crop. Used for both the 21-point hand model and
 * the 5-point face model.
 */
class LandmarkRegressor(context: Context, modelAsset: String, val numKpts: Int) {

    private val interpreter: Interpreter
    private val input = ByteBuffer.allocateDirect(INPUT * INPUT * 3 * 4).order(ByteOrder.nativeOrder())
    private val output = Array(1) { FloatArray(numKpts * 2) }
    private val pixels = IntArray(INPUT * INPUT)

    init {
        interpreter = Interpreter(
            loadModelFile(context, modelAsset),
            Interpreter.Options().apply { setNumThreads(2) },
        )
    }

    private fun loadModelFile(context: Context, asset: String): MappedByteBuffer {
        context.assets.openFd(asset).use { fd ->
            FileInputStream(fd.fileDescriptor).use { input ->
                return input.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    /** Returns numKpts*2 values (x0,y0,x1,y1,...) in [0,1] of the crop. */
    fun predict(crop: Bitmap): FloatArray {
        crop.getPixels(pixels, 0, INPUT, 0, 0, INPUT, INPUT)
        input.rewind()
        for (p in pixels) {
            input.putFloat(((p shr 16) and 0xFF) / 255f) // R
            input.putFloat(((p shr 8) and 0xFF) / 255f)  // G
            input.putFloat((p and 0xFF) / 255f)          // B
        }
        input.rewind()
        interpreter.run(input, output)
        return output[0]
    }

    fun close() = interpreter.close()

    companion object {
        const val INPUT = 224
    }
}
