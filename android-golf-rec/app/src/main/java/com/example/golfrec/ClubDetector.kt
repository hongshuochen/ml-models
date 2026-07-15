package com.example.golfrec

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.qualcomm.qti.QnnDelegate
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

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
 * Club detector — the deployed 2-class YOLO26 model (ball + club_head), exported with the ONE-TO-MANY
 * (raw) head so it runs on the Qualcomm Hexagon NPU: the end-to-end head's topk/GatherNd can't be
 * delegated (GPU *or* NPU), but the raw head is pure conv+attention that the QNN HTP delegate takes
 * whole (581/581 nodes, ~18 ms end-to-end @640 on the S25 vs ~168 ms CPU). See [[android-golf-npu-deploy]].
 *
 * Model output = 3 NHWC feature maps `[1,80,80,6] [1,40,40,6] [1,20,20,6]` (strides 8/16/32). Per cell:
 * `[l,t,r,b, ball_logit, club_head_logit]` (reg_max=1 → distances are direct, NO DFL/softmax). Decode +
 * per-class NMS run here in Kotlin (the topk we moved off the graph). Backend: QNN NPU, else CPU.
 */
class ClubDetector(context: Context) {

    companion object {
        const val INPUT = 640            // square model input
        const val SCORE_THRESHOLD = 0.5f // keep a cell above this class prob (post-sigmoid)
        const val NMS_IOU = 0.5f
        private const val ASSET = "club.tflite"
        private const val TAG = "ClubDetector"
        private val LABELS = arrayOf("ball", "club_head")
    }

    private val interpreter: Interpreter
    private var qnnDelegate: QnnDelegate? = null
    private var backendName = "CPU"

    /** "NPU" / "CPU" — the accelerator that actually initialized (shown in the HUD). */
    val backend: String get() = backendName

    // Reusable buffers (avoid per-frame allocations).
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(INPUT * INPUT * 3 * 4).order(ByteOrder.nativeOrder())
    private val floatInput = inputBuffer.asFloatBuffer()          // bulk-write view (fast copy)
    private val floatArr = FloatArray(INPUT * INPUT * 3)          // scratch NHWC RGB [0,1]
    private val norm = FloatArray(256) { it / 255f }             // byte -> [0,1] LUT (no per-pixel divide)
    private val pixels = IntArray(INPUT * INPUT)

    // One output buffer per head, sized from the model at init: outputIndex -> [1][G][G][6], + its grid G.
    private val outBuffers = HashMap<Int, Array<Array<Array<FloatArray>>>>()
    private val gridByIndex = HashMap<Int, Int>()
    private val outputsMap = HashMap<Int, Any>()

    init {
        interpreter = buildInterpreter(context)
        for (i in 0 until interpreter.outputTensorCount) {
            val shape = interpreter.getOutputTensor(i).shape()   // [1, G, G, 6]
            val g = shape[1]
            val buf = Array(1) { Array(g) { Array(g) { FloatArray(shape[3]) } } }
            outBuffers[i] = buf
            gridByIndex[i] = g
            outputsMap[i] = buf
        }
        Log.i(TAG, "club detector backend = $backendName, ${outBuffers.size} heads")
    }

    /** Try the Hexagon NPU (QNN HTP) first; fall back to CPU if QNN isn't available on this device.
     *  The one-time HTP graph compile is cached (cacheDir + token) so only the first launch is slow.
     *  Runs on a BACKGROUND thread (the analysis executor lazy-builds the detector). */
    private fun buildInterpreter(context: Context): Interpreter {
        try {
            // The cDSP loads libQnnHtpV79Skel.so via fastrpc's file-service from ADSP_LIBRARY_PATH — with
            // useLegacyPackaging=true the skel is a real extracted file in nativeLibraryDir, which the app
            // process can read and stream to the DSP. Set the env BEFORE the QNN backend opens the session.
            val dspDir = context.applicationInfo.nativeLibraryDir
            try {
                android.system.Os.setenv("ADSP_LIBRARY_PATH",
                    "$dspDir;/vendor/dsp/cdsp;/vendor/lib64/rfs/dsp/snap;/vendor/lib/rfsa/adsp;/dsp", true)
            } catch (e: Throwable) { Log.w(TAG, "could not set ADSP_LIBRARY_PATH", e) }
            val opts = QnnDelegate.Options().apply {
                setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)          // Hexagon NPU
                // A third-party app must open the cDSP in UNSIGNED PD (signed PD is OEM-only) — without
                // this the fastrpc transport fails ("Failed to create transport for device, error 4000").
                setHtpPdSession(QnnDelegate.Options.HtpPdSession.HTP_PD_SESSION_UNSIGNED)
                setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)  // model is float16
                setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)   // HMX matrix engine
                setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
                setSkelLibraryDir(dspDir)                                            // DSP-readable skel dir
                setCacheDir(context.cacheDir.absolutePath)                           // cache the compiled graph
                setModelToken("club_rawhead_v1")                                     // -> fast subsequent launches
            }
            val d = QnnDelegate(opts)
            val interp = Interpreter(loadModelFile(context), Interpreter.Options().addDelegate(d))
            qnnDelegate = d
            backendName = "NPU"
            Log.i(TAG, "QNN HTP delegate initialized")
            return interp
        } catch (e: Throwable) {
            Log.w(TAG, "QNN NPU unavailable — falling back to CPU", e)
            qnnDelegate?.close(); qnnDelegate = null
        }
        backendName = "CPU"
        return Interpreter(loadModelFile(context), Interpreter.Options().apply { setNumThreads(4) })
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
        // --- preprocess: ARGB pixels -> NHWC RGB float32 in [0,1] (LUT + one bulk copy) ---
        val tp = System.nanoTime()
        scaled.getPixels(pixels, 0, INPUT, 0, 0, INPUT, INPUT)
        var j = 0
        for (p in pixels) {
            floatArr[j++] = norm[(p shr 16) and 0xFF] // R
            floatArr[j++] = norm[(p shr 8) and 0xFF]  // G
            floatArr[j++] = norm[p and 0xFF]          // B
        }
        floatInput.clear(); floatInput.put(floatArr)
        inputBuffer.rewind()

        // --- inference (3 raw feature maps) ---
        val ti = System.nanoTime()
        interpreter.runForMultipleInputsOutputs(arrayOf<Any>(inputBuffer), outputsMap)
        val td = System.nanoTime()

        // --- decode raw heads: per cell [l,t,r,b,ball,club] -> xyxy (normalized) + per-class NMS ---
        val inv = 1f / INPUT
        val cand = ArrayList<Detection>()
        for ((idx, buf) in outBuffers) {
            val grid = gridByIndex[idx]!!
            val stride = (INPUT / grid).toFloat()
            val map = buf[0]
            for (y in 0 until grid) {
                val row = map[y]
                for (x in 0 until grid) {
                    val c = row[x]
                    val sBall = sigmoid(c[4]); val sClub = sigmoid(c[5])
                    val score = if (sBall >= sClub) sBall else sClub
                    if (score < SCORE_THRESHOLD) continue
                    val cls = if (sClub > sBall) 1 else 0
                    val ax = x + 0.5f; val ay = y + 0.5f
                    val x1 = ((ax - c[0]) * stride * inv).coerceIn(0f, 1f)
                    val y1 = ((ay - c[1]) * stride * inv).coerceIn(0f, 1f)
                    val x2 = ((ax + c[2]) * stride * inv).coerceIn(0f, 1f)
                    val y2 = ((ay + c[3]) * stride * inv).coerceIn(0f, 1f)
                    if (x2 > x1 && y2 > y1) cand.add(Detection(LABELS[cls], score, x1, y1, x2, y2))
                }
            }
        }
        val out = nms(cand)
        if (++dbg % 20 == 0) Log.i(TAG, "pre=%.1f  inf=%.1f  dec=%.1f ms"
            .format((ti - tp) / 1e6f, (td - ti) / 1e6f, (System.nanoTime() - td) / 1e6f))
        return out
    }

    private var dbg = 0

    /** Greedy per-class NMS (few cells survive the 0.5 threshold, so this is cheap). */
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

    private fun sigmoid(v: Float): Float = 1f / (1f + exp(-v))

    fun close() {
        interpreter.close()
        qnnDelegate?.close()
    }
}
