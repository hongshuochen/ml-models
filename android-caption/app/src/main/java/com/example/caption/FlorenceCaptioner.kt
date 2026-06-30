package com.example.caption

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.os.SystemClock
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Florence-2-base-ft captioner on ONNX Runtime — the exact pipeline verified offline against
 * transformers (see android-caption/florence_onnx_demo.py):
 *   preprocess 768² (ImageNet norm) -> vision_encoder -> [image_features ; embed(prompt)] ->
 *   encoder -> greedy NO-CACHE decoder loop (start=2, stop=2) -> byte-level decode.
 * No KV cache (the with-past export has a static-16 bug) — the vision encoder runs once and the
 * tiny BART decoder is re-run on the growing sequence; simple and robust for short captions.
 */
class FlorenceCaptioner(modelDir: File, private val tok: BartTokenizer) {

    private val env = OrtEnvironment.getEnvironment()
    private val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) }
    private val vision = session(modelDir, "vision_encoder_int8.onnx")
    private val embed = session(modelDir, "embed_tokens_int8.onnx")
    private val encoder = session(modelDir, "encoder_model_int8.onnx")
    private val decoder = session(modelDir, "decoder_model_int8.onnx")

    private fun session(dir: File, name: String) =
        env.createSession(File(dir, name).absolutePath, opts)

    fun caption(bitmap: Bitmap, task: BartTokenizer.Task, maxNew: Int = 64): Pair<String, Long> {
        val t0 = SystemClock.elapsedRealtime()

        // 1) image -> features [1, nImg, 768]
        val pxT = OnnxTensor.createTensor(env, fbuf(preprocess(bitmap)), longArrayOf(1, 3, IMG, IMG))
        val imgFeat = vision.run(mapOf("pixel_values" to pxT))[0] as OnnxTensor
        val nImg = imgFeat.info.shape[1].toInt()
        val combined = FloatArray((nImg + tok.promptIds(task).size) * D)
        imgFeat.floatBuffer.get(combined, 0, nImg * D)
        imgFeat.close(); pxT.close()

        // 2) prompt embeds -> append after image features
        val promptIds = tok.promptIds(task)
        val promptEmb = embedTokens(promptIds)
        System.arraycopy(promptEmb, 0, combined, nImg * D, promptIds.size * D)
        val len = nImg + promptIds.size

        // 3) encoder (keep hidden states + mask alive for the whole decode loop)
        val amaskT = OnnxTensor.createTensor(env, lbuf(LongArray(len) { 1L }), longArrayOf(1, len.toLong()))
        val encEmbT = OnnxTensor.createTensor(env, fbuf(combined), longArrayOf(1, len.toLong(), D))
        val ehs = encoder.run(mapOf("inputs_embeds" to encEmbT, "attention_mask" to amaskT))[0] as OnnxTensor
        encEmbT.close()

        // 4) greedy, no-cache decode
        val decIds = ArrayList<Long>().apply { add(EOS) } // decoder_start_token_id = 2
        val out = ArrayList<Int>()
        for (step in 0 until maxNew) {
            val decEmbT = OnnxTensor.createTensor(
                env, fbuf(embedTokens(decIds.toLongArray())), longArrayOf(1, decIds.size.toLong(), D))
            val res = decoder.run(
                mapOf("encoder_attention_mask" to amaskT, "encoder_hidden_states" to ehs, "inputs_embeds" to decEmbT),
                setOf("logits"))
            val logits = res[0] as OnnxTensor
            val vocab = logits.info.shape[2].toInt()
            val lb = logits.floatBuffer
            val base = (decIds.size - 1) * vocab
            var best = 0
            var bestV = Float.NEGATIVE_INFINITY
            for (v in 0 until vocab) {
                val x = lb.get(base + v)
                if (x > bestV) { bestV = x; best = v }
            }
            res.close(); decEmbT.close()
            if (best.toLong() == EOS) break
            out.add(best); decIds.add(best.toLong())
        }
        ehs.close(); amaskT.close()
        return tok.decode(out) to (SystemClock.elapsedRealtime() - t0)
    }

    private fun embedTokens(ids: LongArray): FloatArray {
        val t = OnnxTensor.createTensor(env, lbuf(ids), longArrayOf(1, ids.size.toLong()))
        val r = embed.run(mapOf("input_ids" to t))[0] as OnnxTensor
        val arr = FloatArray(ids.size * D)
        r.floatBuffer.get(arr)
        t.close(); r.close()
        return arr
    }

    private fun preprocess(bmp: Bitmap): FloatArray {
        val scaled = Bitmap.createScaledBitmap(bmp, IMG.toInt(), IMG.toInt(), true)
        val n = IMG.toInt()
        val px = IntArray(n * n)
        scaled.getPixels(px, 0, n, 0, 0, n, n)
        if (scaled != bmp) scaled.recycle()
        val out = FloatArray(3 * n * n)
        val plane = n * n
        for (i in 0 until plane) {
            val p = px[i]
            out[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
            out[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
            out[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
        }
        return out
    }

    private fun fbuf(a: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(a); rewind() }

    private fun lbuf(a: LongArray): LongBuffer =
        ByteBuffer.allocateDirect(a.size * 8).order(ByteOrder.nativeOrder()).asLongBuffer().apply { put(a); rewind() }

    fun close() {
        vision.close(); embed.close(); encoder.close(); decoder.close()
    }

    companion object {
        private const val IMG = 768L
        private const val D = 768
        private const val EOS = 2L
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}
