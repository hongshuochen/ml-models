package com.example.caption

import android.content.Context
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * Fetches the Florence-2-base-ft int8 ONNX components (~261 MB) + vocab.json from Hugging Face
 * into filesDir on first launch (keeps the APK small). Idempotent: skips files already present.
 */
object ModelStore {
    private const val BASE = "https://huggingface.co/onnx-community/Florence-2-base-ft/resolve/main/"
    private val FILES = listOf(
        "onnx/vision_encoder_int8.onnx",
        "onnx/embed_tokens_int8.onnx",
        "onnx/encoder_model_int8.onnx",
        "onnx/decoder_model_int8.onnx",
        "vocab.json",
    )

    /** Returns the dir holding the (flattened) files. [onProgress] gets (message, percent|-1). */
    fun ensure(ctx: Context, onProgress: (String, Int) -> Unit): File {
        val dir = File(ctx.filesDir, "florence").apply { mkdirs() }
        FILES.forEachIndexed { i, rel ->
            val out = File(dir, rel.substringAfterLast('/'))
            if (out.exists() && out.length() > 0) return@forEachIndexed
            val tag = "Downloading ${out.name}  (${i + 1}/${FILES.size})"
            onProgress(tag, 0)
            download(BASE + rel, out) { pct -> onProgress(tag, pct) }
        }
        return dir
    }

    private fun download(url: String, out: File, onPct: (Int) -> Unit) {
        val tmp = File(out.parentFile, out.name + ".part")
        var current = url
        var conn = open(current)
        var redirects = 0
        while (conn.responseCode in 300..399 && redirects < 5) {
            // HF redirects can be RELATIVE (e.g. "/api/resolve-cache/..."); resolve against the
            // current URL so we don't feed a scheme-less path into URL() ("no protocol").
            val loc = conn.getHeaderField("Location") ?: break
            current = URL(URL(current), loc).toString()
            conn.disconnect()
            conn = open(current)
            redirects++
        }
        if (conn.responseCode != 200) {
            conn.disconnect()
            throw IOException("HTTP ${conn.responseCode} for $url")
        }
        val total = conn.contentLengthLong
        conn.inputStream.use { ins ->
            tmp.outputStream().use { os ->
                val buf = ByteArray(1 shl 16)
                var read = 0L
                var lastPct = -1
                while (true) {
                    val n = ins.read(buf)
                    if (n < 0) break
                    os.write(buf, 0, n)
                    read += n
                    if (total > 0) {
                        val pct = ((read * 100) / total).toInt()
                        if (pct != lastPct) { onPct(pct); lastPct = pct }
                    }
                }
            }
        }
        conn.disconnect()
        if (!tmp.renameTo(out)) throw IOException("rename ${tmp.name} -> ${out.name} failed")
    }

    private fun open(url: String): HttpURLConnection =
        (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 60_000
            instanceFollowRedirects = false // we follow manually (handles http<->https)
            connect()
        }
}
