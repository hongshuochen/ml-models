package com.example.caption

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.caption.databinding.ActivityMainBinding
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.max

/**
 * Single-screen app: load Florence-2 (download on first run) -> pick an image -> caption it
 * on-device and show the text + latency. Pick the task (Caption / Detailed / More detailed).
 */
class MainActivity : AppCompatActivity() {

    private lateinit var b: ActivityMainBinding
    private val io = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    @Volatile private var captioner: FlorenceCaptioner? = null
    private var bitmap: Bitmap? = null

    private val pick = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) captionUri(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityMainBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.pickButton.setOnClickListener { pick.launch("image/*") }
        loadModel()
    }

    private fun loadModel() = io.execute {
        try {
            val dir = ModelStore.ensure(this) { msg, pct ->
                main.post {
                    b.status.text = msg
                    b.progress.visibility = View.VISIBLE
                    b.progress.progress = pct
                }
            }
            val cap = FlorenceCaptioner(dir, BartTokenizer(File(dir, "vocab.json")))
            captioner = cap
            main.post {
                b.status.text = "Model ready — Florence-2-base-ft (int8). Pick an image."
                b.progress.visibility = View.GONE
                b.pickButton.isEnabled = true
            }
        } catch (e: Throwable) {
            main.post {
                b.status.text = "Model load failed: ${e.message}"
                b.progress.visibility = View.GONE
            }
        }
    }

    private fun task(): BartTokenizer.Task = when {
        b.taskCaption.isChecked -> BartTokenizer.Task.CAPTION
        b.taskMore.isChecked -> BartTokenizer.Task.MORE_DETAILED
        else -> BartTokenizer.Task.DETAILED
    }

    private fun captionUri(uri: Uri) {
        val cap = captioner ?: return
        val bmp = decodeBounded(uri) ?: run { b.caption.text = "Could not load image."; return }
        bitmap?.recycle(); bitmap = bmp
        b.preview.setImageBitmap(bmp)
        b.caption.text = "Captioning…"
        b.latency.text = ""
        b.pickButton.isEnabled = false
        val t = task()
        io.execute {
            try {
                val (text, ms) = cap.caption(bmp, t)
                main.post {
                    b.caption.text = text
                    b.latency.text = "Florence-2-base-ft · int8 · $ms ms"
                    b.pickButton.isEnabled = true
                }
            } catch (e: Throwable) {
                main.post { b.caption.text = "Caption failed: ${e.message}"; b.pickButton.isEnabled = true }
            }
        }
    }

    /** Decode an image, downsampling so the longest side is <= ~2048 px (avoids OOM on big photos). */
    private fun decodeBounded(uri: Uri): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        val longest = max(bounds.outWidth, bounds.outHeight)
        while (longest / sample > 2048) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        return contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
    }

    override fun onDestroy() {
        super.onDestroy()
        io.shutdown()
        captioner?.close()
        bitmap?.recycle()
    }
}
