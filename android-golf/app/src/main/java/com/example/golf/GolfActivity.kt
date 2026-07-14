package com.example.golf

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

/**
 * Golf hit counter. Back camera → golf detector (ball + club_head) → [GlobalMotion] (local
 * camera-motion estimate) → [HitCounter] (the ego-compensated v3 algorithm: putts AND full
 * swings) → live count + a state HUD. Reuses the app's proven CameraX + [GolfDetector] +
 * [OverlayView] plumbing. Note: a hit is counted 1–2 s after contact (veto windows must elapse).
 */
class GolfActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var countText: TextView
    private lateinit var hudText: TextView
    private lateinit var detector: GolfDetector
    private val hits = HitCounter()
    private val motion = GlobalMotion()
    private var lastHitNanos = -10_000_000_000L   // for the HIT/FOLLOW status window
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null
    private val lensFacing = CameraSelector.LENS_FACING_BACK   // golf = rear camera on the ball

    private val cameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else { Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_LONG).show(); finish() }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_golf)
        previewView = findViewById(R.id.previewView)
        overlay = findViewById(R.id.overlay)
        countText = findViewById(R.id.countText)
        hudText = findViewById(R.id.hudText)
        detector = GolfDetector(this)
        findViewById<Button>(R.id.resetButton).setOnClickListener { hits.reset(); motion.reset(); countText.text = "0" }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            startCamera()
        else cameraPermission.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({ cameraProvider = future.get(); bindCamera() }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera() {
        val provider = cameraProvider ?: return
        val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { it.setAnalyzer(analysisExecutor, ::analyze) }
        try {
            provider.unbindAll()
            provider.bindToLifecycle(this, selector, preview, analysis)
        } catch (e: Exception) {
            Log.e(TAG, "Camera bind failed", e)
            Toast.makeText(this, "Failed to start camera.", Toast.LENGTH_LONG).show()
        }
    }

    /** Analysis thread: detect → camera motion → hit state machine → overlay + HUD. */
    private fun analyze(image: ImageProxy) {
        try {
            val upright = image.toUprightBitmap()
            val scaled = Bitmap.createScaledBitmap(upright, GolfDetector.INPUT, GolfDetector.INPUT, true)
            val t0 = System.nanoTime()
            val dets = detector.detect(scaled)
            motion.prepare(scaled)                       // local camera-motion estimate (~2 ms)
            val counted = hits.update(dets, System.nanoTime() / 1e9, motion)
            val ms = (System.nanoTime() - t0) / 1_000_000f
            val w = upright.width; val h = upright.height
            if (counted) lastHitNanos = System.nanoTime()
            // status in the offline video's clearer vocabulary (IDLE / PREPARE / HIT / FOLLOW)
            val sinceHit = (System.nanoTime() - lastHitNanos) / 1e9
            val status = when {
                sinceHit < 0.4 -> "HIT"
                sinceHit < 1.2 -> "FOLLOW"
                hits.state == "ADDRESS" || hits.state == "PEND" -> "PREPARE"
                else -> "IDLE"
            }
            overlay.post {
                overlay.setResults(dets, w, h, false, ms)
                countText.text = hits.count.toString()
                if (counted) flashPutt()
                hudText.text = "%s   •   %s %d ms".format(status, detector.backend, ms.toInt())
            }
            if (scaled != upright) scaled.recycle()
            upright.recycle()
        } catch (e: Exception) {
            Log.e(TAG, "Frame analysis failed", e)
        } finally {
            image.close()
        }
    }

    private fun flashPutt() {
        countText.animate().scaleX(1.7f).scaleY(1.7f).setDuration(110).withEndAction {
            countText.animate().scaleX(1f).scaleY(1f).setDuration(200).start()
        }.start()
    }

    /** ImageProxy (RGBA_8888) → Bitmap rotated upright per the sensor rotation. */
    private fun ImageProxy.toUprightBitmap(): Bitmap {
        val bitmap = toBitmap()
        val rotation = imageInfo.rotationDegrees
        if (rotation == 0) return bitmap
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated != bitmap) bitmap.recycle()
        return rotated
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }

    companion object { private const val TAG = "GolfActivity" }
}
