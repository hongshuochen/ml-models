package com.example.facehand

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.Button
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
 * Live face + hand detection. CameraX feeds frames to an ImageAnalysis analyzer that
 * runs the TFLite model on each frame (KEEP_ONLY_LATEST = drop frames if inference is
 * slower than the camera, so the UI never lags), and an OverlayView draws the boxes.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var detector: FaceHandDetector

    // Single background thread for inference so the camera/UI thread is never blocked.
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    // Selected lens — toggled by the Flip button. Front (selfie) by default; preview mirrored.
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var cameraProvider: ProcessCameraProvider? = null

    private val cameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else {
                Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_LONG).show()
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        overlay = findViewById(R.id.overlay)
        detector = FaceHandDetector(this)
        findViewById<Button>(R.id.flipButton).setOnClickListener { flipCamera() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            cameraProvider = providerFuture.get()
            bindCamera()
        }, ContextCompat.getMainExecutor(this))
    }

    /** (Re)bind preview + analysis to the currently selected lens — on start and on flip. */
    private fun bindCamera() {
        val provider = cameraProvider ?: return
        val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        if (!provider.hasCamera(selector)) {
            Toast.makeText(this, "That camera isn't available on this device.", Toast.LENGTH_SHORT).show()
            return
        }
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }
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

    /** Toggle front <-> back and rebind. The overlay mirror follows lensFacing automatically. */
    private fun flipCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
            CameraSelector.LENS_FACING_BACK
        } else {
            CameraSelector.LENS_FACING_FRONT
        }
        bindCamera()
    }

    /** Runs on the analysis thread for every (latest) frame. */
    private fun analyze(image: ImageProxy) {
        try {
            val upright = image.toUprightBitmap()
            val scaled = Bitmap.createScaledBitmap(
                upright, FaceHandDetector.INPUT, FaceHandDetector.INPUT, true,
            )
            val t0 = System.nanoTime()
            val detections = detector.detect(scaled)
            val ms = (System.nanoTime() - t0) / 1_000_000f

            val mirror = lensFacing == CameraSelector.LENS_FACING_FRONT
            overlay.post { overlay.setResults(detections, upright.width, upright.height, mirror, ms) }
        } catch (e: Exception) {
            Log.e(TAG, "Frame analysis failed", e)
        } finally {
            image.close() // must close or the camera stalls
        }
    }

    /** ImageProxy (RGBA_8888) -> Bitmap rotated to upright per the sensor rotation. */
    private fun ImageProxy.toUprightBitmap(): Bitmap {
        val bitmap = toBitmap()
        val rotation = imageInfo.rotationDegrees
        if (rotation == 0) return bitmap
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        detector.close()
    }

    companion object {
        private const val TAG = "FaceHandDetector"
    }
}
