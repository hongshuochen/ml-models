package com.example.facehand

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.min

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

    // Separate thread for full-res framing captures, so decoding a big JPEG never stalls analysis.
    private val captureExecutor = Executors.newSingleThreadExecutor()

    // Selected lens — toggled by the Flip button. Front (selfie) by default; preview mirrored.
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var cameraProvider: ProcessCameraProvider? = null

    // Stage-2 landmark regressors (toggled by the top-right button).
    private var showLandmarks = false
    private lateinit var handReg: LandmarkRegressor
    private lateinit var faceReg: LandmarkRegressor
    private lateinit var gestureClf: GestureClassifier

    // Face recognition: detect -> track -> (on new/stale tracks) align -> embed -> match gallery.
    private lateinit var gallery: FaceGallery
    private lateinit var embedder: FaceEmbedder
    private val aligner = FaceAligner()
    private val faceTracker = FaceTracker()
    private val sharpBuf = IntArray(FaceEmbedder.SIZE * FaceEmbedder.SIZE)  // reused pixels for the blur check
    private val sharpGray = IntArray(FaceEmbedder.SIZE * FaceEmbedder.SIZE) // reused luma buffer

    // The two most-recently-seen faces shown under the gallery button.
    private lateinit var recent0: ImageView
    private lateinit var recent1: ImageView
    private var recentBmp0: Bitmap? = null
    private var recentBmp1: Bitmap? = null
    private var lastRecentId = -1

    // Framing capture: when the two-hand "L" frame is active, snapshot the framed square
    // (with padding) into the framing gallery, then cool down before the next shot.
    private lateinit var framingGallery: FramingGallery
    @Volatile private var lastFramingMs = 0L
    @Volatile private var framingArmed = false  // capture only when the toggle is on
    private var imageCapture: ImageCapture? = null     // full-res snapshot use case (null if unsupported)
    private val capturing = AtomicBoolean(false)       // guards against overlapping takePicture calls

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
        handReg = LandmarkRegressor(this, "hand_landmark.tflite", 21)
        faceReg = LandmarkRegressor(this, "face_landmark.tflite", 5)
        gestureClf = GestureClassifier(this)
        gallery = FaceGallery.get(this)
        // The embedder model is 13.6 MB — load it off the main thread so it doesn't lengthen cold
        // start. It initializes on the analysis executor; since frames are analyzed on that same
        // single thread (FIFO), the embedder is ready before the first face is ever embedded.
        analysisExecutor.execute { embedder = FaceEmbedder(this) }
        recent0 = findViewById(R.id.recent0)
        recent1 = findViewById(R.id.recent1)
        framingGallery = FramingGallery.get(this)
        findViewById<ImageButton>(R.id.flipButton).setOnClickListener { flipCamera() }
        findViewById<ImageButton>(R.id.landmarkButton).setOnClickListener { showLandmarks = !showLandmarks }
        findViewById<ImageButton>(R.id.galleryButton).setOnClickListener {
            startActivity(Intent(this, GalleryActivity::class.java))
        }
        // Framing button: tap = arm/disarm capture (icon turns cyan when armed), long-press = gallery.
        val framingBtn = findViewById<ImageButton>(R.id.framingGalleryButton)
        framingBtn.setOnClickListener {
            framingArmed = !framingArmed
            if (framingArmed) framingBtn.setColorFilter(Color.parseColor("#22d3ee")) else framingBtn.clearColorFilter()
            Toast.makeText(this, if (framingArmed) "Framing capture ON" else "Framing capture OFF", Toast.LENGTH_SHORT).show()
        }
        framingBtn.setOnLongClickListener {
            startActivity(Intent(this, FramingGalleryActivity::class.java)); true
        }

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
        // Pin both use cases to 4:3 so the analysis frame and the full-res JPEG share FOV + aspect
        // (a frame-normalized quad then maps directly onto the captured image — no offset).
        val ratio43 = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .build()
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setResolutionSelector(ratio43)
            .build()
            .also { it.setAnalyzer(analysisExecutor, ::analyze) }
        // Full-res snapshot use case for framing capture (latency-optimized for repeated shots).
        val capture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setResolutionSelector(ratio43)
            .build()
        try {
            provider.unbindAll()
            provider.bindToLifecycle(this, selector, preview, analysis, capture)
            imageCapture = capture
        } catch (e: Exception) {
            // Some devices can't bind 3 use cases — fall back to preview+analysis (framing capture off).
            Log.w(TAG, "3-use-case bind failed; retrying without ImageCapture", e)
            imageCapture = null
            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, preview, analysis)
            } catch (e2: Exception) {
                Log.e(TAG, "Camera bind failed", e2)
                Toast.makeText(this, "Failed to start camera.", Toast.LENGTH_LONG).show()
            }
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
            var detections = detector.detect(scaled)

            // Faces: track them so each face is recognized once per appearance, not every frame.
            faceTracker.update(detections.filter { it.label == "face" })

            // Hands always run landmarks (needed for the L-gesture). Faces run landmarks when
            // shown OR when their track needs a (re-)embedding for recognition.
            val lFingertips = ArrayList<FloatArray>()
            detections = detections.map { d ->
                when (d.label) {
                    "hand" -> {
                        val r = landmarksFor(upright, d) ?: return@map d
                        val (kp, isL) = r
                        if (isL) lFingertips.add(floatArrayOf(kp[8], kp[9], kp[16], kp[17])) // thumb tip(4), index tip(8)
                        d.copy(keypoints = if (showLandmarks) kp else null, isL = isL)
                    }
                    "face" -> recognizeFace(upright, d)
                    else -> d
                }
            }
            // Framing gesture: 2+ "L" hands -> quad from the two hands' thumb+index tips (4 points).
            val quad = if (lFingertips.size >= 2) {
                val a = lFingertips[0]; val b = lFingertips[1]
                floatArrayOf(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3])
            } else {
                null
            }
            // Auto-capture the framed square (full-res snapshot, cooldown-gated) while armed + framing.
            if (quad != null) maybeCaptureFraming(quad)
            val ms = (System.nanoTime() - t0) / 1_000_000f

            val w = upright.width
            val h = upright.height
            val mirror = lensFacing == CameraSelector.LENS_FACING_FRONT
            overlay.post { overlay.setResults(detections, w, h, mirror, ms, quad) }
            // Done with this frame's working bitmaps — free them so they don't pile up at 30fps.
            if (scaled != upright) scaled.recycle()
            upright.recycle()
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
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated != bitmap) bitmap.recycle() // free the unrotated source
        return rotated
    }

    /**
     * Crop the box (square + 1.3x pad, off-image padded black to match training), run the
     * matching regressor (hand = 21, face = 5), return frame-normalized keypoints.
     */
    private fun landmarksFor(upright: Bitmap, d: Detection): Pair<FloatArray, Boolean>? {
        val W = upright.width
        val H = upright.height
        val cx = (d.x1 + d.x2) / 2f * W
        val cy = (d.y1 + d.y2) / 2f * H
        val side = (max((d.x2 - d.x1) * W, (d.y2 - d.y1) * H) * 1.3f).toInt()
        if (side < 2) return null
        val x0 = (cx - side / 2f).toInt()
        val y0 = (cy - side / 2f).toInt()
        val srcL = x0.coerceIn(0, W); val srcT = y0.coerceIn(0, H)
        val srcR = (x0 + side).coerceIn(0, W); val srcB = (y0 + side).coerceIn(0, H)
        if (srcR <= srcL || srcB <= srcT) return null
        val square = Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888)
        Canvas(square).drawBitmap(
            upright,
            Rect(srcL, srcT, srcR, srcB),
            RectF((srcL - x0).toFloat(), (srcT - y0).toFloat(), (srcR - x0).toFloat(), (srcB - y0).toFloat()),
            null,
        )
        val crop = Bitmap.createScaledBitmap(square, LandmarkRegressor.INPUT, LandmarkRegressor.INPUT, true)
        val pts = (if (d.label == "hand") handReg else faceReg).predict(crop)
        if (crop != square) crop.recycle()
        square.recycle()
        val isL = d.label == "hand" && gestureClf.scoreL(pts) >= GestureClassifier.THRESHOLD
        val out = FloatArray(pts.size)
        var k = 0
        while (k < pts.size) {
            out[k] = (x0 + pts[k] * side) / W
            out[k + 1] = (y0 + pts[k + 1] * side) / H
            k += 2
        }
        return Pair(out, isL)
    }

    /**
     * For a face box: find its track and, when the track is new/stale and the face is good
     * enough, align -> embed -> match/enroll into the gallery. Heavy work runs at most once per
     * track per ~3s; otherwise the cached identity is reused. Returns the detection tagged with
     * the recognized name (and landmarks when the landmark overlay is on).
     */
    private fun recognizeFace(upright: Bitmap, d: Detection): Detection {
        // Only consider tracks actually matched to a detection THIS frame (miss == 0). The
        // tracker copies the matched detection's box into the track, so the right track has
        // IoU ~= 1 with this box — which avoids inheriting a nearby track's cached identity.
        val track = faceTracker.active
            .filter { it.miss == 0 }
            .map { it to it.iou(d) }
            .filter { it.second > 0.5f }
            .maxByOrNull { it.second }?.first

        val needEmbed = track != null && ::embedder.isInitialized &&
            faceTracker.needsEmbed(track) && faceQualityOk(d)
        var kp: FloatArray? = null
        if (needEmbed || showLandmarks) kp = landmarksFor(upright, d)?.first
        if (needEmbed && kp != null) identify(track!!, upright, kp)

        var out = d
        val person = track?.identityId?.takeIf { it >= 0 }?.let { gallery.person(it) }
        if (person != null && track != null) out = out.copy(name = person.label, nameSim = track.sim)
        if (showLandmarks && kp != null) out = out.copy(keypoints = kp)
        return out
    }

    /** Align the face, embed it, and match/enroll into the gallery; updates the track + thumbnails. */
    private fun identify(track: FaceTracker.Track, upright: Bitmap, kp5: FloatArray) {
        val aligned = aligner.align(upright, kp5) ?: return
        // Skip blurry frames — they make bad embeddings (duplicates) and ugly thumbnails. Don't
        // mark embedded, so the track keeps trying until a sharp frame arrives.
        if (sharpness(aligned) < SHARP_MIN) { aligned.recycle(); return }
        val emb = embedder.embed(aligned)
        faceTracker.markEmbedded(track)
        val m = gallery.match(emb)
        val best = m?.second ?: -1f
        val person = when {
            // Confident match -> refine that identity's template.
            m != null && best >= RECOG_THRESHOLD -> { gallery.reinforce(m.first, emb, aligned); track.sim = best; m.first }
            // Clearly nobody (well below any same-person score) AND not yet recognized -> enroll new.
            track.identityId < 0 && best < ENROLL_MARGIN -> { track.sim = 1f; gallery.enroll(emb, aligned) }
            // Gray zone (0.25-0.30) or a recognized track's off frame -> don't duplicate; keep state.
            else -> { aligned.recycle(); return }
        }
        track.identityId = person.id
        pushRecent(aligned, person.id)
    }

    /** Variance-of-Laplacian sharpness of the aligned 112x112 face — low value = blurry. */
    private fun sharpness(aligned: Bitmap): Double {
        val n = aligned.width
        val h = aligned.height
        val total = n * h
        if (total > sharpBuf.size) return Double.MAX_VALUE // aligned is 112x112; safety only
        aligned.getPixels(sharpBuf, 0, n, 0, 0, n, h)
        val g = sharpGray
        for (i in 0 until total) {
            val p = sharpBuf[i]
            g[i] = ((p shr 16 and 0xFF) * 77 + (p shr 8 and 0xFF) * 150 + (p and 0xFF) * 29) shr 8 // luma 0..255
        }
        var sum = 0.0
        var sumSq = 0.0
        var cnt = 0
        for (y in 1 until h - 1) {
            var i = y * n + 1
            for (x in 1 until n - 1) {
                val lap = (4 * g[i] - g[i - 1] - g[i + 1] - g[i - n] - g[i + n]).toDouble()
                sum += lap; sumSq += lap * lap; cnt++; i++
            }
        }
        if (cnt == 0) return 0.0
        val mean = sum / cnt
        return sumSq / cnt - mean * mean
    }

    /** Only embed confident, big-enough faces — tiny/low-score crops spawn bogus identities. */
    private fun faceQualityOk(d: Detection): Boolean {
        if (d.score < RECOG_MIN_SCORE) return false
        return min(d.x2 - d.x1, d.y2 - d.y1) >= MIN_FACE_FRAC
    }

    /**
     * Show the latest aligned face under the gallery button (two slots, most-recent first).
     * We own [aligned] here: it either becomes a slot bitmap or is recycled. The bitmap that
     * scrolls out of slot 1 (and any skipped one) is recycled so thumbnails don't accumulate.
     */
    private fun pushRecent(aligned: Bitmap, identityId: Int) {
        runOnUiThread {
            if (identityId == lastRecentId) { aligned.recycle(); return@runOnUiThread }
            val leaving = recentBmp1     // bitmap scrolling out of the second slot
            recentBmp1 = recentBmp0      // old newest shifts to the second slot
            recentBmp0 = aligned
            recent0.setImageBitmap(recentBmp0)
            recent0.visibility = View.VISIBLE
            recentBmp1?.let { recent1.setImageBitmap(it); recent1.visibility = View.VISIBLE }
            leaving?.recycle()           // safe: its ImageView was just reassigned above
            lastRecentId = identityId
        }
    }

    /** Armed + framing held: take a FULL-RES snapshot and crop the framed square out of it. */
    private fun maybeCaptureFraming(quad: FloatArray) {
        if (!framingArmed) return
        val ic = imageCapture ?: return
        val now = SystemClock.elapsedRealtime()
        if (now - lastFramingMs < FRAMING_COOLDOWN_MS) return
        if (!capturing.compareAndSet(false, true)) return // a shot is already in flight
        lastFramingMs = now
        val q = quad.copyOf()
        val mirror = lensFacing == CameraSelector.LENS_FACING_FRONT
        overlay.post { overlay.flashCapture() } // immediate shutter feedback
        ic.takePicture(captureExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val rot = image.imageInfo.rotationDegrees
                val bytes = try {
                    val b = image.planes[0].buffer
                    ByteArray(b.remaining()).also { b.get(it) }
                } finally {
                    image.close() // release the capture buffer before the heavy decode
                }
                try {
                    saveFramingShot(bytes, rot, q, mirror)
                } catch (e: Throwable) { // includes OutOfMemoryError from decoding a full-res JPEG
                    Log.e(TAG, "framing save failed", e)
                } finally {
                    capturing.set(false)
                }
            }

            override fun onError(e: ImageCaptureException) {
                Log.e(TAG, "framing capture failed", e)
                capturing.set(false)
            }
        })
    }

    /**
     * Decode the full-res JPEG, rotate it upright (so the frame-normalized quad maps directly — both
     * use cases are pinned to 4:3, same FOV/aspect as the analysis frame), crop the framed square,
     * and mirror it for the front camera. Runs on [captureExecutor], off the analysis thread.
     */
    private fun saveFramingShot(jpeg: ByteArray, rotationDegrees: Int, quad: FloatArray, mirror: Boolean) {
        var full = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size) ?: return
        if (rotationDegrees != 0) {
            val m = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            val r = Bitmap.createBitmap(full, 0, 0, full.width, full.height, m, true)
            if (r != full) full.recycle()
            full = r
        }
        val crop = squareCropFromQuad(full, quad)
        full.recycle()
        if (crop == null) return
        // Front camera preview is mirrored, so mirror the saved shot to match what the user saw.
        val shot = if (mirror) {
            val m = Matrix().apply { preScale(-1f, 1f) }
            Bitmap.createBitmap(crop, 0, 0, crop.width, crop.height, m, false).also { if (it != crop) crop.recycle() }
        } else {
            crop
        }
        framingGallery.save(shot, System.currentTimeMillis())
        shot.recycle()
    }

    /** Bounding box of the 4 frame points -> square + padding -> crop (off-image area padded black). */
    private fun squareCropFromQuad(upright: Bitmap, quad: FloatArray): Bitmap? {
        val w = upright.width
        val h = upright.height
        var minX = 1f; var maxX = 0f; var minY = 1f; var maxY = 0f
        for (i in 0 until 4) {
            val x = quad[i * 2]; val y = quad[i * 2 + 1]
            minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
        }
        val cx = (minX + maxX) / 2f * w
        val cy = (minY + maxY) / 2f * h
        val side = (max((maxX - minX) * w, (maxY - minY) * h) * (1f + 2f * FRAMING_PAD)).toInt()
        if (side < 8) return null
        val x0 = (cx - side / 2f).toInt()
        val y0 = (cy - side / 2f).toInt()
        val srcL = x0.coerceIn(0, w); val srcT = y0.coerceIn(0, h)
        val srcR = (x0 + side).coerceIn(0, w); val srcB = (y0 + side).coerceIn(0, h)
        if (srcR <= srcL || srcB <= srcT) return null
        val out = Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888)
        Canvas(out).apply {
            drawColor(Color.BLACK) // pad off-image area black so it's always a full square
            drawBitmap(
                upright,
                Rect(srcL, srcT, srcR, srcB),
                RectF((srcL - x0).toFloat(), (srcT - y0).toFloat(), (srcR - x0).toFloat(), (srcB - y0).toFloat()),
                null,
            )
        }
        return out
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        captureExecutor.shutdown()
        detector.close()
        handReg.close()
        faceReg.close()
        gestureClf.close()
        if (::embedder.isInitialized) embedder.close()
    }

    companion object {
        private const val TAG = "FaceHandDetector"
        private const val RECOG_THRESHOLD = 0.3f // cosine; verified: different-person never exceeds 0.21
        private const val ENROLL_MARGIN = 0.25f  // enroll a NEW id only if best match < this (between
                                                 // different-person max 0.21 and same-person min 0.26)
        private const val RECOG_MIN_SCORE = 0.6f // detector confidence gate before embedding
        private const val MIN_FACE_FRAC = 0.07f  // min face-box side as a fraction of the frame
        private const val SHARP_MIN = 50.0       // variance-of-Laplacian blur gate (raise to reject more blur)
        private const val FRAMING_COOLDOWN_MS = 5000L // wait this long between framing captures
        private const val FRAMING_PAD = 0.15f         // padding added around the framed bbox
    }
}
