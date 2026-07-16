package com.example.golfrec

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.View
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
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseLandmark
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Golf swing auto-recorder. Rear camera watches a friend; ML Kit pose + our club_head detector spot
 * the ADDRESS posture (about to swing) and prompt "record?". On yes, CameraX records the swing and
 * saves it. (Third-person subject → uses the PUBLIC-trained club model, not the egocentric one.)
 */
class RecActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var statusText: TextView
    private lateinit var promptPanel: View
    private lateinit var recDot: View

    private var club: ClubDetector? = null                 // lazy-built on the analysis thread
    private val address = AddressDetector()
    private var tts: TextToSpeech? = null
    @Volatile private var ttsReady = false
    private var lastSpokenAt = 0L                           // hard TTS cooldown clock
    private val poseDetector = PoseDetection.getClient(
        PoseDetectorOptions.Builder().setDetectorMode(PoseDetectorOptions.STREAM_MODE).build())

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private var cameraProvider: ProcessCameraProvider? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    @Volatile private var busy = false                     // a frame is mid-analysis
    @Volatile private var prompting = false
    @Volatile private var recordingActive = false

    private val perms = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { g ->
        if (g[Manifest.permission.CAMERA] == true) startCamera()
        else { Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show(); finish() }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_rec)
        previewView = findViewById(R.id.previewView)
        overlay = findViewById(R.id.overlay)
        statusText = findViewById(R.id.statusText)
        promptPanel = findViewById(R.id.promptPanel)
        recDot = findViewById(R.id.recDot)
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val lang = tts?.setLanguage(Locale.US) ?: TextToSpeech.LANG_NOT_SUPPORTED
                ttsReady = lang != TextToSpeech.LANG_MISSING_DATA && lang != TextToSpeech.LANG_NOT_SUPPORTED
                Log.i(TAG, "TTS init ok, en-US lang=$lang ready=$ttsReady")
            } else Log.w(TAG, "TTS init failed status=$status")
        }
        findViewById<Button>(R.id.recordButton).setOnClickListener { hidePrompt(); startRecording() }
        findViewById<Button>(R.id.skipButton).setOnClickListener { hidePrompt(); address.rearm() }

        val need = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        if (need.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) startCamera()
        else perms.launch(need)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({ cameraProvider = future.get(); bind() }, ContextCompat.getMainExecutor(this))
    }

    private fun bind() {
        val provider = cameraProvider ?: return
        val selector = CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()
        val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build().also { it.setAnalyzer(analysisExecutor, ::analyze) }
        val recorder = Recorder.Builder().setQualitySelector(
            androidx.camera.video.QualitySelector.from(androidx.camera.video.Quality.HD)).build()
        videoCapture = VideoCapture.withOutput(recorder)
        try {
            provider.unbindAll()
            provider.bindToLifecycle(this, selector, preview, analysis, videoCapture)
        } catch (e: Exception) {
            Log.e(TAG, "bind failed (preview+analysis+video may be unsupported); retry without analysis-during-record", e)
            Toast.makeText(this, "Camera bind failed", Toast.LENGTH_LONG).show()
        }
    }

    /** Analysis thread: club detector (sync) + ML Kit pose (async) -> AddressDetector. */
    private fun analyze(image: ImageProxy) {
        if (busy || prompting || recordingActive) { image.close(); return }   // skip while prompting/recording
        busy = true
        val upright = image.toUprightBitmap()
        image.close()
        val t = System.nanoTime() / 1e9
        val det = club ?: ClubDetector(this).also { club = it }
        val scaled = Bitmap.createScaledBitmap(upright, ClubDetector.INPUT, ClubDetector.INPUT, true)
        val allDets = det.detect(scaled)            // ball + club_head (drawn, and used as golf evidence)
        if (scaled != upright) scaled.recycle()
        val w = upright.width; val h = upright.height

        poseDetector.process(InputImage.fromBitmap(upright, 0))
            .addOnSuccessListener { pose ->
                fun pt(id: Int): Pt? = pose.getPoseLandmark(id)?.takeIf { it.inFrameLikelihood > 0.5f }
                    ?.let { Pt(it.position.x / w, it.position.y / h) }
                val lw = pt(PoseLandmark.LEFT_WRIST); val rw = pt(PoseLandmark.RIGHT_WRIST)
                // golf evidence: a club_head OR ball sitting BELOW the hands — on the ground at the
                // golfer's feet where they rest at address (NOT up near the grip). Either one counts.
                val hands = if (lw != null && rw != null) Pt((lw.x + rw.x) / 2, (lw.y + rw.y) / 2) else (lw ?: rw)
                if (hands != null && allDets.any { (it.y1 + it.y2) / 2 > hands.y }) address.noteClubOrBallLow(t)
                val fired = address.update(t, lw, rw,
                    pt(PoseLandmark.LEFT_ELBOW), pt(PoseLandmark.RIGHT_ELBOW),
                    pt(PoseLandmark.LEFT_SHOULDER), pt(PoseLandmark.RIGHT_SHOULDER),
                    pt(PoseLandmark.LEFT_HIP), pt(PoseLandmark.RIGHT_HIP))
                // draw the ML Kit body skeleton + all golf detections (ball + club_head)
                val posePts = HashMap<Int, android.graphics.PointF>()
                for (lm in pose.allPoseLandmarks) {
                    if (lm.inFrameLikelihood > 0.5f)
                        posePts[lm.landmarkType] = android.graphics.PointF(lm.position.x / w, lm.position.y / h)
                }
                overlay.setResults(allDets, posePts, w, h, false, 0f)
                statusText.text = when (address.state) {
                    AddressDetector.State.SEARCHING -> "Searching…"
                    AddressDetector.State.HUMAN -> "Human detected…"
                    AddressDetector.State.POSTURE -> "Posture detected…"
                    AddressDetector.State.PROMPT -> "Ready to hit!"
                }
                if (fired && !recordingActive) showPrompt()
                busy = false
            }
            .addOnFailureListener { busy = false }
    }

    private fun showPrompt() {
        prompting = true; promptPanel.visibility = View.VISIBLE
        speak("Ready to hit, record?")
    }
    private fun hidePrompt() { prompting = false; promptPanel.visibility = View.GONE }

    /** Speak once. The fire event is already once-per-address; the cooldown is belt-and-braces. */
    private fun speak(text: String) {
        val now = android.os.SystemClock.elapsedRealtime()
        if (!ttsReady || now - lastSpokenAt < TTS_COOLDOWN_MS) return
        lastSpokenAt = now
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "golfrec")
    }

    private fun startRecording() {
        val vc = videoCapture ?: return
        if (recordingActive) return
        val name = "GolfSwing_" + System.currentTimeMillis()
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/GolfRec")
        }
        val opts = MediaStoreOutputOptions.Builder(contentResolver, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            .setContentValues(values).build()
        var rec = vc.output.prepareRecording(this, opts)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)
            rec = rec.withAudioEnabled()
        recording = rec.start(ContextCompat.getMainExecutor(this)) { ev ->
            when (ev) {
                is VideoRecordEvent.Start -> { recordingActive = true; recDot.visibility = View.VISIBLE }
                is VideoRecordEvent.Finalize -> {
                    recordingActive = false; recDot.visibility = View.GONE; address.rearm()
                    Toast.makeText(this, if (ev.hasError()) "Recording error" else "Saved to Movies/GolfRec", Toast.LENGTH_SHORT).show()
                }
            }
        }
        main.postDelayed({ if (recordingActive) stopRecording() }, MAX_REC_MS)  // auto-stop
    }

    private fun stopRecording() { recording?.stop(); recording = null }

    /** ImageProxy (RGBA_8888) → Bitmap rotated upright per the sensor rotation. */
    private fun ImageProxy.toUprightBitmap(): Bitmap {
        val bitmap = toBitmap()
        val r = imageInfo.rotationDegrees
        if (r == 0) return bitmap
        val m = Matrix().apply { postRotate(r.toFloat()) }
        val rot = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
        if (rot != bitmap) bitmap.recycle()
        return rot
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        poseDetector.close()
        club?.close()
        tts?.shutdown()
    }

    companion object {
        private const val TAG = "RecActivity"
        private const val MAX_REC_MS = 10_000L   // auto-stop a swing clip after this (no manual Stop button)
        private const val TTS_COOLDOWN_MS = 4_000L   // min gap between spoken cues
    }
}
