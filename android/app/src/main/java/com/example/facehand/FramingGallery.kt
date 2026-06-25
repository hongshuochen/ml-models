package com.example.facehand

import android.content.Context
import android.graphics.Bitmap
import java.io.File

/**
 * Stores the square "framing" snapshots (the region inside the two-hand frame) as JPEGs in
 * filesDir/framing/<wallclock-ms>.jpg. Much simpler than [FaceGallery] — no embeddings, just
 * files, listed newest-first. Process-wide singleton so the camera screen and the framing
 * gallery screen share one instance.
 */
class FramingGallery private constructor(context: Context) {

    private val dir = File(context.filesDir, "framing").apply { mkdirs() }

    /** Save a square crop. [stampMs] is only used for a unique filename. Returns the file, or null. */
    @Synchronized fun save(bmp: Bitmap, stampMs: Long): File? {
        val f = File(dir, "$stampMs.jpg")
        return try {
            f.outputStream().use { bmp.compress(Bitmap.CompressFormat.JPEG, 92, it) }
            f
        } catch (_: Exception) {
            null // storage full / IO error — skip rather than crash the camera loop
        }
    }

    /** Store the Gemini caption alongside a shot as a `<name>.txt` sidecar. */
    @Synchronized fun saveCaption(jpg: File, caption: String) {
        try { File(dir, jpg.nameWithoutExtension + ".txt").writeText(caption) } catch (_: Exception) {}
    }

    /** Read a shot's caption sidecar, or null if it has none yet. */
    @Synchronized fun caption(jpg: File): String? {
        val t = File(dir, jpg.nameWithoutExtension + ".txt")
        return if (t.exists()) try { t.readText() } catch (_: Exception) { null } else null
    }

    /** Saved shots, newest first. */
    @Synchronized fun all(): List<File> =
        (dir.listFiles { f -> f.extension == "jpg" } ?: emptyArray()).sortedByDescending { it.lastModified() }

    @Synchronized fun count(): Int = dir.listFiles { f -> f.extension == "jpg" }?.size ?: 0

    @Synchronized fun delete(f: File) {
        f.delete()
        File(dir, f.nameWithoutExtension + ".txt").delete() // its caption sidecar, if any
    }

    /** Remove every framing shot (and caption sidecar). */
    @Synchronized fun clear() {
        dir.listFiles()?.forEach { it.delete() }
    }

    companion object {
        @Volatile private var INSTANCE: FramingGallery? = null
        fun get(context: Context): FramingGallery =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: FramingGallery(context.applicationContext).also { INSTANCE = it }
            }
    }
}
