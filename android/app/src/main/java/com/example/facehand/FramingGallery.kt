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

    /** Save a square crop. [stampMs] is only used for a unique filename. */
    @Synchronized fun save(bmp: Bitmap, stampMs: Long) {
        try {
            File(dir, "$stampMs.jpg").outputStream().use { bmp.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        } catch (_: Exception) {
            // storage full / IO error — skip rather than crash the camera loop
        }
    }

    /** Saved shots, newest first. */
    @Synchronized fun all(): List<File> =
        (dir.listFiles { f -> f.extension == "jpg" } ?: emptyArray()).sortedByDescending { it.lastModified() }

    @Synchronized fun count(): Int = dir.listFiles { f -> f.extension == "jpg" }?.size ?: 0

    @Synchronized fun delete(f: File) { f.delete() }

    companion object {
        @Volatile private var INSTANCE: FramingGallery? = null
        fun get(context: Context): FramingGallery =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: FramingGallery(context.applicationContext).also { INSTANCE = it }
            }
    }
}
