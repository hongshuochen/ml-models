package com.example.facehand

import kotlin.math.max
import kotlin.math.min

/**
 * Lightweight per-face tracker so each face is recognized ONCE per appearance instead of
 * embedded every frame. Association is plain IoU with a size gate (boxes whose areas differ
 * by more than [SIZE_RATIO]x never match), which is cheap and good enough for a phone demo.
 *
 * Identity itself comes from the [FaceGallery] (an embedding match) — the tracker only carries
 * it forward across frames and decides WHEN a track needs (re-)embedding: when it's new, when
 * it reappears after being lost (a fresh track), or every [RECONFIRM_FRAMES] frames as a
 * re-confirmation against silent ID swaps.
 */
class FaceTracker {

    /** One tracked face. Coords are normalized [0,1] to the upright frame. */
    class Track(var id: Int, var x1: Float, var y1: Float, var x2: Float, var y2: Float) {
        var identityId: Int = -1          // gallery id; -1 = not recognized yet
        var sim: Float = 0f               // cosine of the last gallery match
        var miss: Int = 0                 // consecutive frames not matched to a detection
        var lastEmbedFrame: Int = -100000 // frame index of the last embedding
        var seen: Int = 0                 // frames matched so far

        fun area() = max(0f, x2 - x1) * max(0f, y2 - y1)
        fun update(b: Detection) { x1 = b.x1; y1 = b.y1; x2 = b.x2; y2 = b.y2; miss = 0; seen++ }

        /** Intersection-over-union with a detection box (both normalized to the frame). */
        fun iou(d: Detection): Float {
            val ix1 = max(x1, d.x1); val iy1 = max(y1, d.y1)
            val ix2 = min(x2, d.x2); val iy2 = min(y2, d.y2)
            val inter = max(0f, ix2 - ix1) * max(0f, iy2 - iy1)
            val uni = area() + max(0f, d.x2 - d.x1) * max(0f, d.y2 - d.y1) - inter
            return if (uni <= 0f) 0f else inter / uni
        }
    }

    private val tracks = ArrayList<Track>()
    private var nextId = 1
    private var frame = 0

    val active: List<Track> get() = tracks
    val frameIndex: Int get() = frame

    /**
     * Advance one frame: match the given face boxes to existing tracks (IoU + size gate),
     * spawn tracks for new faces, and age out tracks unseen for [MAX_MISS] frames.
     */
    fun update(faces: List<Detection>) {
        frame++
        val freeTracks = tracks.toMutableList()
        val freeDets = faces.toMutableList()

        // Greedy: repeatedly bind the highest-IoU (track, detection) pair that passes the gates.
        while (true) {
            var bestIoU = IOU_THRESHOLD
            var bt: Track? = null
            var bd: Detection? = null
            for (t in freeTracks) for (d in freeDets) {
                if (!sizeCompatible(t, d)) continue
                val iou = t.iou(d)
                if (iou > bestIoU) { bestIoU = iou; bt = t; bd = d }
            }
            if (bt == null || bd == null) break
            bt.update(bd)
            freeTracks.remove(bt)
            freeDets.remove(bd)
        }

        // Unmatched existing tracks: age them; drop the stale ones.
        for (t in freeTracks) t.miss++
        tracks.removeAll { it.miss > MAX_MISS }

        // Unmatched detections: brand-new faces -> new tracks.
        for (d in freeDets) tracks.add(Track(nextId++, d.x1, d.y1, d.x2, d.y2).also { it.seen = 1 })
    }

    /** A track needs (re-)embedding when it's unrecognized, or its last embed has gone stale. */
    fun needsEmbed(t: Track): Boolean =
        t.identityId < 0 || (frame - t.lastEmbedFrame) >= RECONFIRM_FRAMES

    fun markEmbedded(t: Track) { t.lastEmbedFrame = frame }

    private fun sizeCompatible(t: Track, d: Detection): Boolean {
        val at = t.area()
        val ad = max(0f, d.x2 - d.x1) * max(0f, d.y2 - d.y1)
        if (at <= 0f || ad <= 0f) return false
        val r = at / ad
        return r in (1f / SIZE_RATIO)..SIZE_RATIO
    }

    companion object {
        private const val IOU_THRESHOLD = 0.3f   // min overlap to bind a detection to a track
        private const val SIZE_RATIO = 2.0f      // areas may differ by at most 2x to match
        private const val MAX_MISS = 8           // drop a track after this many unseen frames
        private const val RECONFIRM_FRAMES = 90  // re-embed a recognized track every ~3s (@30fps)
    }
}
