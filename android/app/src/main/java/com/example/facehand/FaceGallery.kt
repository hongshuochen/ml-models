package com.example.facehand

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Persistent face gallery: maps 512-d face embeddings to identities. Embeddings are stored
 * L2-normalized, so matching is a plain dot product (= cosine similarity). Each identity keeps
 * a running-mean template (more robust than a single frame), a sample count, a display label,
 * and an aligned thumbnail saved to disk so the Gallery screen can show the face.
 *
 * This is what gives recognition across appearances (re-ID) and across app restarts — the
 * tracker handles short-term continuity, the gallery handles long-term identity.
 *
 * Storage: filesDir/gallery/index.json (templates + labels) and filesDir/gallery/<id>.jpg.
 */
class FaceGallery private constructor(context: Context) {

    class Person(
        val id: Int,
        var label: String,
        val template: FloatArray, // 512-d, L2-normalized
        var count: Int,
        val thumbPath: String,
    )

    private val dir = File(context.filesDir, "gallery").apply { mkdirs() }
    private val indexFile = File(dir, "index.json")
    private val people = ArrayList<Person>()
    private var nextId = 1

    init { load() }

    @Synchronized fun count() = people.size
    @Synchronized fun all(): List<Person> = people.toList()
    @Synchronized fun person(id: Int): Person? = people.firstOrNull { it.id == id }

    /** Best cosine match over all identities (caller compares to a threshold), or null if empty. */
    @Synchronized fun match(emb: FloatArray): Pair<Person, Float>? {
        var best: Person? = null
        var bestSim = -1f
        for (p in people) {
            val s = dot(p.template, emb)
            if (s > bestSim) { bestSim = s; best = p }
        }
        return best?.let { it to bestSim }
    }

    /** Add a new identity from one embedding + its aligned face crop. Returns the new Person. */
    @Synchronized fun enroll(emb: FloatArray, aligned: Bitmap): Person {
        val id = nextId++
        val thumb = File(dir, "$id.jpg")
        saveJpg(aligned, thumb)
        val p = Person(id, "Person $id", l2(emb.copyOf()), 1, thumb.absolutePath)
        people.add(p)
        save()
        return p
    }

    /** Fold another observation of a known identity into its running-mean template. */
    @Synchronized fun reinforce(p: Person, emb: FloatArray, aligned: Bitmap?) {
        val n = p.count.toFloat()
        for (i in p.template.indices) p.template[i] = (p.template[i] * n + emb[i]) / (n + 1f)
        l2(p.template)
        p.count++
        if (aligned != null && p.count <= THUMB_REFRESH_UNTIL) saveJpg(aligned, File(p.thumbPath))
        save()
    }

    @Synchronized fun rename(p: Person, name: String) { p.label = name; save() }

    @Synchronized fun delete(p: Person) {
        people.remove(p)
        File(p.thumbPath).delete()
        save()
    }

    /** Remove every identity (and its thumbnail). */
    @Synchronized fun clear() {
        people.forEach { File(it.thumbPath).delete() }
        people.clear()
        save()
    }

    // --- persistence ---
    private fun load() {
        if (!indexFile.exists()) return
        try {
            val arr = JSONArray(indexFile.readText())
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                val t = o.getJSONArray("template")
                val emb = FloatArray(t.length()) { t.getDouble(it).toFloat() }
                val id = o.getInt("id")
                people.add(Person(id, o.getString("label"), emb, o.getInt("count"),
                    File(dir, "$id.jpg").absolutePath))
                nextId = max(nextId, id + 1)
            }
        } catch (_: Exception) {
            // Corrupt index -> start fresh rather than crash.
            people.clear()
        }
    }

    private fun save() {
        val arr = JSONArray()
        for (p in people) {
            val t = JSONArray()
            for (v in p.template) t.put(v.toDouble())
            arr.put(JSONObject().put("id", p.id).put("label", p.label).put("count", p.count).put("template", t))
        }
        indexFile.writeText(arr.toString())
    }

    private fun saveJpg(bmp: Bitmap, f: File) {
        try {
            f.outputStream().use { bmp.compress(Bitmap.CompressFormat.JPEG, 90, it) }
        } catch (_: Exception) {
            // Storage full / read-only — skip the thumbnail rather than crash the camera loop.
        }
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        val n = min(a.size, b.size)
        for (i in 0 until n) s += a[i] * b[i]
        return s
    }

    private fun l2(v: FloatArray): FloatArray {
        var s = 0f
        for (x in v) s += x * x
        val inv = 1f / (sqrt(s) + 1e-9f)
        for (i in v.indices) v[i] *= inv
        return v
    }

    companion object {
        private const val THUMB_REFRESH_UNTIL = 6 // refresh the saved thumb for the first few obs

        // Process-wide single instance so the live camera view and the Gallery screen share
        // the same in-memory gallery (a rename/delete shows up immediately in both).
        @Volatile private var INSTANCE: FaceGallery? = null

        fun get(context: Context): FaceGallery =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: FaceGallery(context.applicationContext).also { INSTANCE = it }
            }
    }
}
