package com.example.caption

import org.json.JSONObject
import java.io.File

/**
 * Minimal BART tokenizer for Florence-2 — no native dependency.
 *
 * We only need two things, both verified bit-exact against HF `tokenizers`:
 *   - the (fixed) task prompts, pre-tokenized to ids and hardcoded here;
 *   - decoding generated ids back to text via the GPT-2 byte-level scheme + `vocab.json`.
 * That avoids shipping a native tokenizer (DJL's isn't built for Android ABIs).
 */
class BartTokenizer(vocabJson: File) {

    enum class Task { CAPTION, DETAILED, MORE_DETAILED }

    private val idToToken: Array<String?>
    private val unicodeToByte = HashMap<Char, Int>()

    init {
        // GPT-2 bytes<->unicode table (identical to the verified Python reference).
        val bs = ArrayList<Int>()
        for (b in 33..126) bs.add(b)
        for (b in 161..172) bs.add(b)
        for (b in 174..255) bs.add(b)
        val cs = ArrayList(bs)
        var n = 0
        for (b in 0..255) if (b !in bs) { bs.add(b); cs.add(256 + n); n++ }
        for (i in bs.indices) unicodeToByte[cs[i].toChar()] = bs[i]

        val obj = JSONObject(vocabJson.readText())
        val pairs = ArrayList<Pair<String, Int>>(obj.length())
        var maxId = 0
        val it = obj.keys()
        while (it.hasNext()) {
            val k = it.next()
            val v = obj.getInt(k)
            pairs.add(k to v)
            if (v > maxId) maxId = v
        }
        idToToken = arrayOfNulls(maxId + 1)
        for ((k, v) in pairs) idToToken[v] = k
    }

    /** Pre-tokenized task prompts (include BOS=0 … EOS=2). Verified via HF tokenizers. */
    fun promptIds(task: Task): LongArray = when (task) {
        Task.CAPTION -> longArrayOf(0, 2264, 473, 5, 2274, 6190, 116, 2)
        Task.DETAILED -> longArrayOf(0, 47066, 21700, 11, 4617, 99, 16, 2343, 11, 5, 2274, 4, 2)
        Task.MORE_DETAILED -> longArrayOf(0, 47066, 21700, 19, 10, 17818, 99, 16, 2343, 11, 5, 2274, 4, 2)
    }

    /** Decode generated ids -> text (skips BOS/PAD/EOS = 0/1/2). */
    fun decode(ids: List<Int>): String {
        val sb = StringBuilder()
        for (id in ids) {
            if (id == 0 || id == 1 || id == 2) continue
            val t = if (id in idToToken.indices) idToToken[id] else null
            if (t != null) sb.append(t)
        }
        val bytes = ArrayList<Byte>(sb.length)
        for (c in sb) unicodeToByte[c]?.let { bytes.add(it.toByte()) }
        return String(bytes.toByteArray(), Charsets.UTF_8).trim()
    }
}
