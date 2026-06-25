package com.example.facehand

import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * Captions a JPEG with the Gemini API (`generateContent`). Call it on a BACKGROUND thread — it
 * blocks on the network. Returns the caption text, or null on any error (including a blank API
 * key, so the rest of the app keeps working until you paste a key in).
 *
 * Paste your key into [API_KEY]. (Hardcoding is fine for a personal / research build; for a
 * shipped app move the key behind a backend.)
 */
object GeminiCaptioner {

    /** <-- paste your Gemini API key here. */
    const val API_KEY = ""

    /** "gemini-2.5-flash" (balanced) or "gemini-2.5-flash-lite" (lowest latency). */
    private const val MODEL = "gemini-2.5-flash"

    // For Traditional Chinese, change to e.g. "用一句自然的話描述這張照片裡有什麼，給看不見的人聽，直接說重點。"
    private const val PROMPT =
        "Describe what is in this photo in one short, natural sentence for a person who cannot " +
        "see it. Name the main objects/people and the scene. No preamble, just the sentence."

    val isConfigured: Boolean get() = API_KEY.isNotBlank()

    fun caption(jpeg: ByteArray): String? {
        if (API_KEY.isBlank()) return null
        val url = URL("https://generativelanguage.googleapis.com/v1beta/models/$MODEL:generateContent?key=$API_KEY")
        var conn: HttpURLConnection? = null
        try {
            val body = JSONObject().put(
                "contents",
                JSONArray().put(
                    JSONObject().put(
                        "parts",
                        JSONArray()
                            .put(JSONObject().put("text", PROMPT))
                            .put(
                                JSONObject().put(
                                    "inline_data",
                                    JSONObject()
                                        .put("mime_type", "image/jpeg")
                                        .put("data", Base64.encodeToString(jpeg, Base64.NO_WRAP)),
                                ),
                            ),
                    ),
                ),
            )
                // Disable 2.5 "thinking" (a big latency win for plain captioning) + cap the output.
                .put(
                    "generationConfig",
                    JSONObject()
                        .put("thinkingConfig", JSONObject().put("thinkingBudget", 0))
                        .put("maxOutputTokens", 60)
                        .put("temperature", 0.4),
                )
                .toString()

            conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                doOutput = true
                connectTimeout = 15_000
                readTimeout = 30_000
                setRequestProperty("Content-Type", "application/json")
            }
            conn.outputStream.use { it.write(body.toByteArray()) }

            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val resp = stream?.bufferedReader()?.use { it.readText() } ?: return null
            if (code !in 200..299) return null

            // candidates[0].content.parts[*].text
            val cands = JSONObject(resp).optJSONArray("candidates") ?: return null
            if (cands.length() == 0) return null
            val parts = cands.getJSONObject(0).optJSONObject("content")?.optJSONArray("parts") ?: return null
            val sb = StringBuilder()
            for (i in 0 until parts.length()) sb.append(parts.getJSONObject(i).optString("text", ""))
            return sb.toString().trim().ifBlank { null }
        } catch (e: Exception) {
            return null
        } finally {
            conn?.disconnect()
        }
    }
}
