package com.example.genionputtest.llamacpp

import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class LlamaCppEngine(
    private val modelPath: String,
    private val mmprojPath: String,
    private val cacheDir: File,
    private val nThreads: Int = 6,
    private val maxNewTokens: Int = 30
) : AutoCloseable {

    private val bridge = LlamaCppBridge()
    private var loaded = false

    suspend fun initialize() {
        withContext(Dispatchers.IO) {
            val ok = bridge.loadModel(modelPath, mmprojPath, nThreads)
            if (!ok) error("LlamaCppEngine: failed to load model from $modelPath")
            loaded = true
        }
    }

    suspend fun generate(bitmap: Bitmap, prompt: String): LlamaCppResponse {
        check(loaded) { "LlamaCppEngine is not initialized." }
        return withContext(Dispatchers.Default) {
            val imageFile = withContext(Dispatchers.IO) { writeTempImage(bitmap) }
            val startNs = System.nanoTime()
            val raw = bridge.generate(imageFile.absolutePath, prompt, maxNewTokens)
            val elapsedMs = (System.nanoTime() - startNs) / 1_000_000.0
            val text = firstSentence(raw)
            LlamaCppResponse(text = text, inferenceMs = elapsedMs)
        }
    }

    override fun close() {
        if (loaded) {
            bridge.freeModel()
            loaded = false
        }
    }

    private fun writeTempImage(bitmap: Bitmap): File {
        val dir = File(cacheDir, "llama-input").apply { mkdirs() }
        val file = File(dir, "input.jpg")
        val resized = resizeToMax(bitmap, 384)
        FileOutputStream(file).use { resized.compress(Bitmap.CompressFormat.JPEG, 90, it) }
        return file
    }

    private fun firstSentence(text: String): String {
        val idx = text.indexOfFirst { it == '.' || it == '!' || it == '?' }
        return if (idx >= 0) text.substring(0, idx + 1).trim() else text.trim()
    }

    private fun resizeToMax(bitmap: Bitmap, maxSize: Int): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w <= maxSize && h <= maxSize) return bitmap
        val scale = maxSize.toFloat() / maxOf(w, h)
        return Bitmap.createScaledBitmap(bitmap, (w * scale).toInt(), (h * scale).toInt(), true)
    }
}

data class LlamaCppResponse(
    val text: String,
    val inferenceMs: Double
) {
    val latencySummary: String
        get() = "Inference: %.3f ms".format(inferenceMs)
}
