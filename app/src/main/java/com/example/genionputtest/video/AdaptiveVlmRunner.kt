package com.example.genionputtest.video

import android.graphics.Bitmap
import com.example.genionputtest.llamacpp.LlamaCppEngine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

data class AdaptiveResult(
    val text: String,
    val tier: Tier,
    val inferenceMs: Double
)

data class TierStats(
    val count: Int,
    val totalMs: Double
) {
    val avgMs: Double get() = if (count == 0) 0.0 else totalMs / count
}

class AdaptiveVlmRunner(
    private val engine: LlamaCppEngine,
    private val prompt: String,
    private val diffAnalyzer: FrameDiffAnalyzer = FrameDiffAnalyzer()
) {
    private var previousFrame: Bitmap? = null
    private val mutex = Mutex()
    private val stats = Array(3) { TierStats(0, 0.0) }
    private var lastResultText: String? = null

    suspend fun processFrame(bitmap: Bitmap): AdaptiveResult {
        // Step 1: Compute diff and determine tier
        val prev = previousFrame
        val (diffScore, tier) = if (prev == null) {
            Pair(1.0f, Tier.TWO)
        } else {
            val diffResult = diffAnalyzer.analyze(bitmap, prev)
            Pair(diffResult.diffScore, diffResult.tier)
        }

        // Steps 2–5: Lock wraps all JNI calls
        return mutex.withLock {
            val startMs = System.currentTimeMillis()

            val (resultText, effectiveTier) = when (tier) {
                Tier.ZERO -> {
                    val cached = lastResultText
                    if (cached == null) {
                        // No cache yet — fall back to Tier.TWO
                        val response = engine.generate(bitmap, prompt)
                        previousFrame = bitmap
                        lastResultText = response.text
                        Pair(response.text, Tier.TWO)
                    } else {
                        Pair(cached, Tier.ZERO)
                    }
                }
                Tier.ONE -> {
                    val response = engine.generateOnly()
                    if (response.text.startsWith("ERROR:")) {
                        // Fall back to Tier.TWO
                        val fallback = engine.generate(bitmap, prompt)
                        previousFrame = bitmap
                        lastResultText = fallback.text
                        Pair(fallback.text, Tier.TWO)
                    } else {
                        lastResultText = response.text
                        Pair(response.text, Tier.ONE)
                    }
                }
                Tier.TWO -> {
                    val response = engine.generate(bitmap, prompt)
                    previousFrame = bitmap
                    lastResultText = response.text
                    Pair(response.text, Tier.TWO)
                }
            }

            val inferenceMs = (System.currentTimeMillis() - startMs).toDouble()

            // Update stats for the effective tier
            val idx = effectiveTier.ordinal
            stats[idx] = stats[idx].copy(
                count = stats[idx].count + 1,
                totalMs = stats[idx].totalMs + inferenceMs
            )

            AdaptiveResult(resultText, effectiveTier, inferenceMs)
        }
    }

    fun tierDistribution(): String {
        val total = stats.sumOf { it.count }
        if (total == 0) return "T0 0% / T1 0% / T2 0%"
        val pct = stats.map { (it.count * 100f / total).toInt() }
        return "T0 ${pct[0]}% / T1 ${pct[1]}% / T2 ${pct[2]}%"
    }

    suspend fun resetStats() {
        mutex.withLock {
            for (i in stats.indices) {
                stats[i] = TierStats(0, 0.0)
            }
            previousFrame = null
            lastResultText = null
        }
    }
}
