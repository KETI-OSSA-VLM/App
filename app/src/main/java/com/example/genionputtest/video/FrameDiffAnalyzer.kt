package com.example.genionputtest.video

import android.graphics.Bitmap
import android.graphics.Color

enum class Tier { ZERO, ONE, TWO }

data class DiffResult(val diffScore: Float, val tier: Tier, val hintText: String = "")

class FrameDiffAnalyzer(
    private val lowThreshold: Float = 0.01f,
    private val highThreshold: Float = 0.05f
) {
    fun analyze(current: Bitmap, previous: Bitmap): DiffResult {
        // Scale both bitmaps to 32×32
        val scaledCurrent = Bitmap.createScaledBitmap(current, 32, 32, false)
        val scaledPrevious = Bitmap.createScaledBitmap(previous, 32, 32, false)

        // Extract pixel arrays
        val pixelCount = 32 * 32
        val currentPixels = IntArray(pixelCount)
        val previousPixels = IntArray(pixelCount)

        scaledCurrent.getPixels(currentPixels, 0, 32, 0, 0, 32, 32)
        scaledPrevious.getPixels(previousPixels, 0, 32, 0, 0, 32, 32)

        scaledCurrent.recycle()
        scaledPrevious.recycle()

        // Compute MAD (Mean Absolute Difference) + per-quadrant sums in a single loop
        var sumDiff = 0f
        val quadrantSums = FloatArray(4)

        for (y in 0 until 32) {
            val yOffset = y * 32
            for (x in 0 until 32) {
                val i = yOffset + x

                val currPixel = currentPixels[i]
                val prevPixel = previousPixels[i]

                // Extract R, G, B channels
                val currR = Color.red(currPixel).toFloat()
                val currG = Color.green(currPixel).toFloat()
                val currB = Color.blue(currPixel).toFloat()

                val prevR = Color.red(prevPixel).toFloat()
                val prevG = Color.green(prevPixel).toFloat()
                val prevB = Color.blue(prevPixel).toFloat()

                // Sum absolute differences
                val pixelDiff = kotlin.math.abs(currR - prevR) +
                                kotlin.math.abs(currG - prevG) +
                                kotlin.math.abs(currB - prevB)

                sumDiff += pixelDiff

                // Quadrant mapping: 0=top-left, 1=top-right, 2=bot-left, 3=bot-right
                val q = (if (y >= 16) 2 else 0) + (if (x >= 16) 1 else 0)
                quadrantSums[q] += pixelDiff
            }
        }

        // Normalize: MAD = sum / (pixels * 3 * 255.0)
        val diffScore = sumDiff / (pixelCount * 3 * 255.0f)

        // Each quadrant is 16×16 = 256 pixels
        val quadrantMad = FloatArray(4) { q -> quadrantSums[q] / (256 * 3 * 255.0f) }

        // Decide tier based on thresholds
        val tier = when {
            diffScore < lowThreshold -> Tier.ZERO
            diffScore < highThreshold -> Tier.ONE
            else -> Tier.TWO
        }

        val hintText = buildHint(diffScore, quadrantMad, tier)

        return DiffResult(diffScore, tier, hintText)
    }

    internal fun buildHint(overallMad: Float, quadrantMad: FloatArray, tier: Tier): String {
        if (tier != Tier.ONE) return ""

        val intensity = if (overallMad < 0.025f) "slight" else "moderate"

        // Uniform check: if max - min < overallMad * 0.5 → overall change
        val maxQ = quadrantMad.maxOrNull()!!
        val minQ = quadrantMad.minOrNull()!!
        if (maxQ - minQ < overallMad * 0.5f) {
            return "[Hint: $intensity overall change detected]"
        }

        // Find quadrants above average, take top 2 by value
        val labels = listOf("top-left", "top-right", "bot-left", "bot-right")
        val average = quadrantMad.average().toFloat()

        val active = quadrantMad
            .mapIndexed { index, mad -> Pair(index, mad) }
            .filter { (_, mad) -> mad > average }
            .sortedByDescending { (_, mad) -> mad }
            .take(2)
            .map { (index, _) -> labels[index] }

        return if (active.isEmpty()) {
            "[Hint: $intensity overall change detected]"
        } else {
            "[Hint: $intensity movement detected in ${active.joinToString(" and ")}]"
        }
    }
}
