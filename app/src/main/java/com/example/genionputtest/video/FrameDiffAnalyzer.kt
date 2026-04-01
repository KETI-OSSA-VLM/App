package com.example.genionputtest.video

import android.graphics.Bitmap
import android.graphics.Color

enum class Tier { ZERO, ONE, TWO }

data class DiffResult(val diffScore: Float, val tier: Tier)

class FrameDiffAnalyzer(
    val lowThreshold: Float = 0.08f,   // raised from 0.05 — compression noise → T0 instead of T1
    val highThreshold: Float = 0.20f
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

        // Compute MAD (Mean Absolute Difference)
        var sumDiff = 0f

        for (i in 0 until pixelCount) {
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
            sumDiff += kotlin.math.abs(currR - prevR)
            sumDiff += kotlin.math.abs(currG - prevG)
            sumDiff += kotlin.math.abs(currB - prevB)
        }

        // Normalize: MAD = sum / (pixels * 3 * 255.0)
        val diffScore = sumDiff / (pixelCount * 3 * 255.0f)

        // Decide tier based on thresholds
        val tier = when {
            diffScore < lowThreshold -> Tier.ZERO
            diffScore < highThreshold -> Tier.ONE
            else -> Tier.TWO
        }

        return DiffResult(diffScore, tier)
    }
}
