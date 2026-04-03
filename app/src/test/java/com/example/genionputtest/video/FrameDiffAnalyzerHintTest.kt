package com.example.genionputtest.video

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FrameDiffAnalyzerHintTest {
    private val analyzer = FrameDiffAnalyzer()

    @Test
    fun `buildHint returns empty for T0 tier`() {
        val result = analyzer.buildHint(0.005f, FloatArray(4) { 0.005f }, Tier.ZERO)
        assertEquals("", result)
    }

    @Test
    fun `buildHint returns empty for T2 tier`() {
        val result = analyzer.buildHint(0.08f, FloatArray(4) { 0.08f }, Tier.TWO)
        assertEquals("", result)
    }

    @Test
    fun `buildHint returns overall when change is uniform`() {
        val quadMad = floatArrayOf(0.015f, 0.016f, 0.014f, 0.015f)
        val overall = 0.015f
        val result = analyzer.buildHint(overall, quadMad, Tier.ONE)
        assertTrue("Expected 'overall' in: $result", result.contains("overall"))
    }

    @Test
    fun `buildHint returns top-right when top-right quadrant dominates`() {
        val quadMad = floatArrayOf(0.010f, 0.040f, 0.010f, 0.010f)
        val overall = quadMad.average().toFloat()
        val result = analyzer.buildHint(overall, quadMad, Tier.ONE)
        assertTrue("Expected 'top-right' in: $result", result.contains("top-right"))
    }

    @Test
    fun `buildHint uses slight when overall MAD is below 0_025`() {
        val quadMad = floatArrayOf(0.010f, 0.030f, 0.010f, 0.010f)
        val overall = 0.015f
        val result = analyzer.buildHint(overall, quadMad, Tier.ONE)
        assertTrue("Expected 'slight' in: $result", result.contains("slight"))
    }

    @Test
    fun `buildHint uses moderate when overall MAD is 0_025 or above`() {
        val quadMad = floatArrayOf(0.010f, 0.060f, 0.010f, 0.010f)
        val overall = 0.030f
        val result = analyzer.buildHint(overall, quadMad, Tier.ONE)
        assertTrue("Expected 'moderate' in: $result", result.contains("moderate"))
    }

    @Test
    fun `buildHint lists at most two regions`() {
        val quadMad = floatArrayOf(0.040f, 0.040f, 0.040f, 0.010f)
        val overall = quadMad.average().toFloat()
        val result = analyzer.buildHint(overall, quadMad, Tier.ONE)
        val andCount = result.split(" and ").size - 1
        assertTrue("Expected at most one 'and' in: $result", andCount <= 1)
    }
}
