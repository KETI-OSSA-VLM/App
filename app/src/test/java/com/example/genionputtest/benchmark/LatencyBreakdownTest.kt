package com.example.genionputtest.benchmark

import org.junit.Assert.assertEquals
import org.junit.Test

class LatencyBreakdownTest {

    @Test
    fun totalMs_sumsAllStages() {
        val breakdown = LatencyBreakdown(
            preprocessMs = 1.25,
            inferenceMs = 2.5,
            postprocessMs = 0.75
        )

        assertEquals(4.5, breakdown.totalMs, 0.0001)
    }
}
