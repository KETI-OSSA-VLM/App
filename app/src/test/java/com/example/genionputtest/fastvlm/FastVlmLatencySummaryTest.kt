package com.example.genionputtest.fastvlm

import org.junit.Assert.assertEquals
import org.junit.Test

class FastVlmLatencySummaryTest {

    @Test
    fun latencySummary_omitsTokenCountsWhenBenchmarkInfoIsUnavailable() {
        assertEquals(
            "Inference: 12.345 ms",
            buildFastVlmLatencySummary(
                elapsedMs = 12.345,
                prefillTokenCount = null,
                decodeTokenCount = null
            )
        )
    }

    @Test
    fun latencySummary_includesTokenCountsWhenBenchmarkInfoExists() {
        assertEquals(
            "Inference: 12.345 ms | Tokens(prefill=7, decode=11)",
            buildFastVlmLatencySummary(
                elapsedMs = 12.345,
                prefillTokenCount = 7,
                decodeTokenCount = 11
            )
        )
    }
}
