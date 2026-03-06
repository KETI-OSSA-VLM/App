package com.example.genionputtest

import org.junit.Assert.assertEquals
import org.junit.Test

class BenchmarkStatusFormatterTest {

    @Test
    fun formatBenchmarkStatus_includesTagAndAverageMs() {
        val message = formatBenchmarkStatus(tag = "CPU", avgMs = 12.3456, runs = 50)

        assertEquals("[CPU] avg=12.346 ms (runs=50)", message)
    }
}
