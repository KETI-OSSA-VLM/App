package com.example.genionputtest.benchmark

data class TimedValue<T>(
    val value: T,
    val durationMs: Double
)

object LatencyTracker {
    inline fun <T> measure(block: () -> T): TimedValue<T> {
        val start = System.nanoTime()
        val value = block()
        val end = System.nanoTime()
        return TimedValue(
            value = value,
            durationMs = (end - start) / 1_000_000.0
        )
    }
}
