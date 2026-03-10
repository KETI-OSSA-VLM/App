package com.example.genionputtest.benchmark

data class LatencyBreakdown(
    val preprocessMs: Double,
    val inferenceMs: Double,
    val postprocessMs: Double
) {
    val totalMs: Double
        get() = preprocessMs + inferenceMs + postprocessMs
}
