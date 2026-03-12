package com.example.genionputtest

import com.example.genionputtest.benchmark.LatencyBreakdown
import com.example.genionputtest.inference.postprocess.EmbeddingOutput
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer

class BenchmarkUtilsTest {

    @Test
    fun runInferenceIterations_rewindsBuffersBeforeEachRun() {
        val input = ByteBuffer.allocateDirect(4)
        val output = ByteBuffer.allocateDirect(4)
        var calls = 0

        runInferenceIterations(iterations = 3, input = input, output = output) { inBuf, outBuf ->
            assertEquals(0, inBuf.position())
            assertEquals(0, outBuf.position())

            while (inBuf.hasRemaining()) {
                inBuf.get()
            }
            while (outBuf.hasRemaining()) {
                outBuf.put(1)
            }
            calls++
        }

        assertEquals(3, calls)
    }

    @Test
    fun formatEmbeddingSummary_includesDimensionNormPreviewAndLatency() {
        val summary = formatEmbeddingSummary(
            sourceLabel = "selected image",
            embedding = EmbeddingOutput(
                values = floatArrayOf(1f, 2f, 3f),
                previewValues = floatArrayOf(1f, 2f, 3f),
                dimension = 3,
                l2Norm = 3.7417f
            ),
            latencyBreakdown = LatencyBreakdown(
                preprocessMs = 1.0,
                inferenceMs = 2.0,
                postprocessMs = 3.0
            )
        )

        assertTrue(summary.contains("selected image"))
        assertTrue(summary.contains("Dimension: 3"))
        assertTrue(summary.contains("L2 norm: 3.742"))
        assertTrue(summary.contains("Preview: [1.000, 2.000, 3.000]"))
        assertTrue(summary.contains("Preprocess: 1.000 ms"))
        assertTrue(summary.contains("Total: 6.000 ms"))
    }
}