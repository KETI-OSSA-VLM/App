package com.example.genionputtest

import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.ByteBuffer

class ClassificationUtilsTest {

    @Test
    fun extractTopClasses_returnsHighestScoresInOrder() {
        val output = ByteBuffer.allocateDirect(6).apply {
            put(byteArrayOf(1, 100.toByte(), 7, 90.toByte(), 3, 80.toByte()))
            rewind()
        }

        val predictions = extractTopClasses(output = output, topK = 3)

        assertEquals(listOf(1, 3, 5), predictions.map { it.index })
        assertEquals(listOf(100, 90, 80), predictions.map { it.score })
    }

    @Test
    fun formatClassificationResults_formatsRankedLines() {
        val text = formatClassificationResults(
            listOf(
                Prediction(index = 42, score = 200),
                Prediction(index = 7, score = 128)
            )
        )

        assertEquals("#1 class 42 score=200\n#2 class 7 score=128", text)
    }

    @Test
    fun formatClassificationSummary_includesSourceAndLatency() {
        val text = formatClassificationSummary(
            sourceLabel = "selected image",
            inferenceMs = 3.25,
            labels = listOf("background", "one", "two", "forty-two", "seven"),
            predictions = listOf(
                Prediction(index = 3, score = 200),
                Prediction(index = 4, score = 128)
            )
        )

        assertEquals(
            "selected image\nInference: 3.250 ms\n#1 forty-two (class 3) score=200\n#2 seven (class 4) score=128",
            text
        )
    }
}
