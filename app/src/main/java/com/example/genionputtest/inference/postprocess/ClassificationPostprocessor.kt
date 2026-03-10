package com.example.genionputtest.inference.postprocess

import com.example.genionputtest.Prediction
import java.nio.ByteBuffer

internal class ClassificationPostprocessor(
    private val topK: Int = 3
) {
    fun fromOutput(output: ByteBuffer): ClassificationOutput {
        val scores = output.duplicate().apply { rewind() }
        val predictions = mutableListOf<Prediction>()
        for (index in 0 until scores.remaining()) {
            predictions += Prediction(index = index, score = scores.get().toInt() and 0xFF)
        }
        return ClassificationOutput(
            predictions = predictions
                .sortedByDescending { it.score }
                .take(topK)
        )
    }
}
