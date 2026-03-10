package com.example.genionputtest.inference.postprocess

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
import kotlin.math.sqrt

class EmbeddingPostprocessor(
    private val previewValueCount: Int = 8
) {
    fun fromOutput(output: ByteBuffer): EmbeddingOutput {
        val floatBuffer = output.duplicate()
            .order(ByteOrder.nativeOrder())
            .apply { rewind() }
            .asFloatBuffer()
        val values = FloatArray(floatBuffer.remaining())
        floatBuffer.get(values)
        val squaredMagnitude = values.fold(0f) { acc, value -> acc + (value * value) }
        return EmbeddingOutput(
            values = values,
            previewValues = values.copyOf(min(previewValueCount, values.size)),
            dimension = values.size,
            l2Norm = sqrt(squaredMagnitude)
        )
    }
}
