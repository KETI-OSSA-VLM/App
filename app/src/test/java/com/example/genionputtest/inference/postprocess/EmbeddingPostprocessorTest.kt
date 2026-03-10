package com.example.genionputtest.inference.postprocess

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

class EmbeddingPostprocessorTest {

    @Test
    fun fromOutput_readsVectorAndComputesNorm() {
        val output = ByteBuffer.allocateDirect(4 * 4)
            .order(ByteOrder.nativeOrder())
            .apply {
                putFloat(1.0f)
                putFloat(2.0f)
                putFloat(3.0f)
                putFloat(4.0f)
                rewind()
            }

        val embedding = EmbeddingPostprocessor(previewValueCount = 3).fromOutput(output)

        assertEquals(4, embedding.dimension)
        assertEquals(sqrt(30.0).toFloat(), embedding.l2Norm, 0.0001f)
        assertArrayEquals(floatArrayOf(1.0f, 2.0f, 3.0f), embedding.previewValues, 0.0001f)
    }
}
