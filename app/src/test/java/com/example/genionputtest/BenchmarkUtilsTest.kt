package com.example.genionputtest

import org.junit.Assert.assertEquals
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
}
