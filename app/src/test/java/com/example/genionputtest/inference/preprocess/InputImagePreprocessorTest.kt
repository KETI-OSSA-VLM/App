package com.example.genionputtest.inference.preprocess

import org.junit.Assert.assertEquals
import org.junit.Test
import org.tensorflow.lite.DataType

class InputImagePreprocessorTest {

    @Test
    fun expectedInputByteSize_returnsRgbByteCountForUint8() {
        assertEquals(224 * 224 * 3, expectedInputByteSize(intArrayOf(1, 224, 224, 3), DataType.UINT8))
    }

    @Test
    fun expectedInputByteSize_returnsFloatByteCountForFloat32() {
        assertEquals(224 * 224 * 3 * 4, expectedInputByteSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32))
    }
}
