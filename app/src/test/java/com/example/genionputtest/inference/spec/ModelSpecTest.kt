package com.example.genionputtest.inference.spec

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import org.tensorflow.lite.DataType

class ModelSpecTest {

    @Test
    fun mobileNetSpec_exposesQuantizedClassificationContract() {
        assertEquals("mobilenet_v1_1.0_224_quant.tflite", MobileNetSpec.assetName)
        assertEquals(224, MobileNetSpec.inputWidth)
        assertEquals(224, MobileNetSpec.inputHeight)
        assertEquals(DataType.UINT8, MobileNetSpec.inputDataType)
        assertEquals(OutputKind.CLASSIFICATION, MobileNetSpec.outputKind)
        assertFalse(MobileNetSpec.requiresNormalization)
    }
}
