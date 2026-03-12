package com.example.genionputtest.inference.spec

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.tensorflow.lite.DataType

class ModelSpecTest {

    @Test
    fun mobileClip2S0Spec_exposesFloatEmbeddingContract() {
        assertEquals("mobileclip2_s0_image_encoder_float16.tflite", MobileClip2S0Spec.assetName)
        assertEquals(256, MobileClip2S0Spec.inputWidth)
        assertEquals(256, MobileClip2S0Spec.inputHeight)
        assertEquals(DataType.FLOAT32, MobileClip2S0Spec.inputDataType)
        assertEquals(OutputKind.EMBEDDING, MobileClip2S0Spec.outputKind)
        assertTrue(MobileClip2S0Spec.requiresNormalization)
    }
}