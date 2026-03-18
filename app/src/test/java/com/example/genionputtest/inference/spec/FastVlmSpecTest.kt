package com.example.genionputtest.inference.spec

import org.junit.Assert.assertEquals
import org.junit.Test

class FastVlmSpecTest {

    @Test
    fun fastVlmSpec_defaultsToGemmaBaselinePackage() {
        assertEquals("Gemma 3n E2B IT Int4", FastVlmSpec.modelName)
        assertEquals("Gemma-3n-E2B-it-int4.litertlm", FastVlmSpec.assetName)
    }
}
