package com.example.genionputtest.fastvlm

import org.junit.Assert.assertEquals
import org.junit.Test

class FastVlmConversationConfigTest {

    @Test
    fun conversationConfig_usesExplicitSamplerSettings() {
        val config = buildFastVlmConversationConfig()
        val samplerConfig = config.samplerConfig!!

        assertEquals(1, samplerConfig.topK)
        assertEquals(1.0, samplerConfig.topP, 0.0)
        assertEquals(0.0, samplerConfig.temperature, 0.0)
        assertEquals(7, samplerConfig.seed)
    }
}
