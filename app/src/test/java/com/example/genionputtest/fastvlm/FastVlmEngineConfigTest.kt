package com.example.genionputtest.fastvlm

import com.google.ai.edge.litertlm.Backend
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class FastVlmEngineConfigTest {

    @Test
    fun buildFastVlmEngineConfig_setsGemmaCompatibleBackendsAndPaths() {
        val config = buildFastVlmEngineConfig(
            modelPath = "/models/FastVLM-0.5B.litertlm",
            cacheDirPath = "/cache"
        )

        assertEquals("/models/FastVLM-0.5B.litertlm", config.modelPath)
        assertNotNull(config.backend)
        assertNotNull(config.visionBackend)
        assertEquals("CPU", config.backend!!.name)
        assertEquals("GPU", config.visionBackend!!.name)
        assertEquals(null, config.audioBackend)
        assertEquals("/cache", config.cacheDir)
        assertEquals(512, config.maxNumTokens)
    }
}
