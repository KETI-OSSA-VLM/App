package com.example.genionputtest.fastvlm

import org.junit.Assert.assertTrue
import org.junit.Test

class FastVlmPromptDefaultsTest {

    @Test
    fun defaultPrompt_matchesCctvUseCase() {
        assertTrue(defaultFastVlmPrompt().contains("CCTV", ignoreCase = true))
    }
}
