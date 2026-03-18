package com.example.genionputtest

import com.example.genionputtest.inference.spec.OutputKind
import org.junit.Assert.assertEquals
import org.junit.Test

class FastVlmUiModeTest {

    @Test
    fun promptOnlyButtonVisibility_isEnabledOnlyForTextResponseModels() {
        assertEquals(true, shouldShowPromptOnlyButton(OutputKind.TEXT_RESPONSE))
        assertEquals(false, shouldShowPromptOnlyButton(OutputKind.EMBEDDING))
        assertEquals(false, shouldShowPromptOnlyButton(OutputKind.CLASSIFICATION))
    }
}
