package com.example.genionputtest

import com.example.genionputtest.inference.spec.OutputKind
import org.junit.Assert.assertEquals
import org.junit.Test

class ModelCatalogTest {

    @Test
    fun availableModelSpecs_returnsPreparedModelsInMenuOrder() {
        val specs = availableModelSpecs()

        assertEquals(
            listOf("Gemma 3n E2B IT Int4", "MobileCLIP2-S0 Image Encoder", "MobileNet v1 Quantized"),
            specs.map { it.modelName }
        )
        assertEquals(
            listOf(OutputKind.TEXT_RESPONSE, OutputKind.EMBEDDING, OutputKind.CLASSIFICATION),
            specs.map { it.outputKind }
        )
    }

    @Test
    fun resultPlaceholderFor_usesOutputKindSpecificCopy() {
        assertEquals(
            "Run a model prompt with the selected image to see the latest response.",
            resultPlaceholderFor(OutputKind.TEXT_RESPONSE)
        )
        assertEquals(
            "Run embedding inference to see the latest summary.",
            resultPlaceholderFor(OutputKind.EMBEDDING)
        )
        assertEquals(
            "Run classification inference to see the latest top results.",
            resultPlaceholderFor(OutputKind.CLASSIFICATION)
        )
    }

    @Test
    fun textResponseCopy_usesModelLabelInsteadOfFastVlm() {
        assertEquals(
            "Latest model text response for the selected image and prompt.",
            resultSectionDescriptionFor(OutputKind.TEXT_RESPONSE)
        )
        assertEquals(
            "Model request latency will appear in the response summary.",
            benchmarkPlaceholderFor(availableModelSpecs().first())
        )
    }
}

