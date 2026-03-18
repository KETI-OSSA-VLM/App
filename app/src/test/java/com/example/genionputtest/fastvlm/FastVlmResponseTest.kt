package com.example.genionputtest.fastvlm

import com.google.ai.edge.litertlm.Content
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FastVlmResponseTest {

    @Test
    fun textResponse_exposesNonBlankOutput() {
        val response = FastVlmResponse(text = "A person is walking.", latencySummary = "ready")

        assertEquals("A person is walking.", response.text)
        assertEquals("ready", response.latencySummary)
    }

    @Test
    fun extractResponseText_returnsJoinedTextContentsOnly() {
        val text = extractFastVlmText(
            listOf(
                Content.Text("First line"),
                Content.ImageFile("/tmp/example.jpg"),
                Content.Text("Second line")
            )
        )

        assertEquals("First line\nSecond line", text)
    }

    @Test
    fun summarizeFastVlmContents_reportsTypesAndTextPreview() {
        val summary = summarizeFastVlmContents(
            listOf(
                Content.Text("<start_of_turn>user"),
                Content.ImageFile("/tmp/example.jpg"),
                Content.Text("Second line")
            )
        )

        assertTrue(summary.contains("Text"))
        assertTrue(summary.contains("ImageFile"))
        assertTrue(summary.contains("<start_of_turn>user"))
    }

    @Test
    fun isPromptOnlyRequest_returnsTrueWhenImagePathMissing() {
        val result = isPromptOnlyRequest(
            FastVlmRequest(
                prompt = "Describe the scene.",
                imagePath = null
            )
        )

        assertTrue(result)
    }

    @Test
    fun isPromptOnlyRequest_returnsFalseWhenImagePathExists() {
        val result = isPromptOnlyRequest(
            FastVlmRequest(
                prompt = "Describe the scene.",
                imagePath = "/tmp/example.jpg"
            )
        )

        assertFalse(result)
    }

    @Test
    fun buildFastVlmContents_returnsTextOnlyForPromptOnlyRequest() {
        val contents = buildFastVlmContents(
            FastVlmRequest(
                prompt = "Describe the scene.",
                imagePath = null
            )
        )

        assertEquals(1, contents.contents.size)
        assertEquals("Describe the scene.", (contents.contents[0] as Content.Text).text)
    }

    @Test
    fun buildFastVlmContents_ordersImageBeforeTextForMultimodalRequest() {
        val contents = buildFastVlmContents(
            FastVlmRequest(
                prompt = "Describe the scene.",
                imagePath = "/tmp/example.jpg"
            )
        )

        assertEquals(2, contents.contents.size)
        assertTrue(contents.contents[0] is Content.ImageFile)
        assertEquals("Describe the scene.", (contents.contents[1] as Content.Text).text)
    }
}

