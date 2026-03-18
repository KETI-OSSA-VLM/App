package com.example.genionputtest.fastvlm

import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

interface FastVlmEngine : AutoCloseable {
    suspend fun initialize()
    suspend fun generate(request: FastVlmRequest): FastVlmResponse
}

internal fun buildFastVlmLatencySummary(
    elapsedMs: Double,
    prefillTokenCount: Int?,
    decodeTokenCount: Int?
): String {
    return buildString {
        append("Inference: ")
        append("%.3f ms".format(elapsedMs))
        if (prefillTokenCount != null && decodeTokenCount != null) {
            append(" | Tokens(prefill=")
            append(prefillTokenCount)
            append(", decode=")
            append(decodeTokenCount)
            append(')')
        }
    }
}

internal fun extractFastVlmText(contents: List<Content>): String {
    return contents
        .filterIsInstance<Content.Text>()
        .joinToString(separator = "\n") { it.text }
        .ifBlank { "FastVLM returned an empty response." }
}

internal fun summarizeFastVlmContents(contents: List<Content>): String {
    return contents.joinToString(separator = " | ") { content ->
        when (content) {
            is Content.Text -> "Text(${content.text.take(120)})"
            is Content.ImageFile -> "ImageFile"
            is Content.ImageBytes -> "ImageBytes"
            is Content.AudioFile -> "AudioFile"
            is Content.AudioBytes -> "AudioBytes"
            is Content.ToolResponse -> "ToolResponse"
        }
    }.ifBlank { "No LiteRT-LM contents returned." }
}

internal fun isPromptOnlyRequest(request: FastVlmRequest): Boolean {
    return request.imagePath.isNullOrBlank()
}

internal fun buildFastVlmContents(request: FastVlmRequest): Contents {
    val contents = mutableListOf<Content>()
    if (!request.imagePath.isNullOrBlank()) {
        contents.add(Content.ImageFile(request.imagePath!!))
    }
    if (request.prompt.isNotBlank()) {
        contents.add(Content.Text(request.prompt))
    }
    return Contents.of(contents)
}

internal fun buildFastVlmConversationConfig(): ConversationConfig {
    return ConversationConfig(
        samplerConfig = SamplerConfig(
            topK = 1,
            topP = 1.0,
            temperature = 0.0,
            seed = 7
        )
    )
}

internal fun buildFastVlmEngineConfig(modelPath: String, cacheDirPath: String): EngineConfig {
    return EngineConfig(
        modelPath = modelPath,
        backend = Backend.CPU(),
        visionBackend = Backend.GPU(),
        maxNumTokens = 512,
        cacheDir = cacheDirPath
    )
}

class LiteRtFastVlmEngine(
    private val modelPath: String,
    private val cacheDirPath: String
) : FastVlmEngine {

    private var engine: Engine? = null

    override suspend fun initialize() {
        withContext(Dispatchers.IO) {
            if (engine?.isInitialized() == true) {
                return@withContext
            }
            val newEngine = Engine(buildFastVlmEngineConfig(modelPath, cacheDirPath))
            newEngine.initialize()
            engine = newEngine
        }
    }

    @OptIn(ExperimentalApi::class)
    override suspend fun generate(request: FastVlmRequest): FastVlmResponse {
        return withContext(Dispatchers.Default) {
            val readyEngine = engine ?: error("FastVLM engine is not initialized.")
            val startedAtNs = System.nanoTime()
            readyEngine.createConversation(buildFastVlmConversationConfig()).use { conversation ->
                val response = conversation.sendMessage(buildFastVlmContents(request))
                val elapsedMs = (System.nanoTime() - startedAtNs) / 1_000_000.0
                val benchmark = runCatching { conversation.getBenchmarkInfo() }.getOrNull()
                val debugSummary = summarizeFastVlmContents(response.contents.contents)
                if (
                    response.contents.contents.any { it is Content.Text && it.text.contains("start_of_turn") } ||
                    response.contents.contents.any { it is Content.Text && it.text.contains("<|im_start|>") }
                ) {
                    Log.w("GENIO_TEST", "Suspicious FastVLM response markers: $debugSummary")
                } else {
                    Log.i("GENIO_TEST", "FastVLM raw response: $debugSummary")
                }
                FastVlmResponse(
                    text = extractText(response),
                    latencySummary = buildFastVlmLatencySummary(
                        elapsedMs = elapsedMs,
                        prefillTokenCount = benchmark?.lastPrefillTokenCount,
                        decodeTokenCount = benchmark?.lastDecodeTokenCount
                    ),
                    debugSummary = debugSummary
                )
            }
        }
    }

    override fun close() {
        engine?.close()
        engine = null
    }

    private fun extractText(message: Message): String {
        return extractFastVlmText(message.contents.contents)
    }
}
