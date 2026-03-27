package com.example.genionputtest

import com.example.genionputtest.benchmark.LatencyBreakdown
import com.example.genionputtest.inference.postprocess.ClassificationPostprocessor
import com.example.genionputtest.inference.postprocess.EmbeddingOutput
import java.nio.ByteBuffer

internal data class Prediction(
    val index: Int,
    val score: Int
)


internal fun extractTopClasses(output: ByteBuffer, topK: Int): List<Prediction> {
    return ClassificationPostprocessor(topK = topK).fromOutput(output).predictions
}

internal fun formatClassificationResults(predictions: List<Prediction>): String {
    return predictions.mapIndexed { rank, prediction ->
        "#${rank + 1} class ${prediction.index} score=${prediction.score}"
    }.joinToString("\n")
}

internal fun formatClassificationSummary(
    sourceLabel: String,
    inferenceMs: Double,
    labels: List<String>,
    predictions: List<Prediction>
): String {
    return buildString {
        append(sourceLabel)
        append('\n')
        append("Inference: ")
        append("%.3f ms".format(inferenceMs))
        append('\n')
        append(formatClassificationResults(predictions, labels))
    }
}

internal fun formatClassificationSummary(
    sourceLabel: String,
    labels: List<String>,
    predictions: List<Prediction>,
    latencyBreakdown: LatencyBreakdown
): String {
    return buildString {
        append(sourceLabel)
        append('\n')
        append(formatLatencyBreakdown(latencyBreakdown))
        append('\n')
        append(formatClassificationResults(predictions, labels))
    }
}

internal fun formatClassificationResults(predictions: List<Prediction>, labels: List<String>): String {
    return predictions.mapIndexed { rank, prediction ->
        val label = labels.getOrNull(prediction.index) ?: "unknown"
        "#${rank + 1} $label (class ${prediction.index}) score=${prediction.score}"
    }.joinToString("\n")
}

internal fun formatEmbeddingSummary(
    sourceLabel: String,
    embedding: EmbeddingOutput,
    latencyBreakdown: LatencyBreakdown
): String {
    val preview = embedding.previewValues.joinToString(prefix = "[", postfix = "]") { "%.3f".format(it) }
    return buildString {
        append(sourceLabel)
        append('\n')
        append(formatLatencyBreakdown(latencyBreakdown))
        append('\n')
        append("Dimension: ")
        append(embedding.dimension)
        append('\n')
        append("L2 norm: ")
        append("%.3f".format(embedding.l2Norm))
        append('\n')
        append("Preview: ")
        append(preview)
    }
}

internal inline fun runInferenceIterations(
    iterations: Int,
    input: ByteBuffer,
    output: ByteBuffer,
    run: (ByteBuffer, ByteBuffer) -> Unit
) {
    repeat(iterations) {
        input.rewind()
        output.rewind()
        run(input, output)
    }
}

internal fun formatLatencyBreakdown(latencyBreakdown: LatencyBreakdown): String {
    return buildString {
        append("Preprocess: ")
        append("%.3f ms".format(latencyBreakdown.preprocessMs))
        append('\n')
        append("Inference: ")
        append("%.3f ms".format(latencyBreakdown.inferenceMs))
        append('\n')
        append("Postprocess: ")
        append("%.3f ms".format(latencyBreakdown.postprocessMs))
        append('\n')
        append("Total: ")
        append("%.3f ms".format(latencyBreakdown.totalMs))
    }
}