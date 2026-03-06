package com.example.genionputtest

import java.nio.ByteBuffer

internal data class Prediction(
    val index: Int,
    val score: Int
)

internal fun formatBenchmarkStatus(tag: String, avgMs: Double, runs: Int): String {
    return "[$tag] avg=%.3f ms (runs=%d)".format(avgMs, runs)
}

internal fun extractTopClasses(output: ByteBuffer, topK: Int): List<Prediction> {
    val scores = output.duplicate().apply { rewind() }
    val predictions = mutableListOf<Prediction>()
    for (index in 0 until scores.remaining()) {
        predictions += Prediction(index = index, score = scores.get().toInt() and 0xFF)
    }
    return predictions
        .sortedByDescending { it.score }
        .take(topK)
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

internal fun formatClassificationResults(predictions: List<Prediction>, labels: List<String>): String {
    return predictions.mapIndexed { rank, prediction ->
        val label = labels.getOrNull(prediction.index) ?: "unknown"
        "#${rank + 1} $label (class ${prediction.index}) score=${prediction.score}"
    }.joinToString("\n")
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
