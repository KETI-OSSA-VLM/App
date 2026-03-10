package com.example.genionputtest.inference.spec

enum class NormalizationMode {
    NONE,
    UNIT_RANGE,
    CHANNEL_STANDARDIZATION
}

data class NormalizationSpec(
    val mode: NormalizationMode = NormalizationMode.NONE,
    val mean: FloatArray = floatArrayOf(0f, 0f, 0f),
    val std: FloatArray = floatArrayOf(1f, 1f, 1f)
)
