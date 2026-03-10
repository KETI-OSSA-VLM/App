package com.example.genionputtest.inference.postprocess

data class EmbeddingOutput(
    val values: FloatArray,
    val previewValues: FloatArray,
    val dimension: Int,
    val l2Norm: Float
)
