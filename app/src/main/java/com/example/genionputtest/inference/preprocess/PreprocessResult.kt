package com.example.genionputtest.inference.preprocess

import java.nio.ByteBuffer

data class PreprocessResult(
    val inputBuffer: ByteBuffer,
    val width: Int,
    val height: Int
)
