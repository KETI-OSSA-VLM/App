package com.example.genionputtest.inference.core

import org.tensorflow.lite.DataType
import java.nio.ByteBuffer

data class InferenceResult(
    val outputBuffer: ByteBuffer,
    val inferenceMs: Double,
    val inputShape: IntArray,
    val outputShape: IntArray,
    val inputDataType: DataType,
    val outputDataType: DataType
)
