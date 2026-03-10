package com.example.genionputtest.inference.postprocess

import com.example.genionputtest.Prediction

internal data class ClassificationOutput(
    val predictions: List<Prediction>
)
