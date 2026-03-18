package com.example.genionputtest.fastvlm

data class FastVlmResponse(
    val text: String,
    val latencySummary: String,
    val debugSummary: String = ""
)
