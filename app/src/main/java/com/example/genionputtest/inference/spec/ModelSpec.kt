package com.example.genionputtest.inference.spec

import org.tensorflow.lite.DataType

interface ModelSpec {
    val modelName: String
    val assetName: String
    val inputWidth: Int
    val inputHeight: Int
    val inputChannels: Int
    val inputDataType: DataType
    val outputKind: OutputKind
    val normalization: NormalizationSpec

    val requiresNormalization: Boolean
        get() = normalization.mode != NormalizationMode.NONE
}
