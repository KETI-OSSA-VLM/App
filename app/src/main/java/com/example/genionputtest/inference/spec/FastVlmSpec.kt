package com.example.genionputtest.inference.spec

import org.tensorflow.lite.DataType

object FastVlmSpec : ModelSpec {
    override val modelName: String = "Gemma 3n E2B IT Int4"
    override val assetName: String = "Gemma-3n-E2B-it-int4.litertlm"
    override val inputWidth: Int = 0
    override val inputHeight: Int = 0
    override val inputChannels: Int = 3
    override val inputDataType: DataType = DataType.FLOAT32
    override val outputKind: OutputKind = OutputKind.TEXT_RESPONSE
    override val normalization: NormalizationSpec = NormalizationSpec()
}
