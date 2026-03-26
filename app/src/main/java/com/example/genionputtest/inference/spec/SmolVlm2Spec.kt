package com.example.genionputtest.inference.spec

import org.tensorflow.lite.DataType

object SmolVlm2Spec : ModelSpec {
    override val modelName: String = "SmolVLM2 500M"
    override val assetName: String = "SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
    val mmprojAssetName: String = "mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf"
    override val inputWidth: Int = 0
    override val inputHeight: Int = 0
    override val inputChannels: Int = 3
    override val inputDataType: DataType = DataType.FLOAT32
    override val outputKind: OutputKind = OutputKind.TEXT_RESPONSE
    override val normalization: NormalizationSpec = NormalizationSpec()
}
