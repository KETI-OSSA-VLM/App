package com.example.genionputtest.inference.spec

import org.tensorflow.lite.DataType

object MobileNetSpec : ModelSpec {
    override val modelName: String = "MobileNet v1 Quantized"
    override val assetName: String = "mobilenet_v1_1.0_224_quant.tflite"
    override val inputWidth: Int = 224
    override val inputHeight: Int = 224
    override val inputChannels: Int = 3
    override val inputDataType: DataType = DataType.UINT8
    override val outputKind: OutputKind = OutputKind.CLASSIFICATION
    override val normalization: NormalizationSpec = NormalizationSpec()
}
