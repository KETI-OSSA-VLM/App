package com.example.genionputtest.inference.spec

import org.tensorflow.lite.DataType

object MobileClip2S0Spec : ModelSpec {
    override val modelName: String = "MobileCLIP2-S0 Image Encoder"
    override val assetName: String = "mobileclip2_s0_image_encoder_float16.tflite"
    override val inputWidth: Int = 256
    override val inputHeight: Int = 256
    override val inputChannels: Int = 3
    override val inputDataType: DataType = DataType.FLOAT32
    override val outputKind: OutputKind = OutputKind.EMBEDDING
    override val normalization: NormalizationSpec = NormalizationSpec(
        mode = NormalizationMode.CHANNEL_STANDARDIZATION,
        mean = floatArrayOf(0f, 0f, 0f),
        std = floatArrayOf(1f, 1f, 1f)
    )
}