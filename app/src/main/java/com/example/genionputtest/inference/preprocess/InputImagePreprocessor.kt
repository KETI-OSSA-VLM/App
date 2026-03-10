package com.example.genionputtest.inference.preprocess

import android.graphics.Bitmap
import com.example.genionputtest.inference.spec.ModelSpec
import com.example.genionputtest.inference.spec.NormalizationMode
import org.tensorflow.lite.DataType
import java.nio.ByteBuffer
import java.nio.ByteOrder

class InputImagePreprocessor {

    fun preprocess(bitmap: Bitmap, modelSpec: ModelSpec): PreprocessResult {
        val scaled = Bitmap.createScaledBitmap(bitmap, modelSpec.inputWidth, modelSpec.inputHeight, true)
        val readableBitmap = if (scaled.config == Bitmap.Config.HARDWARE) {
            scaled.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            scaled
        }
        val pixels = IntArray(modelSpec.inputWidth * modelSpec.inputHeight)
        readableBitmap.getPixels(
            pixels,
            0,
            modelSpec.inputWidth,
            0,
            0,
            modelSpec.inputWidth,
            modelSpec.inputHeight
        )

        val input = ByteBuffer.allocateDirect(
            expectedInputByteSize(
                intArrayOf(1, modelSpec.inputHeight, modelSpec.inputWidth, modelSpec.inputChannels),
                modelSpec.inputDataType
            )
        ).order(ByteOrder.nativeOrder())

        for (pixel in pixels) {
            val red = ((pixel shr 16) and 0xFF).toFloat()
            val green = ((pixel shr 8) and 0xFF).toFloat()
            val blue = (pixel and 0xFF).toFloat()
            writeChannelValue(input, modelSpec, channelIndex = 0, rawValue = red)
            writeChannelValue(input, modelSpec, channelIndex = 1, rawValue = green)
            writeChannelValue(input, modelSpec, channelIndex = 2, rawValue = blue)
        }

        input.rewind()
        return PreprocessResult(
            inputBuffer = input,
            width = modelSpec.inputWidth,
            height = modelSpec.inputHeight
        )
    }

    private fun writeChannelValue(
        input: ByteBuffer,
        modelSpec: ModelSpec,
        channelIndex: Int,
        rawValue: Float
    ) {
        when (modelSpec.inputDataType) {
            DataType.UINT8 -> input.put(rawValue.toInt().toByte())
            DataType.FLOAT32 -> input.putFloat(normalize(rawValue, modelSpec, channelIndex))
            else -> error("Unsupported input type: ${modelSpec.inputDataType}")
        }
    }

    private fun normalize(rawValue: Float, modelSpec: ModelSpec, channelIndex: Int): Float {
        return when (modelSpec.normalization.mode) {
            NormalizationMode.NONE -> rawValue
            NormalizationMode.UNIT_RANGE -> rawValue / 255f
            NormalizationMode.CHANNEL_STANDARDIZATION -> {
                val unitRange = rawValue / 255f
                val mean = modelSpec.normalization.mean[channelIndex]
                val std = modelSpec.normalization.std[channelIndex]
                (unitRange - mean) / std
            }
        }
    }
}

fun expectedInputByteSize(shape: IntArray, dataType: DataType): Int {
    val elementCount = shape.fold(1, Int::times)
    val bytesPerElement = when (dataType) {
        DataType.UINT8, DataType.INT8 -> 1
        DataType.FLOAT32 -> 4
        else -> error("Unsupported input type: $dataType")
    }
    return elementCount * bytesPerElement
}
