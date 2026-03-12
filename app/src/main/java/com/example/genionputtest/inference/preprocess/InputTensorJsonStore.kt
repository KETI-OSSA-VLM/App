package com.example.genionputtest.inference.preprocess

import com.example.genionputtest.inference.spec.ModelSpec
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.DataType
import java.io.File
import java.nio.ByteOrder

class InputTensorJsonStore(
    private val fileName: String = "mobileclip2_last_input.json"
) {
    fun write(filesDir: File, preprocessResult: PreprocessResult, modelSpec: ModelSpec): File {
        val outputFile = File(filesDir, fileName)
        val shape = listOf(1, preprocessResult.height, preprocessResult.width, modelSpec.inputChannels)
        val values = when (modelSpec.inputDataType) {
            DataType.FLOAT32 -> preprocessResult.inputBuffer.duplicate()
                .order(ByteOrder.nativeOrder())
                .apply { rewind() }
                .asFloatBuffer()
                .let { buffer ->
                    FloatArray(buffer.remaining()).also(buffer::get).map(Float::toDouble)
                }
            else -> error("Unsupported input type for JSON export: ${modelSpec.inputDataType}")
        }
        val payload = JSONObject().apply {
            put("shape", JSONArray(shape))
            put("layout", "NHWC")
            put("dataType", modelSpec.inputDataType.toString())
            put("values", JSONArray(values))
        }
        outputFile.writeText(payload.toString(2), Charsets.UTF_8)
        return outputFile
    }
}
