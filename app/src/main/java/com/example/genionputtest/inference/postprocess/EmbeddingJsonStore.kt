package com.example.genionputtest.inference.postprocess

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

class EmbeddingJsonStore(
    private val fileName: String = "mobileclip2_last_embedding.json"
) {
    fun write(filesDir: File, embedding: EmbeddingOutput): File {
        val outputFile = File(filesDir, fileName)
        val payload = JSONObject().apply {
            put("dimension", embedding.dimension)
            put("l2Norm", embedding.l2Norm.toDouble())
            put("previewValues", JSONArray(embedding.previewValues.map(Float::toDouble)))
            put("values", JSONArray(embedding.values.map(Float::toDouble)))
        }
        outputFile.writeText(payload.toString(2), Charsets.UTF_8)
        return outputFile
    }
}
