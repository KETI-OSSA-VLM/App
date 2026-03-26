package com.example.genionputtest.llamacpp

internal class LlamaCppBridge {

    external fun loadModel(modelPath: String, mmprojPath: String, nThreads: Int): Boolean
    external fun generate(imagePath: String, promptText: String, maxNewTokens: Int): String
    external fun freeModel()

    companion object {
        init {
            System.loadLibrary("accv_llama")
        }
    }
}
