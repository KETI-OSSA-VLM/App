package com.example.genionputtest

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.genionputtest.benchmark.LatencyBreakdown
import com.example.genionputtest.benchmark.LatencyTracker
import com.example.genionputtest.inference.core.InferenceEngine
import com.example.genionputtest.inference.core.InferenceOptions
import com.example.genionputtest.inference.core.ModelAssetLoader
import com.example.genionputtest.inference.postprocess.EmbeddingJsonStore
import com.example.genionputtest.inference.postprocess.EmbeddingOutput
import com.example.genionputtest.inference.postprocess.EmbeddingPostprocessor
import com.example.genionputtest.inference.preprocess.InputImagePreprocessor
import com.example.genionputtest.inference.preprocess.InputTensorJsonStore
import com.example.genionputtest.inference.spec.MobileClip2S0Spec
import com.example.genionputtest.inference.spec.ModelSpec
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.MappedByteBuffer

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var pickImageButton: Button
    private lateinit var statusView: TextView
    private lateinit var resultView: TextView
    private lateinit var benchmarkView: TextView
    private lateinit var modelBuffer: MappedByteBuffer
    private val modelSpec: ModelSpec = MobileClip2S0Spec
    private val inputImagePreprocessor = InputImagePreprocessor()
    private val inputTensorJsonStore = InputTensorJsonStore()
    private val embeddingPostprocessor = EmbeddingPostprocessor(previewValueCount = 8)
    private val embeddingJsonStore = EmbeddingJsonStore()

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) {
            appendStatus("Image selection canceled.")
            return@registerForActivityResult
        }
        runEmbeddingForSelectedImage(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(createContentView())

        lifecycleScope.launch {
            appendStatus("Loading model...")
            modelBuffer = withContext(Dispatchers.IO) {
                ModelAssetLoader(assets).loadMapped(modelSpec.assetName)
            }
            appendStatus("Model loaded: ${modelSpec.assetName}")
            appendStatus("Pick an image from the device to run embedding inference.")

            appendStatus("Running CPU benchmark...")
            val cpuResult = withContext(Dispatchers.Default) {
                runBenchmark(tag = "CPU", modelBuffer = modelBuffer, useNnapi = false)
            }

            appendStatus("Running NNAPI benchmark...")
            val nnapiResult = withContext(Dispatchers.Default) {
                runBenchmark(tag = "NNAPI", modelBuffer = modelBuffer, useNnapi = true)
            }

            benchmarkView.text = buildString {
                append("Benchmark summary\n")
                append(cpuResult)
                append('\n')
                append(nnapiResult)
            }
            appendStatus("Benchmarks finished.")
        }
    }

    private fun createContentView(): ScrollView {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 48, 48, 48)
            gravity = Gravity.CENTER_HORIZONTAL
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }

        pickImageButton = Button(this).apply {
            text = "Pick image"
            setOnClickListener {
                pickImageLauncher.launch("image/*")
            }
        }

        imageView = ImageView(this).apply {
            adjustViewBounds = true
            minimumHeight = 480
            minimumWidth = 320
        }

        statusView = TextView(this).apply {
            textSize = 18f
            text = "Preparing embedding inference..."
        }

        resultView = TextView(this).apply {
            textSize = 16f
            text = "Embedding results will appear here."
        }

        benchmarkView = TextView(this).apply {
            textSize = 16f
            text = "Benchmark results will appear here."
        }

        container.addView(pickImageButton)
        container.addView(imageView)
        container.addView(statusView)
        container.addView(resultView)
        container.addView(benchmarkView)

        return ScrollView(this).apply {
            addView(container)
        }
    }

    private fun runEmbeddingForSelectedImage(uri: Uri) {
        lifecycleScope.launch {
            try {
                appendStatus("Loading selected image...")
                val bitmap = withContext(Dispatchers.IO) {
                    loadBitmapFromUri(uri)
                }
                imageView.setImageBitmap(bitmap)

                appendStatus("Running embedding inference for selected image...")
                val embeddingResult = withContext(Dispatchers.Default) {
                    runEmbedding(bitmap)
                }

                resultView.text = formatEmbeddingSummary(
                    sourceLabel = "selected image",
                    embedding = embeddingResult.embedding,
                    latencyBreakdown = embeddingResult.latencyBreakdown
                ) + "\nInput JSON: ${embeddingResult.inputJsonFile.absolutePath}" +
                    "\nEmbedding JSON: ${embeddingResult.embeddingJsonFile.absolutePath}"
                appendStatus("Input JSON saved: ${embeddingResult.inputJsonFile.absolutePath}")
                appendStatus("Embedding JSON saved: ${embeddingResult.embeddingJsonFile.absolutePath}")
                Log.i("GENIO_TEST", "Input JSON saved: ${embeddingResult.inputJsonFile.absolutePath}")
                Log.i("GENIO_TEST", "Embedding JSON saved: ${embeddingResult.embeddingJsonFile.absolutePath}")
                appendStatus("Embedding inference complete.")
            } catch (t: Throwable) {
                Log.e("GENIO_TEST", "Embedding inference failed", t)
                resultView.text = "Embedding inference failed: ${t.javaClass.simpleName}"
                appendStatus("Embedding inference failed. Check logcat.")
            }
        }
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }
    }

    private fun runEmbedding(bitmap: Bitmap): EmbeddingResult {
        val preprocess = LatencyTracker.measure {
            inputImagePreprocessor.preprocess(bitmap, modelSpec)
        }
        val inputJsonFile = inputTensorJsonStore.write(filesDir, preprocess.value, modelSpec)

        InferenceEngine(modelBuffer).use { engine ->
            val inference = engine.run(preprocess.value.inputBuffer)
            val postprocess = LatencyTracker.measure {
                embeddingPostprocessor.fromOutput(inference.outputBuffer)
            }
            val embeddingJsonFile = embeddingJsonStore.write(filesDir, postprocess.value)

            return EmbeddingResult(
                embedding = postprocess.value,
                latencyBreakdown = LatencyBreakdown(
                    preprocessMs = preprocess.durationMs,
                    inferenceMs = inference.inferenceMs,
                    postprocessMs = postprocess.durationMs
                ),
                inputJsonFile = inputJsonFile,
                embeddingJsonFile = embeddingJsonFile
            )
        }
    }

    private fun runBenchmark(tag: String, modelBuffer: MappedByteBuffer, useNnapi: Boolean): String {
        val runs = 50
        InferenceEngine(
            modelBuffer = modelBuffer,
            options = InferenceOptions(useNnapi = useNnapi)
        ).use { engine ->
            Log.i("GENIO_TEST", "[$tag] input shape=${engine.inputShape().contentToString()} type=${engine.inputDataType()}")
            Log.i("GENIO_TEST", "[$tag] output shape=${engine.outputShape().contentToString()} type=${engine.outputDataType()}")

            val avgMs = engine.benchmark(
                warmupRuns = 5,
                runs = runs,
                inputBuffer = engine.createBenchmarkInput()
            )
            val result = formatBenchmarkStatus(tag = tag, avgMs = avgMs, runs = runs)
            Log.i("GENIO_TEST", result)
            return result
        }
    }

    private fun appendStatus(message: String) {
        statusView.text = buildString {
            val current = statusView.text.toString()
            if (current.isNotEmpty()) {
                append(current)
                append('\n')
            }
            append(message)
        }
    }
}

private data class EmbeddingResult(
    val embedding: EmbeddingOutput,
    val latencyBreakdown: LatencyBreakdown,
    val inputJsonFile: File,
    val embeddingJsonFile: File
)
