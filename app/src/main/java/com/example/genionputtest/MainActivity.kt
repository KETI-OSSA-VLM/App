package com.example.genionputtest

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageDecoder
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.util.TypedValue
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
import kotlin.math.roundToInt

internal data class MainScreenCopy(
    val title: String,
    val subtitle: String,
    val actionTitle: String,
    val previewTitle: String,
    val resultTitle: String,
    val benchmarkTitle: String,
    val statusTitle: String,
    val initialStatus: String,
    val initialResult: String,
    val initialBenchmark: String
)

internal fun initialScreenCopy(): MainScreenCopy = MainScreenCopy(
    title = "MobileCLIP2 Encoder Demo",
    subtitle = "Pick an image to inspect the latest embedding, benchmark, and run log.",
    actionTitle = "Action",
    previewTitle = "Selected image",
    resultTitle = "Embedding result",
    benchmarkTitle = "Benchmark",
    statusTitle = "Status log",
    initialStatus = "Model is loading.",
    initialResult = "Run embedding inference to see the latest summary.",
    initialBenchmark = "Benchmarks will appear after model initialization."
)

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var pickImageButton: Button
    private lateinit var statusView: TextView
    private lateinit var resultView: TextView
    private lateinit var benchmarkView: TextView
    private lateinit var modelBuffer: MappedByteBuffer
    private val modelSpec: ModelSpec = MobileClip2S0Spec
    private val screenCopy = initialScreenCopy()
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
        val pagePadding = dp(20)
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pagePadding, pagePadding, pagePadding, pagePadding)
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }
        val cardSpacing = dp(14)

        pickImageButton = Button(this).apply {
            text = "Pick image"
            setAllCaps(false)
            textSize = 16f
            setPadding(dp(18), dp(14), dp(18), dp(14))
            setOnClickListener {
                pickImageLauncher.launch("image/*")
            }
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }

        imageView = ImageView(this).apply {
            adjustViewBounds = true
            minimumHeight = dp(220)
            scaleType = ImageView.ScaleType.CENTER_CROP
            setBackgroundColor(Color.parseColor("#ECF2F8"))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }

        statusView = TextView(this).apply {
            textSize = 13f
            setTextColor(Color.parseColor("#5F6B76"))
            setLineSpacing(0f, 1.15f)
            text = screenCopy.initialStatus
        }

        resultView = TextView(this).apply {
            textSize = 15f
            setTextColor(Color.parseColor("#122033"))
            setLineSpacing(0f, 1.2f)
            text = screenCopy.initialResult
        }

        benchmarkView = TextView(this).apply {
            textSize = 15f
            setTextColor(Color.parseColor("#122033"))
            setLineSpacing(0f, 1.2f)
            text = screenCopy.initialBenchmark
        }

        container.addView(createHeaderBlock(), createCardLayoutParams(bottomMargin = cardSpacing))

        val actionCard = createSectionCard(
            title = screenCopy.actionTitle,
            description = "Choose a test image from the device."
        )
        actionCard.addView(pickImageButton)
        container.addView(actionCard, createCardLayoutParams(bottomMargin = cardSpacing))

        val previewCard = createSectionCard(
            title = screenCopy.previewTitle,
            description = "The selected frame is shown here before inference."
        )
        previewCard.addView(imageView)
        container.addView(previewCard, createCardLayoutParams(bottomMargin = cardSpacing))

        val resultCard = createSectionCard(
            title = screenCopy.resultTitle,
            description = "Latest embedding summary and exported JSON locations."
        )
        resultCard.addView(resultView)
        container.addView(resultCard, createCardLayoutParams(bottomMargin = cardSpacing))

        val benchmarkCard = createSectionCard(
            title = screenCopy.benchmarkTitle,
            description = "Startup benchmark snapshot for CPU and NNAPI."
        )
        benchmarkCard.addView(benchmarkView)
        container.addView(benchmarkCard, createCardLayoutParams(bottomMargin = cardSpacing))

        val logCard = createSectionCard(
            title = screenCopy.statusTitle,
            description = "Lower-priority progress history."
        )
        val logScroll = ScrollView(this).apply {
            isFillViewport = true
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(150)
            )
            addView(statusView)
        }
        logCard.addView(logScroll)
        container.addView(logCard)

        return ScrollView(this).apply {
            isFillViewport = true
            setBackgroundColor(Color.parseColor("#EEF3F7"))
            addView(container)
        }
    }

    private fun createHeaderBlock(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(4), dp(4), dp(4), dp(4))
            addView(TextView(context).apply {
                text = screenCopy.title
                setTextColor(Color.parseColor("#102033"))
                setTypeface(typeface, Typeface.BOLD)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 26f)
            })
            addView(TextView(context).apply {
                text = modelSpec.modelName
                setTextColor(Color.parseColor("#2E5B88"))
                setTypeface(typeface, Typeface.BOLD)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
                setPadding(0, dp(8), 0, 0)
            })
            addView(TextView(context).apply {
                text = screenCopy.subtitle
                setTextColor(Color.parseColor("#5F6B76"))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
                setLineSpacing(0f, 1.2f)
                setPadding(0, dp(10), 0, 0)
            })
        }
    }

    private fun createSectionCard(title: String, description: String): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(18), dp(18), dp(18))
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = dp(20).toFloat()
                setColor(Color.WHITE)
                setStroke(dp(1), Color.parseColor("#D7E0E8"))
            }
            elevation = dp(2).toFloat()
            addView(createSectionTitle(title))
            addView(createSectionDescription(description))
        }
    }

    private fun createSectionTitle(title: String): TextView {
        return TextView(this).apply {
            text = title
            setTextColor(Color.parseColor("#102033"))
            setTypeface(typeface, Typeface.BOLD)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
        }
    }

    private fun createSectionDescription(description: String): TextView {
        return TextView(this).apply {
            text = description
            setTextColor(Color.parseColor("#6A7681"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            setLineSpacing(0f, 1.15f)
            setPadding(0, dp(6), 0, dp(14))
        }
    }

    private fun createCardLayoutParams(bottomMargin: Int = 0): LinearLayout.LayoutParams {
        return LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply {
            this.bottomMargin = bottomMargin
        }
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).roundToInt()

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
