package com.example.genionputtest

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageDecoder
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.util.TypedValue
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.genionputtest.benchmark.LatencyBreakdown
import com.example.genionputtest.benchmark.LatencyTracker
import com.example.genionputtest.fastvlm.FastVlmEngine
import com.example.genionputtest.fastvlm.FastVlmRequest
import com.example.genionputtest.fastvlm.FastVlmResponse
import com.example.genionputtest.fastvlm.LiteRtFastVlmEngine
import com.example.genionputtest.fastvlm.defaultFastVlmPrompt
import com.example.genionputtest.llamacpp.LlamaCppEngine
import com.example.genionputtest.inference.core.InferenceEngine
import com.example.genionputtest.inference.core.InferenceOptions
import com.example.genionputtest.inference.core.ModelAssetLoader
import com.example.genionputtest.inference.postprocess.ClassificationPostprocessor
import com.example.genionputtest.inference.postprocess.EmbeddingJsonStore
import com.example.genionputtest.inference.postprocess.EmbeddingOutput
import com.example.genionputtest.inference.postprocess.EmbeddingPostprocessor
import com.example.genionputtest.inference.preprocess.InputImagePreprocessor
import com.example.genionputtest.inference.preprocess.InputTensorJsonStore
import com.example.genionputtest.inference.spec.FastVlmSpec
import com.example.genionputtest.inference.spec.MobileClip2S0Spec
import com.example.genionputtest.inference.spec.MobileNetSpec
import com.example.genionputtest.inference.spec.ModelSpec
import com.example.genionputtest.inference.spec.OutputKind
import com.example.genionputtest.inference.spec.SmolVlm2Spec
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.MappedByteBuffer
import kotlin.math.roundToInt

internal data class MainScreenCopy(
    val title: String,
    val subtitle: String,
    val actionTitle: String,
    val modelPickerLabel: String,
    val previewTitle: String,
    val resultTitle: String,
    val benchmarkTitle: String,
    val statusTitle: String,
    val initialStatus: String,
    val initialBenchmark: String
)

internal data class PreviewImageSpec(
    val maxWidthDp: Int,
    val maxHeightDp: Int,
    val minHeightDp: Int,
    val scaleTypeName: String
)

internal fun initialScreenCopy(): MainScreenCopy = MainScreenCopy(
    title = "Edge Vision Model Demo",
    subtitle = "Pick the active model, then choose an image to run the latest on-device inference flow.",
    actionTitle = "Action",
    modelPickerLabel = "Active model",
    previewTitle = "Selected image",
    resultTitle = "Result",
    benchmarkTitle = "Benchmark",
    statusTitle = "Status log",
    initialStatus = "Model is loading.",
    initialBenchmark = "Benchmarks will appear after model initialization."
)

internal fun previewImageSpec(): PreviewImageSpec = PreviewImageSpec(
    maxWidthDp = 280,
    maxHeightDp = 320,
    minHeightDp = 220,
    scaleTypeName = "FIT_CENTER"
)

internal fun availableModelSpecs(): List<ModelSpec> = listOf(
    SmolVlm2Spec,
    FastVlmSpec,
    MobileClip2S0Spec,
    MobileNetSpec
)

internal fun resultSectionTitleFor(outputKind: OutputKind): String = when (outputKind) {
    OutputKind.EMBEDDING -> "Embedding result"
    OutputKind.CLASSIFICATION -> "Classification result"
    OutputKind.TEXT_RESPONSE -> "Model response"
}

internal fun resultSectionDescriptionFor(outputKind: OutputKind): String = when (outputKind) {
    OutputKind.EMBEDDING -> "Latest embedding summary and exported JSON locations."
    OutputKind.CLASSIFICATION -> "Latest top predictions for the selected image."
    OutputKind.TEXT_RESPONSE -> "Latest model text response for the selected image and prompt."
}

internal fun resultPlaceholderFor(outputKind: OutputKind): String = when (outputKind) {
    OutputKind.EMBEDDING -> "Run embedding inference to see the latest summary."
    OutputKind.CLASSIFICATION -> "Run classification inference to see the latest top results."
    OutputKind.TEXT_RESPONSE -> "Run a model prompt with the selected image to see the latest response."
}

internal fun benchmarkPlaceholderFor(modelSpec: ModelSpec): String = when (modelSpec.outputKind) {
    OutputKind.TEXT_RESPONSE -> "Model request latency will appear in the response summary."
    else -> "Benchmarks will appear after ${modelSpec.modelName} initialization."
}

internal fun shouldShowPromptOnlyButton(outputKind: OutputKind): Boolean {
    return outputKind == OutputKind.TEXT_RESPONSE
}

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var pickImageButton: Button
    private lateinit var runPromptOnlyButton: Button
    private lateinit var modelSpinner: Spinner
    private lateinit var promptInput: EditText
    private lateinit var statusView: TextView
    private lateinit var resultView: TextView
    private lateinit var benchmarkView: TextView
    private lateinit var modelNameView: TextView
    private lateinit var resultSectionTitleView: TextView
    private lateinit var resultSectionDescriptionView: TextView
    private var modelBuffer: MappedByteBuffer? = null
    private var fastVlmEngine: FastVlmEngine? = null
    private var llamaCppEngine: LlamaCppEngine? = null
    private var spinnerInitialized = false
    private val availableModels = availableModelSpecs()
    private var selectedModelSpec: ModelSpec = availableModels.first()
    private val screenCopy = initialScreenCopy()
    private val modelAssetLoader by lazy { ModelAssetLoader(assets) }
    private val inputImagePreprocessor = InputImagePreprocessor()
    private val inputTensorJsonStore = InputTensorJsonStore()
    private val embeddingPostprocessor = EmbeddingPostprocessor(previewValueCount = 8)
    private val classificationPostprocessor = ClassificationPostprocessor(topK = 3)
    private val embeddingJsonStore = EmbeddingJsonStore()
    private var imageNetLabels: List<String>? = null

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) {
            appendStatus("Image selection canceled.")
            return@registerForActivityResult
        }
        runInferenceForSelectedImage(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(createContentView())
        configureModelPicker()
        applyModelPresentation(selectedModelSpec)
        loadSelectedModel(selectedModelSpec)
    }

    override fun onDestroy() {
        closeFastVlmEngine()
        closeLlamaCppEngine()
        super.onDestroy()
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

        modelSpinner = Spinner(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = dp(12)
            }
        }

        promptInput = EditText(this).apply {
            setText(defaultFastVlmPrompt())
            hint = "Describe the scene"
            minLines = 3
            maxLines = 5
            setTextColor(Color.parseColor("#122033"))
            setHintTextColor(Color.parseColor("#7C8792"))
            setBackgroundColor(Color.parseColor("#F5F8FB"))
            setPadding(dp(14), dp(12), dp(14), dp(12))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = dp(12)
            }
        }

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
            ).apply {
                bottomMargin = dp(12)
            }
        }

        runPromptOnlyButton = Button(this).apply {
            text = "Run prompt only"
            setAllCaps(false)
            textSize = 16f
            setPadding(dp(18), dp(14), dp(18), dp(14))
            setOnClickListener {
                runFastVlmTextOnlyClick()
            }
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }

        imageView = ImageView(this).apply {
            val previewSpec = previewImageSpec()
            adjustViewBounds = true
            minimumHeight = dp(previewSpec.minHeightDp)
            maxWidth = dp(previewSpec.maxWidthDp)
            maxHeight = dp(previewSpec.maxHeightDp)
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.parseColor("#ECF2F8"))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply {
                gravity = android.view.Gravity.CENTER_HORIZONTAL
            }
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
            text = resultPlaceholderFor(selectedModelSpec.outputKind)
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
            description = "Choose the active model, then pick a test image from the device."
        )
        actionCard.addView(createFieldLabel(screenCopy.modelPickerLabel))
        actionCard.addView(modelSpinner)
        actionCard.addView(createFieldLabel("Prompt"))
        actionCard.addView(promptInput)
        actionCard.addView(pickImageButton)
        actionCard.addView(runPromptOnlyButton)
        container.addView(actionCard, createCardLayoutParams(bottomMargin = cardSpacing))

        val previewCard = createSectionCard(
            title = screenCopy.previewTitle,
            description = "The selected frame is shown here before inference."
        )
        previewCard.addView(imageView)
        container.addView(previewCard, createCardLayoutParams(bottomMargin = cardSpacing))

        val resultCard = createCardContainer().apply {
            resultSectionTitleView = createSectionTitle(resultSectionTitleFor(selectedModelSpec.outputKind))
            resultSectionDescriptionView = createSectionDescription(resultSectionDescriptionFor(selectedModelSpec.outputKind))
            addView(resultSectionTitleView)
            addView(resultSectionDescriptionView)
            addView(resultView)
        }
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

    private fun configureModelPicker() {
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            availableModels.map { it.modelName }
        ).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        modelSpinner.adapter = adapter
        modelSpinner.setSelection(availableModels.indexOf(selectedModelSpec), false)
        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (!spinnerInitialized) {
                    spinnerInitialized = true
                    return
                }
                val nextModel = availableModels[position]
                if (nextModel == selectedModelSpec && (modelBuffer != null || fastVlmEngine != null || llamaCppEngine != null)) {
                    return
                }
                selectedModelSpec = nextModel
                applyModelPresentation(nextModel)
                loadSelectedModel(nextModel)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
    }

    private fun loadSelectedModel(modelSpec: ModelSpec) {
        lifecycleScope.launch {
            setInteractionEnabled(false)
            closeFastVlmEngine()
            closeLlamaCppEngine()
            modelBuffer = null
            try {
                appendStatus("Loading model: ${modelSpec.modelName}...")

                if (modelSpec is SmolVlm2Spec) {
                    val modelPath = withContext(Dispatchers.IO) { resolveModelFilePath(modelSpec.assetName) }
                    val mmprojPath = withContext(Dispatchers.IO) { resolveModelFilePath(modelSpec.mmprojAssetName) }
                    val engine = LlamaCppEngine(
                        modelPath = modelPath,
                        mmprojPath = mmprojPath,
                        cacheDir = cacheDir
                    )
                    withContext(Dispatchers.IO) { engine.initialize() }
                    llamaCppEngine = engine
                    benchmarkView.text = benchmarkPlaceholderFor(modelSpec)
                    appendStatus("SmolVLM2 ready (llama.cpp): $modelPath")
                    appendStatus("Pick an image and run the prompt.")
                    return@launch
                }

                if (modelSpec.outputKind == OutputKind.TEXT_RESPONSE) {
                    val modelPath = withContext(Dispatchers.IO) {
                        resolveFastVlmModelPath(modelSpec)
                    }
                    val engine = LiteRtFastVlmEngine(
                        modelPath = modelPath,
                        cacheDirPath = cacheDir.absolutePath
                    )
                    withContext(Dispatchers.IO) {
                        engine.initialize()
                    }
                    fastVlmEngine = engine
                    benchmarkView.text = benchmarkPlaceholderFor(modelSpec)
                    appendStatus("Model runtime ready: $modelPath")
                    appendStatus("Pick an image or run the prompt without an image.")
                    return@launch
                }

                modelBuffer = withContext(Dispatchers.IO) {
                    modelAssetLoader.loadMapped(modelSpec.assetName)
                }
                ensureSupportData(modelSpec)
                appendStatus("Model loaded: ${modelSpec.assetName}")
                appendStatus("Pick an image from the device to run ${resultSectionTitleFor(modelSpec.outputKind).lowercase()}.")

                appendStatus("Running CPU benchmark...")
                val cpuResult = withContext(Dispatchers.Default) {
                    runBenchmark(tag = "CPU", modelBuffer = requireModelBuffer(), useNnapi = false)
                }

                appendStatus("Running NNAPI benchmark...")
                val nnapiResult = withContext(Dispatchers.Default) {
                    runBenchmark(tag = "NNAPI", modelBuffer = requireModelBuffer(), useNnapi = true)
                }

                benchmarkView.text = buildString {
                    append(modelSpec.modelName)
                    append('\n')
                    append(cpuResult)
                    append('\n')
                    append(nnapiResult)
                }
                appendStatus("Benchmarks finished for ${modelSpec.modelName}.")
            } catch (t: Throwable) {
                Log.e("GENIO_TEST", "Model load failed", t)
                resultView.text = "Model load failed: ${t.message ?: t.javaClass.simpleName}"
                benchmarkView.text = "Benchmark unavailable while the model is not loaded."
                appendStatus("Model load failed. Check logcat.")
            } finally {
                setInteractionEnabled(true)
            }
        }
    }

    private fun applyModelPresentation(modelSpec: ModelSpec) {
        modelNameView.text = modelSpec.modelName
        resultSectionTitleView.text = resultSectionTitleFor(modelSpec.outputKind)
        resultSectionDescriptionView.text = resultSectionDescriptionFor(modelSpec.outputKind)
        resultView.text = resultPlaceholderFor(modelSpec.outputKind)
        benchmarkView.text = benchmarkPlaceholderFor(modelSpec)
        val promptVisible = shouldShowPromptOnlyButton(modelSpec.outputKind)
        promptInput.visibility = if (promptVisible) View.VISIBLE else View.GONE
        runPromptOnlyButton.visibility = if (promptVisible) View.VISIBLE else View.GONE
        pickImageButton.text = if (promptVisible) "Pick image and run prompt" else "Pick image"
    }

    private fun ensureSupportData(modelSpec: ModelSpec) {
        if (modelSpec.outputKind == OutputKind.CLASSIFICATION && imageNetLabels == null) {
            imageNetLabels = assets.open("ImageNetLabels.txt").bufferedReader().useLines { lines ->
                lines.filter { it.isNotBlank() }.toList()
            }
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
                modelNameView = this
                text = selectedModelSpec.modelName
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

    private fun createFieldLabel(text: String): TextView {
        return TextView(this).apply {
            this.text = text
            setTextColor(Color.parseColor("#4B5A67"))
            setTypeface(typeface, Typeface.BOLD)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            setPadding(0, 0, 0, dp(8))
        }
    }

    private fun createCardContainer(): LinearLayout {
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
        }
    }

    private fun createSectionCard(title: String, description: String): LinearLayout {
        return createCardContainer().apply {
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

    private fun setInteractionEnabled(enabled: Boolean) {
        modelSpinner.isEnabled = enabled
        pickImageButton.isEnabled = enabled
        runPromptOnlyButton.isEnabled = enabled
        promptInput.isEnabled = enabled
    }

    private fun runInferenceForSelectedImage(uri: Uri) {
        lifecycleScope.launch {
            setInteractionEnabled(false)
            try {
                appendStatus("Loading selected image...")
                val bitmap = withContext(Dispatchers.IO) {
                    loadBitmapFromUri(uri)
                }
                imageView.setImageBitmap(bitmap)

                val promptText = promptInput.text.toString()
                val modelResult = withContext(Dispatchers.Default) {
                    when {
                        selectedModelSpec is SmolVlm2Spec -> runLlamaCpp(bitmap, promptText)
                        selectedModelSpec.outputKind == OutputKind.TEXT_RESPONSE -> runFastVlm(bitmap, promptText)
                        else -> runSelectedModel(bitmap)
                    }
                }

                resultView.text = formatModelResultSummary("selected image", modelResult)
                appendStatus("Inference complete for ${selectedModelSpec.modelName}.")
            } catch (t: Throwable) {
                Log.e("GENIO_TEST", "Inference failed", t)
                resultView.text = "Inference failed: ${t.message ?: t.javaClass.simpleName}"
                appendStatus("Inference failed. Check logcat.")
            } finally {
                setInteractionEnabled(true)
            }
        }
    }

    private fun runFastVlmTextOnlyClick() {
        lifecycleScope.launch {
            setInteractionEnabled(false)
            try {
                appendStatus("Running model prompt without image...")
                val prompt = promptInput.text.toString()
                val modelResult = withContext(Dispatchers.Default) {
                    runFastVlmPromptOnly(prompt)
                }
                resultView.text = formatModelResultSummary("prompt only", modelResult)
                appendStatus("Text-only model request complete.")
            } catch (t: Throwable) {
                Log.e("GENIO_TEST", "Text-only model inference failed", t)
                resultView.text = "Text-only model inference failed: ${t.message ?: t.javaClass.simpleName}"
                appendStatus("Text-only model inference failed. Check logcat.")
            } finally {
                setInteractionEnabled(true)
            }
        }
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }
    }

    private suspend fun runLlamaCpp(bitmap: Bitmap, promptText: String): ModelExecutionResult {
        val prompt = promptText.trim().ifBlank { defaultFastVlmPrompt() }
        val response = requireLlamaCppEngine().generate(bitmap, prompt)
        return LlamaCppExecutionResult(response = response, prompt = prompt)
    }

    private suspend fun runFastVlm(bitmap: Bitmap, promptText: String): ModelExecutionResult {
        val imageFile = withContext(Dispatchers.IO) {
            writeFastVlmInputImage(bitmap)
        }
        val prompt = promptText.trim().ifBlank { defaultFastVlmPrompt() }
        val response = requireFastVlmEngine().generate(
            FastVlmRequest(
                prompt = prompt,
                imagePath = imageFile.absolutePath
            )
        )
        return TextExecutionResult(
            response = response,
            prompt = prompt,
            imageFile = imageFile
        )
    }

    private suspend fun runFastVlmPromptOnly(promptText: String): ModelExecutionResult {
        val prompt = promptText.trim().ifBlank { defaultFastVlmPrompt() }
        val response = requireFastVlmEngine().generate(
            FastVlmRequest(
                prompt = prompt,
                imagePath = null
            )
        )
        return TextExecutionResult(
            response = response,
            prompt = prompt,
            imageFile = null
        )
    }

    private fun writeFastVlmInputImage(bitmap: Bitmap): File {
        val directory = File(cacheDir, "fastvlm-input").apply { mkdirs() }
        val imageFile = File(directory, "selected-image.jpg")
        FileOutputStream(imageFile).use { output ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, output)
        }
        return imageFile
    }

    private fun runSelectedModel(bitmap: Bitmap): ModelExecutionResult {
        val modelSpec = selectedModelSpec
        val preprocess = LatencyTracker.measure {
            inputImagePreprocessor.preprocess(bitmap, modelSpec)
        }
        val inputJsonFile = inputTensorJsonStore.write(filesDir, preprocess.value, modelSpec)

        InferenceEngine(requireModelBuffer()).use { engine ->
            val inference = engine.run(preprocess.value.inputBuffer)
            return when (modelSpec.outputKind) {
                OutputKind.TEXT_RESPONSE -> error("Model text requests should not use the TFLite inference engine.")
                OutputKind.EMBEDDING -> {
                    val postprocess = LatencyTracker.measure {
                        embeddingPostprocessor.fromOutput(inference.outputBuffer)
                    }
                    val embeddingJsonFile = embeddingJsonStore.write(filesDir, postprocess.value)
                    EmbeddingExecutionResult(
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

                OutputKind.CLASSIFICATION -> {
                    val postprocess = LatencyTracker.measure {
                        classificationPostprocessor.fromOutput(inference.outputBuffer).predictions
                    }
                    ClassificationExecutionResult(
                        predictions = postprocess.value,
                        labels = requireImageNetLabels(),
                        latencyBreakdown = LatencyBreakdown(
                            preprocessMs = preprocess.durationMs,
                            inferenceMs = inference.inferenceMs,
                            postprocessMs = postprocess.durationMs
                        ),
                        inputJsonFile = inputJsonFile
                    )
                }
            }
        }
    }

    private fun formatModelResultSummary(sourceLabel: String, result: ModelExecutionResult): String {
        return when (result) {
            is LlamaCppExecutionResult -> {
                buildString {
                    append(sourceLabel)
                    append('\n')
                    append("Prompt: ")
                    append(result.prompt)
                    append('\n')
                    append(result.response.latencySummary)
                    append('\n')
                    append(result.response.text)
                }
            }
            is TextExecutionResult -> {
                buildString {
                    append(sourceLabel)
                    append('\n')
                    append("Prompt: ")
                    append(result.prompt)
                    append('\n')
                    append(result.response.latencySummary)
                    append('\n')
                    append(result.response.text)
                    if (result.response.debugSummary.isNotBlank()) {
                        append("\nDiagnostics: ")
                        append(result.response.debugSummary)
                    }
                    if (result.imageFile != null) {
                        append("\nImage file: ${result.imageFile.absolutePath}")
                    }
                }
            }

            is EmbeddingExecutionResult -> {
                formatEmbeddingSummary(
                    sourceLabel = sourceLabel,
                    embedding = result.embedding,
                    latencyBreakdown = result.latencyBreakdown
                ) + "\nInput JSON: ${result.inputJsonFile.absolutePath}" +
                    "\nEmbedding JSON: ${result.embeddingJsonFile.absolutePath}"
            }

            is ClassificationExecutionResult -> {
                formatClassificationSummary(
                    sourceLabel = sourceLabel,
                    labels = result.labels,
                    predictions = result.predictions,
                    latencyBreakdown = result.latencyBreakdown
                ) + "\nInput JSON: ${result.inputJsonFile.absolutePath}"
            }
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

    private fun resolveFastVlmModelPath(modelSpec: ModelSpec): String {
        val internalFile = File(filesDir, "models/${modelSpec.assetName}")
        if (internalFile.exists()) {
            return internalFile.absolutePath
        }

        val externalFile = getExternalFilesDir(null)?.let { File(it, "models/${modelSpec.assetName}") }
        if (externalFile?.exists() == true) {
            return externalFile.absolutePath
        }

        val downloadFile = File("/sdcard/Download/${modelSpec.assetName}")
        if (downloadFile.exists()) {
            return downloadFile.absolutePath
        }

        if (modelAssetLoader.assetExists(modelSpec.assetName)) {
            val copiedFile = modelAssetLoader.copyAssetToFile(modelSpec.assetName, internalFile)
            return copiedFile.absolutePath
        }

        error(
            "Model file not found. Package ${modelSpec.assetName} in app assets or copy it to ${internalFile.absolutePath}."
        )
    }

    private fun closeFastVlmEngine() {
        fastVlmEngine?.close()
        fastVlmEngine = null
    }

    private fun closeLlamaCppEngine() {
        llamaCppEngine?.close()
        llamaCppEngine = null
    }

    private fun requireFastVlmEngine(): FastVlmEngine {
        return fastVlmEngine ?: error("Model engine is not initialized.")
    }

    private fun requireLlamaCppEngine(): LlamaCppEngine {
        return llamaCppEngine ?: error("LlamaCpp engine is not initialized.")
    }

    private fun resolveModelFilePath(fileName: String): String {
        val internalFile = File(filesDir, "models/$fileName")
        if (internalFile.exists()) return internalFile.absolutePath

        val externalFile = getExternalFilesDir(null)?.let { File(it, "models/$fileName") }
        if (externalFile?.exists() == true) return externalFile.absolutePath

        val downloadFile = File("/sdcard/Download/$fileName")
        if (downloadFile.exists()) return downloadFile.absolutePath

        error("Model file not found: $fileName. Copy it to ${internalFile.absolutePath}")
    }

    private fun requireModelBuffer(): MappedByteBuffer {
        return modelBuffer ?: error("TFLite model is not initialized.")
    }

    private fun requireImageNetLabels(): List<String> {
        return imageNetLabels ?: error("ImageNet labels are not initialized.")
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

private sealed interface ModelExecutionResult

private data class LlamaCppExecutionResult(
    val response: com.example.genionputtest.llamacpp.LlamaCppResponse,
    val prompt: String
) : ModelExecutionResult

private data class TextExecutionResult(
    val response: FastVlmResponse,
    val prompt: String,
    val imageFile: File?
) : ModelExecutionResult

private data class EmbeddingExecutionResult(
    val embedding: EmbeddingOutput,
    val latencyBreakdown: LatencyBreakdown,
    val inputJsonFile: File,
    val embeddingJsonFile: File
) : ModelExecutionResult

private data class ClassificationExecutionResult(
    val predictions: List<Prediction>,
    val labels: List<String>,
    val latencyBreakdown: LatencyBreakdown,
    val inputJsonFile: File
) : ModelExecutionResult
