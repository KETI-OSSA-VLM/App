package com.example.genionputtest

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var pickImageButton: Button
    private lateinit var statusView: TextView
    private lateinit var resultView: TextView
    private lateinit var benchmarkView: TextView
    private lateinit var modelBuffer: MappedByteBuffer
    private var labels: List<String> = emptyList()

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) {
            appendStatus("Image selection canceled.")
            return@registerForActivityResult
        }
        classifySelectedImage(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(createContentView())

        lifecycleScope.launch {
            val modelName = "mobilenet_v1_1.0_224_quant.tflite"
            appendStatus("Loading model...")
            modelBuffer = withContext(Dispatchers.IO) {
                loadModelMapped(modelName)
            }
            labels = withContext(Dispatchers.IO) {
                loadLabels("ImageNetLabels.txt")
            }
            appendStatus("Model loaded: $modelName")
            appendStatus("Loaded ${labels.size} labels.")
            appendStatus("Pick an image from the device to run classification.")

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
            text = "Preparing image classification..."
        }

        resultView = TextView(this).apply {
            textSize = 16f
            text = "Classification results will appear here."
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

    private fun classifySelectedImage(uri: Uri) {
        lifecycleScope.launch {
            try {
                appendStatus("Loading selected image...")
                val bitmap = withContext(Dispatchers.IO) {
                    loadBitmapFromUri(uri)
                }
                imageView.setImageBitmap(bitmap)

                appendStatus("Running classification for selected image...")
                val classification = withContext(Dispatchers.Default) {
                    runClassification(modelBuffer, bitmap)
                }

                resultView.text = formatClassificationSummary(
                    sourceLabel = "selected image",
                    inferenceMs = classification.inferenceMs,
                    labels = labels,
                    predictions = classification.predictions
                )
                appendStatus("Classification complete.")
            } catch (t: Throwable) {
                Log.e("GENIO_TEST", "Image classification failed", t)
                resultView.text = "Classification failed: ${t.javaClass.simpleName}"
                appendStatus("Classification failed. Check logcat.")
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

    private fun loadModelMapped(assetName: String): MappedByteBuffer {
        assets.openFd(assetName).use { afd ->
            FileInputStream(afd.fileDescriptor).use { fis ->
                val fc = fis.channel
                return fc.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            }
        }
    }

    private fun loadLabels(assetName: String): List<String> {
        return assets.open(assetName).bufferedReader().use { reader ->
            reader.readLines()
        }
    }

    private fun runClassification(modelBuffer: MappedByteBuffer, bitmap: Bitmap): ClassificationResult {
        val interpreter = Interpreter(modelBuffer)
        try {
            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)
            val input = bitmapToModelInput(bitmap, inputTensor.dataType(), inputTensor.shape())
            val output = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())

            val t0 = SystemClock.elapsedRealtimeNanos()
            interpreter.run(input, output)
            val t1 = SystemClock.elapsedRealtimeNanos()

            return ClassificationResult(
                predictions = extractTopClasses(output, topK = 3),
                inferenceMs = (t1 - t0) / 1_000_000.0
            )
        } finally {
            interpreter.close()
        }
    }

    private fun bitmapToModelInput(bitmap: Bitmap, type: DataType, shape: IntArray): ByteBuffer {
        require(type == DataType.UINT8) { "Expected UINT8 model input, got $type" }
        val height = shape[1]
        val width = shape[2]
        val scaled = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val readableBitmap = if (scaled.config == Bitmap.Config.HARDWARE) {
            scaled.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            scaled
        }
        val pixels = IntArray(width * height)
        readableBitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val input = ByteBuffer.allocateDirect(width * height * 3).order(ByteOrder.nativeOrder())
        for (pixel in pixels) {
            input.put(((pixel shr 16) and 0xFF).toByte())
            input.put(((pixel shr 8) and 0xFF).toByte())
            input.put((pixel and 0xFF).toByte())
        }
        input.rewind()
        return input
    }

    private fun runBenchmark(tag: String, modelBuffer: MappedByteBuffer, useNnapi: Boolean): String {
        val options = Interpreter.Options()
        var nnApiDelegate: NnApiDelegate? = null
        if (useNnapi) {
            nnApiDelegate = NnApiDelegate()
            options.addDelegate(nnApiDelegate)
        }

        val interpreter = Interpreter(modelBuffer, options)
        try {
            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)

            Log.i("GENIO_TEST", "[$tag] input shape=${inputTensor.shape().contentToString()} type=${inputTensor.dataType()}")
            Log.i("GENIO_TEST", "[$tag] output shape=${outputTensor.shape().contentToString()} type=${outputTensor.dataType()}")

            val input = allocAndFillInput(inputTensor.dataType(), inputTensor.numBytes())
            val output = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())

            runInferenceIterations(iterations = 5, input = input, output = output) { inBuf, outBuf ->
                interpreter.run(inBuf, outBuf)
            }

            val runs = 50
            val t0 = SystemClock.elapsedRealtimeNanos()
            runInferenceIterations(iterations = runs, input = input, output = output) { inBuf, outBuf ->
                interpreter.run(inBuf, outBuf)
            }
            val t1 = SystemClock.elapsedRealtimeNanos()

            val avgMs = (t1 - t0) / 1_000_000.0 / runs
            val result = formatBenchmarkStatus(tag = tag, avgMs = avgMs, runs = runs)
            Log.i("GENIO_TEST", result)
            return result
        } finally {
            interpreter.close()
            nnApiDelegate?.close()
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

    private fun allocAndFillInput(type: DataType, numBytes: Int): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(numBytes).order(ByteOrder.nativeOrder())
        when (type) {
            DataType.UINT8 -> repeat(numBytes) { buf.put(128.toByte()) }
            DataType.INT8 -> repeat(numBytes) { buf.put(0.toByte()) }
            else -> repeat(numBytes) { buf.put(0.toByte()) }
        }
        buf.rewind()
        return buf
    }
}

private data class ClassificationResult(
    val predictions: List<Prediction>,
    val inferenceMs: Double
)
