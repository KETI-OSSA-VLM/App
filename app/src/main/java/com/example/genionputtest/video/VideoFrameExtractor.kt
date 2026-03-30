package com.example.genionputtest.video

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext

import java.util.concurrent.Executors

fun interface FrameCallback {
    fun onFrame(bitmap: Bitmap)
}

fun interface SuspendFrameCallback {
    suspend fun onFrame(bitmap: Bitmap)
}

class VideoFileFrameExtractor(
    private val context: Context,
    private val intervalMs: Long = 500L
) {
    suspend fun extract(uri: Uri, callback: SuspendFrameCallback) = withContext(Dispatchers.IO) {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)
            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val durationMs = durationStr?.toLongOrNull() ?: 0L
            var timeUs = 0L
            val endUs = durationMs * 1000L
            val stepUs = intervalMs * 1000L
            while (timeUs <= endUs) {
                coroutineContext.ensureActive()
                val bitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                if (bitmap != null) {
                    callback.onFrame(bitmap)
                }
                timeUs += stepUs
            }
        } finally {
            retriever.release()
        }
    }
}

class CameraFrameStream(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView
) {
    private var cameraProvider: ProcessCameraProvider? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    fun start(callback: FrameCallback) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()
            cameraProvider = provider

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(analysisExecutor) { imageProxy: ImageProxy ->
                val bitmap = imageProxy.toBitmap()
                imageProxy.close()
                callback.onFrame(bitmap)
            }

            provider.unbindAll()
            provider.bindToLifecycle(
                lifecycleOwner,
                androidx.camera.core.CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis
            )
        }, androidx.core.content.ContextCompat.getMainExecutor(context))
    }

    fun stop() {
        cameraProvider?.unbindAll()
        analysisExecutor.shutdown()
    }
}
