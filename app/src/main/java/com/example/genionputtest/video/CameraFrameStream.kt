package com.example.genionputtest.video

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executors

/**
 * CameraX 기반 실시간 프레임 스트림.
 * STRATEGY_KEEP_ONLY_LATEST 로 처리 속도보다 빠른 프레임은 자동 드롭.
 * 카메라가 90도 CCW 회전 장착된 보드 특성에 맞춰 비트맵도 동일하게 회전.
 */
class CameraFrameStream(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView
) {
    private var cameraProvider: ProcessCameraProvider? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    fun start(onFrame: (Bitmap) -> Unit) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            cameraProvider = future.get()
            bindUseCases(onFrame)
        }, ContextCompat.getMainExecutor(context))
    }

    private fun bindUseCases(onFrame: (Bitmap) -> Unit) {
        val provider = cameraProvider ?: return

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        analysis.setAnalyzer(analysisExecutor) { proxy ->
            val bitmap = proxy.rotateBitmap()
            proxy.close()
            if (bitmap != null) onFrame(bitmap)
        }

        try {
            provider.unbindAll()
            provider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        } catch (e: Exception) {
            android.util.Log.e("CameraFrameStream", "bindToLifecycle failed", e)
        }
    }

    fun stop() {
        cameraProvider?.unbindAll()
        cameraProvider = null
        analysisExecutor.shutdown()
    }

    /** ImageProxy (RGBA_8888) → Bitmap 변환 후 rotationDegrees만큼 회전 */
    private fun ImageProxy.rotateBitmap(): Bitmap? {
        val plane = this.planes.firstOrNull() ?: return null
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val w = this.width
        val h = this.height

        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val rowBytes = w * pixelStride
        if (rowStride == rowBytes) {
            bmp.copyPixelsFromBuffer(buffer)
        } else {
            // row padding 있는 경우 행 단위로 복사
            val rowBuf = ByteArray(rowStride)
            val pixels = IntArray(w * h)
            for (y in 0 until h) {
                buffer.get(rowBuf, 0, rowStride)
                for (x in 0 until w) {
                    val base = x * pixelStride
                    val r = rowBuf[base].toInt() and 0xFF
                    val g = rowBuf[base + 1].toInt() and 0xFF
                    val b = rowBuf[base + 2].toInt() and 0xFF
                    val a = rowBuf[base + 3].toInt() and 0xFF
                    pixels[y * w + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            bmp.setPixels(pixels, 0, w, 0, 0, w, h)
        }

        val rotation = this.imageInfo.rotationDegrees
        if (rotation == 0) return bmp
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
            .also { if (it !== bmp) bmp.recycle() }
    }
}
