package com.example.genionputtest.inference.core

import android.content.res.AssetManager
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ModelAssetLoader(
    private val assetManager: AssetManager
) {
    fun loadMapped(assetName: String): MappedByteBuffer {
        assetManager.openFd(assetName).use { afd ->
            FileInputStream(afd.fileDescriptor).use { fis ->
                val fc = fis.channel
                return fc.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            }
        }
    }

    fun assetExists(assetName: String): Boolean {
        return assetManager.list("")?.contains(assetName) == true
    }

    fun copyAssetToFile(assetName: String, destination: File): File {
        destination.parentFile?.mkdirs()
        assetManager.open(assetName).use { input ->
            FileOutputStream(destination).use { output ->
                input.copyTo(output)
            }
        }
        return destination
    }
}
