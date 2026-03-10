package com.example.genionputtest.inference.core

import android.os.SystemClock
import com.example.genionputtest.runInferenceIterations
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer

class InferenceEngine(
    modelBuffer: MappedByteBuffer,
    options: InferenceOptions = InferenceOptions()
) : Closeable {

    private val nnApiDelegate: NnApiDelegate?
    private val interpreter: Interpreter

    init {
        val interpreterOptions = Interpreter.Options()
        nnApiDelegate = if (options.useNnapi) {
            NnApiDelegate().also(interpreterOptions::addDelegate)
        } else {
            null
        }
        interpreter = Interpreter(modelBuffer, interpreterOptions)
    }

    fun inputShape(): IntArray = interpreter.getInputTensor(0).shape()

    fun outputShape(): IntArray = interpreter.getOutputTensor(0).shape()

    fun inputDataType(): DataType = interpreter.getInputTensor(0).dataType()

    fun outputDataType(): DataType = interpreter.getOutputTensor(0).dataType()

    fun createBenchmarkInput(): ByteBuffer {
        val tensor = interpreter.getInputTensor(0)
        val buf = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
        when (tensor.dataType()) {
            DataType.UINT8 -> repeat(tensor.numBytes()) { buf.put(128.toByte()) }
            DataType.INT8 -> repeat(tensor.numBytes()) { buf.put(0.toByte()) }
            DataType.FLOAT32 -> repeat(tensor.numBytes() / 4) { buf.putFloat(0f) }
            else -> error("Unsupported benchmark input type: ${tensor.dataType()}")
        }
        buf.rewind()
        return buf
    }

    fun run(inputBuffer: ByteBuffer): InferenceResult {
        val outputTensor = interpreter.getOutputTensor(0)
        val output = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())

        val t0 = SystemClock.elapsedRealtimeNanos()
        interpreter.run(inputBuffer, output)
        val t1 = SystemClock.elapsedRealtimeNanos()

        output.rewind()
        return InferenceResult(
            outputBuffer = output,
            inferenceMs = (t1 - t0) / 1_000_000.0,
            inputShape = interpreter.getInputTensor(0).shape(),
            outputShape = outputTensor.shape(),
            inputDataType = interpreter.getInputTensor(0).dataType(),
            outputDataType = outputTensor.dataType()
        )
    }

    fun benchmark(warmupRuns: Int, runs: Int, inputBuffer: ByteBuffer): Double {
        val outputTensor = interpreter.getOutputTensor(0)
        val output = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())

        runInferenceIterations(iterations = warmupRuns, input = inputBuffer, output = output) { inBuf, outBuf ->
            interpreter.run(inBuf, outBuf)
        }

        val t0 = SystemClock.elapsedRealtimeNanos()
        runInferenceIterations(iterations = runs, input = inputBuffer, output = output) { inBuf, outBuf ->
            interpreter.run(inBuf, outBuf)
        }
        val t1 = SystemClock.elapsedRealtimeNanos()
        return (t1 - t0) / 1_000_000.0 / runs
    }

    override fun close() {
        interpreter.close()
        nnApiDelegate?.close()
    }
}
