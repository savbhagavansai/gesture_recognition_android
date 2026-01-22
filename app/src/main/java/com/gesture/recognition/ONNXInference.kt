package com.gesture.recognition

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer

/**
 * ONNX Runtime inference engine
 * Loads INT8 quantized TCN model and performs gesture classification
 */
class ONNXInference(context: Context) {
    
    private val TAG = "ONNXInference"
    
    private val ortEnvironment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    
    private var inputName: String? = null
    private var outputName: String? = null
    
    init {
        loadModel(context)
    }
    
    /**
     * Load ONNX model from assets
     */
    private fun loadModel(context: Context) {
        try {
            Log.d(TAG, "Loading ONNX model: ${Config.ONNX_MODEL_FILENAME}")
            
            // Create session options
            val sessionOptions = OrtSession.SessionOptions()

            // Copy model files to cache directory (needed for external data)
            val modelFile = copyAssetToCache(context, Config.ONNX_MODEL_FILENAME)

            // Check if external data file exists and copy it
            val dataFileName = Config.ONNX_MODEL_FILENAME + ".data"
            try {
                context.assets.open(dataFileName).use {
                    Log.d(TAG, "External data file found: $dataFileName")
                    copyAssetToCache(context, dataFileName)
                }
            } catch (e: Exception) {
                Log.d(TAG, "No external data file (this is OK if model is small)")
            }

            Log.d(TAG, "Model file copied to: ${modelFile.absolutePath}")

            // Create ONNX session from file path (handles external data automatically)
            ortSession = ortEnvironment.createSession(
                modelFile.absolutePath,
                sessionOptions
            )

            // Get input/output names
            inputName = ortSession?.inputNames?.iterator()?.next()
            outputName = ortSession?.outputNames?.iterator()?.next()

            Log.d(TAG, "ONNX session created successfully")
            Log.d(TAG, "Input name: $inputName")
            Log.d(TAG, "Output name: $outputName")

            // Log input shape
            ortSession?.inputInfo?.get(inputName)?.info?.let { info ->
                Log.d(TAG, "Input info: ${info}")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ONNX model", e)
            throw RuntimeException("Failed to load ONNX model: ${e.message}", e)
        }
    }

    /**
     * Copy asset file to cache directory
     */
    private fun copyAssetToCache(context: Context, assetName: String): java.io.File {
        val cacheFile = java.io.File(context.cacheDir, assetName)

        // Copy from assets to cache
        context.assets.open(assetName).use { inputStream ->
            cacheFile.outputStream().use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        }

        Log.d(TAG, "Copied $assetName to cache (${cacheFile.length()} bytes)")
        return cacheFile
    }

    /**
     * Run inference on sequence
     *
     * @param sequence Shape: [sequence_length, num_features]
     * @return Pair of (predicted_class_index, confidence_scores)
     */
    fun predict(sequence: Array<FloatArray>): Pair<Int, FloatArray>? {
        val session = ortSession ?: run {
            Log.e(TAG, "ONNX session not initialized")
            return null
        }

        val input = inputName ?: run {
            Log.e(TAG, "Input name not found")
            return null
        }

        try {
            // Prepare input tensor
            // Expected shape: [batch=1, sequence_length=15, features=63]
            val batchSize = 1L
            val sequenceLength = Config.SEQUENCE_LENGTH.toLong()
            val numFeatures = Config.NUM_FEATURES.toLong()

            val shape = longArrayOf(batchSize, sequenceLength, numFeatures)

            // Flatten sequence to 1D array
            val flatSequence = FloatArray(Config.SEQUENCE_LENGTH * Config.NUM_FEATURES)
            var idx = 0
            for (frame in sequence) {
                for (value in frame) {
                    flatSequence[idx++] = value
                }
            }

            // Create tensor
            val inputTensor = OnnxTensor.createTensor(
                ortEnvironment,
                FloatBuffer.wrap(flatSequence),
                shape
            )

            // Run inference
            val results = session.run(mapOf(input to inputTensor))

            // Get output tensor
            val outputTensor = results[0].value as Array<*>
            val probabilities = (outputTensor[0] as FloatArray)

            // Find predicted class (argmax)
            var maxIdx = 0
            var maxProb = probabilities[0]

            for (i in 1 until probabilities.size) {
                if (probabilities[i] > maxProb) {
                    maxProb = probabilities[i]
                    maxIdx = i
                }
            }

            // Clean up
            inputTensor.close()
            results.close()

            return Pair(maxIdx, probabilities)

        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            return null
        }
    }

    /**
     * Run inference with confidence threshold
     *
     * @param sequence Shape: [sequence_length, num_features]
     * @param threshold Minimum confidence threshold
     * @return Triple of (gesture_name, confidence, all_probabilities) or null
     */
    fun predictWithConfidence(
        sequence: Array<FloatArray>,
        threshold: Float = Config.CONFIDENCE_THRESHOLD
    ): Triple<String, Float, FloatArray>? {

        val (predictedIdx, probabilities) = predict(sequence) ?: return null

        val confidence = probabilities[predictedIdx]

        // Apply confidence threshold
        if (confidence < threshold) {
            return null
        }

        val gestureName = Config.IDX_TO_LABEL[predictedIdx] ?: "unknown"

        return Triple(gestureName, confidence, probabilities)
    }

    /**
     * Get model input shape
     */
    fun getInputShape(): String? {
        return try {
            ortSession?.inputInfo?.get(inputName)?.info?.toString()
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Get model output shape
     */
    fun getOutputShape(): String? {
        return try {
            ortSession?.outputInfo?.get(outputName)?.info?.toString()
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Release resources
     */
    fun close() {
        try {
            ortSession?.close()
            Log.d(TAG, "ONNX session closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ONNX session", e)
        }
    }
}