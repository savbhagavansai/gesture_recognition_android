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
            
            // Read model file from assets
            val modelBytes = context.assets.open(Config.ONNX_MODEL_FILENAME).use { inputStream ->
                inputStream.readBytes()
            }
            
            Log.d(TAG, "Model loaded: ${modelBytes.size} bytes")
            
            // Create ONNX session
            ortSession = ortEnvironment.createSession(modelBytes)
            
            // Get input/output names
            inputName = ortSession?.inputNames?.iterator()?.next()
            outputName = ortSession?.outputNames?.iterator()?.next()
            
            Log.d(TAG, "ONNX session created successfully")
            Log.d(TAG, "Input name: $inputName")
            Log.d(TAG, "Output name: $outputName")
            
            // Log input shape
            ortSession?.inputInfo?.get(inputName)?.info?.let { info ->
                Log.d(TAG, "Input shape: ${info}")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ONNX model", e)
            throw RuntimeException("Failed to load ONNX model: ${e.message}", e)
        }
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
    fun getInputShape(): LongArray? {
        return ortSession?.inputInfo?.get(inputName)?.info?.shape
    }
    
    /**
     * Get model output shape
     */
    fun getOutputShape(): LongArray? {
        return ortSession?.outputInfo?.get(outputName)?.info?.shape
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
