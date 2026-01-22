package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log

/**
 * Main gesture recognition orchestrator
 * Combines MediaPipe, normalization, buffering, and ONNX inference
 */
class GestureRecognizer(context: Context) {
    
    private val TAG = "GestureRecognizer"
    
    // Core components
    private val mediaPipeProcessor: MediaPipeProcessor
    private val onnxInference: ONNXInference
    private val sequenceBuffer: SequenceBuffer
    private val predictionSmoother: PredictionSmoother
    
    // State tracking
    private var frameCount = 0
    private var lastGesture: String? = null
    private var lastConfidence: Float = 0f
    
    init {
        Log.d(TAG, "Initializing GestureRecognizer...")
        
        mediaPipeProcessor = MediaPipeProcessor(context)
        onnxInference = ONNXInference(context)
        sequenceBuffer = SequenceBuffer()
        predictionSmoother = PredictionSmoother()
        
        Log.d(TAG, "GestureRecognizer initialized successfully")
    }
    
    /**
     * Process a single frame
     * 
     * @param bitmap Input frame
     * @return GestureResult or null if no prediction
     */
    fun processFrame(bitmap: Bitmap): GestureResult? {
        frameCount++
        
        // Step 1: Extract landmarks
        val landmarks = mediaPipeProcessor.extractLandmarks(bitmap)
        
        if (landmarks == null) {
            // No hand detected - clear buffer
            if (sequenceBuffer.size() > 0) {
                sequenceBuffer.clear()
                predictionSmoother.clear()
                Log.d(TAG, "Hand lost - buffers cleared")
            }
            return null
        }
        
        // Step 2: Normalize landmarks
        val normalized = LandmarkNormalizer.normalize(landmarks)
        
        // Step 3: Add to sequence buffer
        sequenceBuffer.add(normalized)
        
        // Step 4: Run inference if buffer is full
        if (!sequenceBuffer.isFull()) {
            return GestureResult(
                gesture = "Collecting frames...",
                confidence = 0f,
                allProbabilities = FloatArray(Config.NUM_CLASSES),
                handDetected = true,
                bufferProgress = sequenceBuffer.size().toFloat() / Config.SEQUENCE_LENGTH.toFloat()
            )
        }
        
        val sequence = sequenceBuffer.getSequence() ?: return null
        
        // Step 5: Run inference
        val prediction = onnxInference.predictWithConfidence(sequence) ?: return null
        
        val (gestureName, confidence, probabilities) = prediction
        
        // Step 6: Apply smoothing
        val gestureIdx = Config.LABEL_TO_IDX[gestureName] ?: 0
        predictionSmoother.addPrediction(gestureIdx)
        
        val smoothedIdx = predictionSmoother.getSmoothedPrediction()
        val smoothedGesture = if (smoothedIdx != null) {
            Config.IDX_TO_LABEL[smoothedIdx] ?: gestureName
        } else {
            gestureName
        }
        
        // Update state
        lastGesture = smoothedGesture
        lastConfidence = confidence
        
        return GestureResult(
            gesture = smoothedGesture,
            confidence = confidence,
            allProbabilities = probabilities,
            handDetected = true,
            bufferProgress = 1f,
            isStable = predictionSmoother.isStable()
        )
    }
    
    /**
     * Reset all buffers
     */
    fun reset() {
        sequenceBuffer.clear()
        predictionSmoother.clear()
        frameCount = 0
        lastGesture = null
        lastConfidence = 0f
        Log.d(TAG, "GestureRecognizer reset")
    }
    
    /**
     * Get current state info
     */
    fun getStateInfo(): String {
        return """
            Frame: $frameCount
            Buffer: ${sequenceBuffer.size()}/${Config.SEQUENCE_LENGTH}
            Last Gesture: ${lastGesture ?: "None"}
            Confidence: ${String.format("%.1f%%", lastConfidence * 100)}
            Stable: ${predictionSmoother.isStable()}
        """.trimIndent()
    }
    
    /**
     * Release resources
     */
    fun close() {
        try {
            mediaPipeProcessor.close()
            onnxInference.close()
            Log.d(TAG, "GestureRecognizer resources released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing resources", e)
        }
    }
}

/**
 * Result of gesture recognition
 */
data class GestureResult(
    val gesture: String,
    val confidence: Float,
    val allProbabilities: FloatArray,
    val handDetected: Boolean,
    val bufferProgress: Float = 1f,
    val isStable: Boolean = false
) {
    
    /**
     * Get formatted gesture name
     */
    fun getFormattedGesture(): String {
        return gesture.replace('_', ' ')
            .split(' ')
            .joinToString(" ") { it.capitalize() }
    }
    
    /**
     * Get confidence percentage
     */
    fun getConfidencePercent(): Int {
        return (confidence * 100).toInt()
    }
    
    /**
     * Check if prediction meets threshold
     */
    fun meetsThreshold(threshold: Float = Config.CONFIDENCE_THRESHOLD): Boolean {
        return confidence >= threshold
    }
    
    /**
     * Get top N predictions
     */
    fun getTopPredictions(n: Int = 3): List<Pair<String, Float>> {
        val predictions = allProbabilities.mapIndexed { idx, prob ->
            val label = Config.IDX_TO_LABEL[idx] ?: "unknown_$idx"
            label to prob
        }
        return predictions.sortedByDescending { it.second }.take(n)
    }
    
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as GestureResult

        if (gesture != other.gesture) return false
        if (confidence != other.confidence) return false
        if (!allProbabilities.contentEquals(other.allProbabilities)) return false
        if (handDetected != other.handDetected) return false

        return true
    }

    override fun hashCode(): Int {
        var result = gesture.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + allProbabilities.contentHashCode()
        result = 31 * result + handDetected.hashCode()
        return result
    }
}
