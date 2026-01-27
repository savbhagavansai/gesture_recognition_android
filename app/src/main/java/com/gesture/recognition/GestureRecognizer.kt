package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log

/**
 * Gesture recognizer with rolling buffer and continuous predictions
 * Matches Python predict.py behavior exactly
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
    private var lastLandmarks: FloatArray? = null
    private var missedFrameCount = 0
    private val maxMissedFrames = 3  // Clear buffer after 3 missed frames

    init {
        Log.d(TAG, "Initializing GestureRecognizer...")

        mediaPipeProcessor = MediaPipeProcessor(context)
        onnxInference = ONNXInference(context)
        sequenceBuffer = SequenceBuffer()
        predictionSmoother = PredictionSmoother()

        Log.d(TAG, "GestureRecognizer initialized successfully")
    }

    /**
     * Process a frame - CONTINUOUS MODE (like Python)
     * - Rolling buffer (always maintains last 15 frames)
     * - Predicts EVERY frame once buffer >= 15
     * - Only clears buffer after multiple missed frames
     */
    fun processFrame(bitmap: Bitmap): GestureResult? {
        frameCount++

        // Step 1: Extract landmarks
        val landmarks = mediaPipeProcessor.extractLandmarks(bitmap)

        if (landmarks == null) {
            // Hand not detected
            missedFrameCount++
            lastLandmarks = null

            // Only clear buffer after multiple missed frames (like Python)
            if (missedFrameCount > maxMissedFrames) {
                if (sequenceBuffer.size() > 0) {
                    sequenceBuffer.clear()
                    predictionSmoother.clear()
                    Log.d(TAG, "Buffer cleared after $missedFrameCount missed frames")
                }
            }

            return GestureResult(
                gesture = "No hand detected",
                confidence = 0f,
                allProbabilities = FloatArray(Config.NUM_CLASSES),
                handDetected = false,
                bufferProgress = 0f
            )
        }

        // Hand detected - reset missed frame counter
        missedFrameCount = 0
        lastLandmarks = landmarks

        // Step 2: Normalize landmarks
        val normalized = LandmarkNormalizer.normalize(landmarks)

        // Step 3: Add to rolling buffer
        sequenceBuffer.add(normalized)

        // Step 4: Check if buffer is ready for prediction
        val currentBufferSize = sequenceBuffer.size()

        if (currentBufferSize < Config.SEQUENCE_LENGTH) {
            // Still collecting frames
            return GestureResult(
                gesture = "Collecting frames...",
                confidence = 0f,
                allProbabilities = FloatArray(Config.NUM_CLASSES),
                handDetected = true,
                bufferProgress = currentBufferSize.toFloat() / Config.SEQUENCE_LENGTH.toFloat()
            )
        }

        // Step 5: Buffer is full - run prediction EVERY FRAME (continuous)
        val sequence = sequenceBuffer.getSequence() ?: return null

        val prediction = onnxInference.predictWithConfidence(sequence)

        if (prediction == null) {
            Log.w(TAG, "Prediction returned null")
            return GestureResult(
                gesture = "Prediction failed",
                confidence = 0f,
                allProbabilities = FloatArray(Config.NUM_CLASSES),
                handDetected = true,
                bufferProgress = 1f
            )
        }

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
     * Get last detected landmarks (for overlay drawing)
     */
    fun getLastLandmarks(): FloatArray? {
        return lastLandmarks
    }

    /**
     * Get current buffer size
     */
    fun getBufferSize(): Int {
        return sequenceBuffer.size()
    }

    /**
     * Reset recognizer
     */
    fun reset() {
        sequenceBuffer.clear()
        predictionSmoother.clear()
        frameCount = 0
        lastLandmarks = null
        missedFrameCount = 0
        Log.d(TAG, "GestureRecognizer reset")
    }

    /**
     * Get state info for debugging
     */
    fun getStateInfo(): String {
        return """
            Frame: $frameCount
            Buffer: ${sequenceBuffer.size()}/${Config.SEQUENCE_LENGTH}
            Missed frames: $missedFrameCount
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