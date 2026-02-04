package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log

/**
 * Gesture recognizer with rolling buffer, continuous predictions, and rotation support
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
     * - Uses RAW landmarks (matching Python training data)
     *
     * @param bitmap Input frame
     */
    fun processFrame(bitmap: Bitmap): GestureResult? {
        frameCount++
        val t0 = System.nanoTime() // High precision timing

        try {
            // Step 1: Extract RAW landmarks (no transformation - matching Python!)
            val landmarks = mediaPipeProcessor.extractLandmarks(bitmap)
            val t1 = System.nanoTime()
            val mediaPipeMs = (t1 - t0) / 1_000_000.0 // Convert nanoseconds to milliseconds

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
                    bufferProgress = 0f,
                    mediaPipeTimeMs = mediaPipeMs,
                    onnxTimeMs = 0.0,
                    totalTimeMs = mediaPipeMs
                )
            }

            // Hand detected - reset missed frame counter
            missedFrameCount = 0
            lastLandmarks = landmarks

            // Step 2: Normalize landmarks
            val normalized = LandmarkNormalizer.normalize(landmarks)
            val t2 = System.nanoTime()

            // DEBUG: Log first frame of normalized data
            if (frameCount % 30 == 1) {
                Log.d(TAG, "=== NORMALIZATION DEBUG (Frame $frameCount) ===")
                Log.d(TAG, "Raw landmark[0] (wrist): [${landmarks[0]}, ${landmarks[1]}, ${landmarks[2]}]")
                Log.d(TAG, "Normalized[0-2]: [${normalized[0]}, ${normalized[1]}, ${normalized[2]}]")
                Log.d(TAG, "Normalized min/max: ${normalized.minOrNull()} / ${normalized.maxOrNull()}")
            }

            // Step 3: Add to rolling buffer
            sequenceBuffer.add(normalized)

            // Step 4: Check if buffer is ready for prediction
            val currentBufferSize = sequenceBuffer.size()

            if (currentBufferSize < Config.SEQUENCE_LENGTH) {
                val totalMs = (t2 - t0) / 1_000_000.0
                // Still collecting frames
                return GestureResult(
                    gesture = "Collecting frames...",
                    confidence = 0f,
                    allProbabilities = FloatArray(Config.NUM_CLASSES),
                    handDetected = true,
                    bufferProgress = currentBufferSize.toFloat() / Config.SEQUENCE_LENGTH.toFloat(),
                    mediaPipeTimeMs = mediaPipeMs,
                    onnxTimeMs = 0.0,
                    totalTimeMs = totalMs
                )
            }

            // Step 5: Buffer is full - run prediction EVERY FRAME (continuous)
            val sequence = sequenceBuffer.getSequence() ?: return null
            val t3 = System.nanoTime()

            // Try to predict (returns null if confidence < threshold)
            val prediction = onnxInference.predictWithConfidence(sequence)
            val t4 = System.nanoTime()
            val onnxMs = (t4 - t3) / 1_000_000.0
            val totalMs = (t4 - t0) / 1_000_000.0

            // Log detailed timing every 30 frames
            if (frameCount % 30 == 0) {
                Log.d(TAG, "⏱️ MediaPipe: ${String.format("%.1f", mediaPipeMs)}ms | ONNX: ${String.format("%.1f", onnxMs)}ms | Total: ${String.format("%.1f", totalMs)}ms")
            }

            if (prediction == null) {
                // Low confidence - still show some info
                // Try prediction without threshold to get probabilities
                val rawPrediction = onnxInference.predict(sequence)

                if (rawPrediction != null) {
                    val (predictedIdx, probabilities) = rawPrediction
                    val gestureName = Config.IDX_TO_LABEL[predictedIdx] ?: "unknown"
                    val confidence = probabilities[predictedIdx]

                    return GestureResult(
                        gesture = gestureName,
                        confidence = confidence,
                        allProbabilities = probabilities,
                        handDetected = true,
                        bufferProgress = 1f,
                        isStable = false,
                        mediaPipeTimeMs = mediaPipeMs,
                        onnxTimeMs = onnxMs,
                        totalTimeMs = totalMs
                    )
                } else {
                    Log.w(TAG, "Raw prediction returned null")
                    return GestureResult(
                        gesture = "Prediction failed",
                        confidence = 0f,
                        allProbabilities = FloatArray(Config.NUM_CLASSES),
                        handDetected = true,
                        bufferProgress = 1f,
                        mediaPipeTimeMs = mediaPipeMs,
                        onnxTimeMs = onnxMs,
                        totalTimeMs = totalMs
                    )
                }
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
                isStable = predictionSmoother.isStable(),
                mediaPipeTimeMs = mediaPipeMs,
                onnxTimeMs = onnxMs,
                totalTimeMs = totalMs
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame", e)
            e.printStackTrace()
            return GestureResult(
                gesture = "Error: ${e.message}",
                confidence = 0f,
                allProbabilities = FloatArray(Config.NUM_CLASSES),
                handDetected = false,
                bufferProgress = 0f,
                mediaPipeTimeMs = 0.0,
                onnxTimeMs = 0.0,
                totalTimeMs = 0.0
            )
        }
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