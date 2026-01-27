package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/**
 * MediaPipe Hands processor for hand landmark detection
 * Extracts 21 landmarks (x, y, z) per hand
 */
class MediaPipeProcessor(context: Context) {

    private val TAG = "MediaPipeProcessor"

    private var handLandmarker: HandLandmarker? = null

    init {
        initializeMediaPipe(context)
    }

    /**
     * Initialize MediaPipe HandLandmarker
     */
    private fun initializeMediaPipe(context: Context) {
        try {
            Log.d(TAG, "Initializing MediaPipe HandLandmarker...")

            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(1)  // Track only one hand
                .setMinHandDetectionConfidence(Config.MP_HANDS_CONFIDENCE)
                .setMinTrackingConfidence(Config.MP_HANDS_TRACKING_CONFIDENCE)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, options)

            Log.d(TAG, "MediaPipe initialized successfully")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize MediaPipe", e)
            throw RuntimeException("Failed to initialize MediaPipe: ${e.message}", e)
        }
    }

    /**
     * Extract hand landmarks from bitmap
     *
     * @param bitmap Input image
     * @param mirrorHorizontal Whether to mirror X coordinates (for front camera)
     * @return FloatArray of 63 values (21 landmarks × 3 coords) or null if no hand detected
     */
    fun extractLandmarks(bitmap: Bitmap, mirrorHorizontal: Boolean = false): FloatArray? {
        val landmarker = handLandmarker ?: run {
            Log.e(TAG, "HandLandmarker not initialized")
            return null
        }

        try {
            // Convert bitmap to MediaPipe image
            val mpImage = BitmapImageBuilder(bitmap).build()

            // Detect hands
            val result: HandLandmarkerResult = landmarker.detect(mpImage)

            // Check if hand detected
            if (result.landmarks().isEmpty()) {
                return null
            }

            // Get first hand landmarks
            val handLandmarks = result.landmarks()[0]

            if (handLandmarks.size != 21) {
                Log.w(TAG, "Expected 21 landmarks, got ${handLandmarks.size}")
                return null
            }

            // Extract x, y, z coordinates
            val landmarks = FloatArray(63)
            var idx = 0

            for (landmark in handLandmarks) {
                // Mirror X coordinate for front camera (to match display)
                val x = if (mirrorHorizontal) 1.0f - landmark.x() else landmark.x()

                landmarks[idx++] = x
                landmarks[idx++] = landmark.y()
                landmarks[idx++] = landmark.z()
            }

            return landmarks

        } catch (e: Exception) {
            Log.e(TAG, "Landmark extraction failed", e)
            return null
        }
    }

    /**
     * Extract landmarks with additional metadata
     *
     * @param bitmap Input image
     * @return Triple of (landmarks, handedness, confidence) or null
     */
    fun extractLandmarksWithMetadata(bitmap: Bitmap): Triple<FloatArray, String, Float>? {
        val landmarker = handLandmarker ?: return null

        try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = landmarker.detect(mpImage)

            if (result.landmarks().isEmpty()) {
                return null
            }

            // Extract landmarks
            val handLandmarks = result.landmarks()[0]
            val landmarks = FloatArray(63)
            var idx = 0

            for (landmark in handLandmarks) {
                landmarks[idx++] = landmark.x()
                landmarks[idx++] = landmark.y()
                landmarks[idx++] = landmark.z()
            }

            // Get handedness (Left/Right)
            val handedness = if (result.handednesses().isNotEmpty()) {
                result.handednesses()[0][0].categoryName()
            } else {
                "Unknown"
            }

            // Get confidence
            val confidence = if (result.handednesses().isNotEmpty()) {
                result.handednesses()[0][0].score()
            } else {
                0f
            }

            return Triple(landmarks, handedness, confidence)

        } catch (e: Exception) {
            Log.e(TAG, "Landmark extraction with metadata failed", e)
            return null
        }
    }

    /**
     * Check if hand is detected in image
     *
     * @param bitmap Input image
     * @return True if hand detected
     */
    fun isHandDetected(bitmap: Bitmap): Boolean {
        return extractLandmarks(bitmap) != null
    }

    /**
     * Release resources
     */
    fun close() {
        try {
            handLandmarker?.close()
            Log.d(TAG, "MediaPipe resources released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing MediaPipe resources", e)
        }
    }
}