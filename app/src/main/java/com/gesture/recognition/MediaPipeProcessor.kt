package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.core.BaseOptions

/**
 * MediaPipe hand landmark processor with rotation and mirroring support
 * Handles coordinate transformation for different camera orientations
 */
class MediaPipeProcessor(context: Context) {

    private val TAG = "MediaPipeProcessor"

    private var handLandmarker: HandLandmarker? = null

    init {
        try {
            // Create HandLandmarker options
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setMinHandDetectionConfidence(Config.MP_HANDS_CONFIDENCE)
                .setMinTrackingConfidence(Config.MP_HANDS_TRACKING_CONFIDENCE)
                .setNumHands(2)  // Detect up to 2 hands
                .build()

            // Create HandLandmarker
            handLandmarker = HandLandmarker.createFromOptions(context, options)

            Log.d(TAG, "MediaPipe HandLandmarker initialized")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize MediaPipe", e)
            throw RuntimeException("Failed to initialize MediaPipe: ${e.message}", e)
        }
    }

    /**
     * Extract hand landmarks from bitmap with rotation and mirroring
     *
     * @param bitmap Input image
     * @param rotation Image rotation in degrees (0, 90, 180, 270)
     * @param mirrorHorizontal Whether to mirror horizontally (for front camera)
     * @return FloatArray of 63 values (21 landmarks × 3 coords) or null if no hand detected
     */
    fun extractLandmarks(
        bitmap: Bitmap,
        rotation: Int = 0,
        mirrorHorizontal: Boolean = false
    ): FloatArray? {
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

            // Extract x, y, z coordinates with transformation
            val landmarks = FloatArray(63)
            var idx = 0

            for (landmark in handLandmarks) {
                // Get raw coordinates from MediaPipe (in sensor space)
                val rawX = landmark.x()
                val rawY = landmark.y()
                val rawZ = landmark.z()

                // Transform coordinates (sensor space → display space)
                val (transformedX, transformedY) = transformCoordinates(
                    rawX, rawY, rotation, mirrorHorizontal
                )

                landmarks[idx++] = transformedX
                landmarks[idx++] = transformedY
                landmarks[idx++] = rawZ  // Z doesn't need transformation
            }

            Log.d(TAG, "Extracted landmarks with rotation=$rotation, mirror=$mirrorHorizontal")

            return landmarks

        } catch (e: Exception) {
            Log.e(TAG, "Landmark extraction failed", e)
            return null
        }
    }

    /**
     * Transform coordinates based on rotation and mirroring
     *
     * This handles the transformation from camera sensor space to display space
     *
     * @param x Original X coordinate (0.0 to 1.0) in sensor space
     * @param y Original Y coordinate (0.0 to 1.0) in sensor space
     * @param rotation Rotation in degrees (0, 90, 180, 270)
     * @param mirror Whether to mirror horizontally (after rotation)
     * @return Transformed (x, y) coordinates in display space
     */
    private fun transformCoordinates(
        x: Float,
        y: Float,
        rotation: Int,
        mirror: Boolean
    ): Pair<Float, Float> {
        var newX = x
        var newY = y

        // Step 1: Apply rotation (sensor space → display space)
        // Most Android devices have camera sensor in landscape, but display in portrait
        when (rotation) {
            90 -> {
                // 90° COUNTER-CLOCKWISE rotation (anti-clockwise)
                // Sensor X → Display (1-Y)
                // Sensor Y → Display X
                newX = 1.0f - y
                newY = x
            }
            180 -> {
                // 180° rotation
                // Invert both axes
                newX = 1.0f - x
                newY = 1.0f - y
            }
            270 -> {
                // 270° counter-clockwise (or 90° clockwise)
                // Sensor X → Display Y
                // Sensor Y → Display (1-X)
                newX = y
                newY = 1.0f - x
            }
            0 -> {
                // No rotation needed
                newX = x
                newY = y
            }
            else -> {
                Log.w(TAG, "Unsupported rotation: $rotation degrees. Using no rotation.")
                newX = x
                newY = y
            }
        }

        // Step 2: Apply mirroring (AFTER rotation, in display space)
        // Front camera typically shows mirror image to match user expectation
        if (mirror) {
            newX = 1.0f - newX
        }

        return Pair(newX, newY)
    }

    /**
     * Extract landmarks with additional metadata
     *
     * @param bitmap Input image
     * @param rotation Image rotation in degrees (0, 90, 180, 270)
     * @param mirrorHorizontal Whether to mirror horizontally (for front camera)
     * @return Triple of (landmarks, handedness, confidence) or null
     */
    fun extractLandmarksWithMetadata(
        bitmap: Bitmap,
        rotation: Int = 0,
        mirrorHorizontal: Boolean = false
    ): Triple<FloatArray, String, Float>? {
        val landmarker = handLandmarker ?: return null

        try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = landmarker.detect(mpImage)

            if (result.landmarks().isEmpty()) {
                return null
            }

            // Extract landmarks with transformation
            val handLandmarks = result.landmarks()[0]
            val landmarks = FloatArray(63)
            var idx = 0

            for (landmark in handLandmarks) {
                val rawX = landmark.x()
                val rawY = landmark.y()
                val rawZ = landmark.z()

                val (transformedX, transformedY) = transformCoordinates(
                    rawX, rawY, rotation, mirrorHorizontal
                )

                landmarks[idx++] = transformedX
                landmarks[idx++] = transformedY
                landmarks[idx++] = rawZ
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