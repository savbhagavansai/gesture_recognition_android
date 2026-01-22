package com.gesture.recognition

import kotlin.math.max

/**
 * Normalizes hand landmarks exactly as in Python preprocessing
 * 
 * Normalization Steps (matching Python):
 * 1. Wrist-relative normalization (position invariance)
 * 2. Scale by hand size (size invariance)
 * 3. Outlier clipping
 * 4. Flatten to 63 features
 */
object LandmarkNormalizer {
    
    /**
     * Normalize hand landmarks
     * 
     * @param landmarks 21 landmarks × 3 coordinates (x, y, z) = 63 values
     * @param wristIndex Index of wrist landmark (default: 0)
     * @return Normalized and flattened array of 63 values
     */
    fun normalize(landmarks: FloatArray, wristIndex: Int = 0): FloatArray {
        require(landmarks.size == Config.NUM_FEATURES) {
            "Expected ${Config.NUM_FEATURES} features, got ${landmarks.size}"
        }
        
        // Reshape to 21 landmarks × 3 coordinates
        val landmarksReshaped = FloatArray(63)
        System.arraycopy(landmarks, 0, landmarksReshaped, 0, 63)
        
        // Step 1: Extract wrist position
        val wristX = landmarks[wristIndex * 3]
        val wristY = landmarks[wristIndex * 3 + 1]
        val wristZ = landmarks[wristIndex * 3 + 2]
        
        // Step 2: Make wrist-relative (position invariance)
        val relative = FloatArray(63)
        for (i in 0 until 21) {
            val idx = i * 3
            relative[idx] = landmarks[idx] - wristX
            relative[idx + 1] = landmarks[idx + 1] - wristY
            relative[idx + 2] = landmarks[idx + 2] - wristZ
        }
        
        // Step 3: Calculate hand scale for size normalization
        var xMin = Float.MAX_VALUE
        var xMax = Float.MIN_VALUE
        var yMin = Float.MAX_VALUE
        var yMax = Float.MIN_VALUE
        
        for (i in 0 until 21) {
            val x = relative[i * 3]
            val y = relative[i * 3 + 1]
            
            if (x < xMin) xMin = x
            if (x > xMax) xMax = x
            if (y < yMin) yMin = y
            if (y > yMax) yMax = y
        }
        
        val xRange = xMax - xMin
        val yRange = yMax - yMin
        
        // Use larger dimension as scale (matching Python)
        var handScale = max(xRange, yRange)
        
        // Safety check: avoid division by zero
        if (handScale < Config.MIN_HAND_SCALE) {
            handScale = Config.MIN_HAND_SCALE
        }
        
        // Step 4: Scale normalization (including Z-axis)
        val scaled = FloatArray(63)
        for (i in 0 until 63) {
            scaled[i] = relative[i] / handScale
        }
        
        // Step 5: Outlier clipping
        val clipped = FloatArray(63)
        for (i in 0 until 63) {
            clipped[i] = scaled[i].coerceIn(
                -Config.NORMALIZATION_CLIP_RANGE,
                Config.NORMALIZATION_CLIP_RANGE
            )
        }
        
        return clipped
    }
    
    /**
     * Normalize a batch of landmarks
     * 
     * @param landmarksBatch List of landmark arrays
     * @return List of normalized landmark arrays
     */
    fun normalizeBatch(landmarksBatch: List<FloatArray>): List<FloatArray> {
        return landmarksBatch.map { normalize(it) }
    }
}
