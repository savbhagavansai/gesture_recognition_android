package com.gesture.recognition

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    
    private val TAG = "MainActivity"
    
    // UI Components
    private lateinit var previewView: PreviewView
    private lateinit var gestureTextView: TextView
    private lateinit var confidenceTextView: TextView
    private lateinit var fpsTextView: TextView
    private lateinit var statusTextView: TextView
    private lateinit var overlayView: View
    
    // Camera
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraExecutor: ExecutorService? = null
    private var imageAnalyzer: ImageAnalysis? = null
    
    // Gesture Recognition
    private var gestureRecognizer: GestureRecognizer? = null
    
    // FPS tracking
    private val fpsBuffer = mutableListOf<Long>()
    private var lastFrameTime = System.currentTimeMillis()
    
    // Permission launcher
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
            finish()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize UI
        initializeViews()
        
        // Initialize gesture recognizer
        try {
            gestureRecognizer = GestureRecognizer(this)
            Log.d(TAG, "GestureRecognizer initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize GestureRecognizer", e)
            Toast.makeText(this, "Failed to load model: ${e.message}", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        
        // Initialize camera executor
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        // Check camera permission
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                startCamera()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }
    
    private fun initializeViews() {
        previewView = findViewById(R.id.previewView)
        gestureTextView = findViewById(R.id.gestureTextView)
        confidenceTextView = findViewById(R.id.confidenceTextView)
        fpsTextView = findViewById(R.id.fpsTextView)
        statusTextView = findViewById(R.id.statusTextView)
        overlayView = findViewById(R.id.overlayView)
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
            } catch (e: Exception) {
                Log.e(TAG, "Camera initialization failed", e)
                Toast.makeText(this, "Camera error: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return
        
        // Unbind all first
        provider.unbindAll()
        
        // Preview use case
        val preview = Preview.Builder()
            .setTargetResolution(android.util.Size(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT))
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
        
        // Image analysis use case
        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor!!, ImageAnalyzer())
            }
        
        // Select back camera
        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        
        try {
            // Bind use cases to lifecycle
            provider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )
            
            Log.d(TAG, "Camera use cases bound successfully")
            
        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failed", e)
            Toast.makeText(this, "Camera binding error: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }
    
    /**
     * Image analyzer for frame-by-frame processing
     */
    private inner class ImageAnalyzer : ImageAnalysis.Analyzer {
        
        override fun analyze(imageProxy: ImageProxy) {
            val currentTime = System.currentTimeMillis()
            
            try {
                // Convert ImageProxy to Bitmap
                val bitmap = imageProxyToBitmap(imageProxy)
                
                if (bitmap != null) {
                    // Process frame
                    lifecycleScope.launch(Dispatchers.Default) {
                        val result = gestureRecognizer?.processFrame(bitmap)
                        
                        // Calculate FPS
                        val fps = calculateFPS(currentTime)
                        
                        // Update UI
                        withContext(Dispatchers.Main) {
                            updateUI(result, fps)
                        }
                    }
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Frame analysis error", e)
            } finally {
                imageProxy.close()
            }
        }
        
        private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
            val buffer: ByteBuffer = imageProxy.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            
            return try {
                val bitmap = Bitmap.createBitmap(
                    imageProxy.width,
                    imageProxy.height,
                    Bitmap.Config.ARGB_8888
                )
                
                // Convert YUV to RGB (simplified - you may need proper conversion)
                // For now, using a simple grayscale conversion
                val pixels = IntArray(imageProxy.width * imageProxy.height)
                for (i in pixels.indices) {
                    val y = bytes[i].toInt() and 0xff
                    pixels[i] = Color.rgb(y, y, y)
                }
                bitmap.setPixels(pixels, 0, imageProxy.width, 0, 0, imageProxy.width, imageProxy.height)
                bitmap
                
            } catch (e: Exception) {
                Log.e(TAG, "Bitmap conversion error", e)
                null
            }
        }
    }
    
    private fun calculateFPS(currentTime: Long): Float {
        val elapsed = currentTime - lastFrameTime
        lastFrameTime = currentTime
        
        if (elapsed > 0) {
            val fps = 1000f / elapsed
            fpsBuffer.add(fps.toLong())
            
            if (fpsBuffer.size > Config.FPS_BUFFER_SIZE) {
                fpsBuffer.removeAt(0)
            }
        }
        
        return if (fpsBuffer.isNotEmpty()) {
            fpsBuffer.average().toFloat()
        } else {
            0f
        }
    }
    
    private fun updateUI(result: GestureResult?, fps: Float) {
        if (result == null) {
            gestureTextView.text = "No hand detected"
            gestureTextView.setTextColor(Color.GRAY)
            confidenceTextView.text = ""
            statusTextView.text = "Waiting for hand..."
            return
        }
        
        // Update gesture
        if (result.meetsThreshold()) {
            gestureTextView.text = result.getFormattedGesture()
            gestureTextView.setTextColor(
                if (result.confidence > 0.8f) Color.GREEN
                else if (result.confidence > 0.6f) Color.YELLOW
                else Color.ORANGE
            )
            
            confidenceTextView.text = String.format("%.1f%%", result.confidence * 100)
            statusTextView.text = if (result.isStable) "✓ Stable" else "Processing..."
            
        } else {
            if (result.bufferProgress < 1f) {
                gestureTextView.text = "Collecting frames..."
                val progress = (result.bufferProgress * 100).toInt()
                confidenceTextView.text = "$progress%"
            } else {
                gestureTextView.text = "Low confidence"
                confidenceTextView.text = String.format("%.1f%%", result.confidence * 100)
            }
            gestureTextView.setTextColor(Color.GRAY)
            statusTextView.text = "Move hand clearly"
        }
        
        // Update FPS
        fpsTextView.text = String.format("FPS: %.1f", fps)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        
        // Cleanup
        cameraExecutor?.shutdown()
        gestureRecognizer?.close()
        cameraProvider?.unbindAll()
        
        Log.d(TAG, "MainActivity destroyed")
    }
}
