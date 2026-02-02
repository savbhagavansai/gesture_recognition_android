package com.gesture.recognition

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
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
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private val TAG = "MainActivity"

    // UI Components
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: GestureOverlayView

    // Camera
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraExecutor: ExecutorService? = null
    private var useFrontCamera = true  // Start with front camera for tablets

    // Gesture Recognition
    private var gestureRecognizer: GestureRecognizer? = null

    // FPS tracking
    private val fpsBuffer = mutableListOf<Long>()
    private var lastFrameTime = System.currentTimeMillis()
    private var frameCount = 0

    // Frame processing control
    private val isProcessing = AtomicBoolean(false)
    private var frameSkipCounter = 0

    // Hand tracking state (thread-safe)
    private val landmarksLock = Any()
    @Volatile private var currentLandmarks: FloatArray? = null

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
        overlayView = findViewById(R.id.overlayView)

        // Set preview scale type to match overlay
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER

        // Double tap to switch camera
        var lastTapTime = 0L
        overlayView.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                val currentTime = System.currentTimeMillis()
                if (currentTime - lastTapTime < 300) {
                    // Double tap detected
                    switchCamera()
                }
                lastTapTime = currentTime
                true
            } else {
                false
            }
        }
    }

    private fun switchCamera() {
        useFrontCamera = !useFrontCamera
        cameraProvider?.unbindAll()
        startCamera()
        Toast.makeText(
            this,
            if (useFrontCamera) "Front Camera" else "Back Camera",
            Toast.LENGTH_SHORT
        ).show()
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

        provider.unbindAll()

        // Preview - lower resolution for edge devices
        val preview = Preview.Builder()
            .setTargetResolution(android.util.Size(240, 180))  // Reduced for performance
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

        // Image analysis - match preview resolution
        val imageAnalyzer = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(240, 180))  // Reduced for performance
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor!!) { imageProxy ->
                    processImageProxy(imageProxy)
                }
            }

        // Select camera
        val cameraSelector = if (useFrontCamera) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }

        try {
            provider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            Log.d(TAG, "Camera bound: ${if (useFrontCamera) "Front" else "Back"}")

        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failed", e)
        }
    }

    private fun processImageProxy(imageProxy: ImageProxy) {
        // AGGRESSIVE FRAME DROPPING: Skip if still processing previous frame
        if (!isProcessing.compareAndSet(false, true)) {
            imageProxy.close()
            return  // Drop this frame immediately
        }

        // FRAME SKIPPING: Process every 2nd frame for better performance
        frameSkipCounter++
        if (frameSkipCounter % 2 != 0) {
            isProcessing.set(false)
            imageProxy.close()
            return
        }

        val currentTime = System.currentTimeMillis()
        frameCount++

        try {
            // Get image rotation from CameraX
            val imageRotation = imageProxy.imageInfo.rotationDegrees

            val bitmap = imageProxy.toBitmap()

            if (bitmap != null) {
                lifecycleScope.launch(Dispatchers.Default) {
                    try {
                        // Process frame with RAW landmarks (for model)
                        val result = gestureRecognizer?.processFrame(bitmap)

                        // Get landmarks with thread safety
                        val landmarks = synchronized(landmarksLock) {
                            gestureRecognizer?.getLastLandmarks()?.copyOf()
                        }
                        currentLandmarks = landmarks

                        // Calculate FPS
                        val fps = calculateFPS(currentTime)

                        // Update UI on main thread (pass rotation for display transformation)
                        withContext(Dispatchers.Main) {
                            try {
                                updateOverlay(
                                    result,
                                    fps,
                                    imageProxy.width,
                                    imageProxy.height,
                                    imageRotation,
                                    useFrontCamera
                                )
                            } catch (e: Exception) {
                                Log.e(TAG, "UI update failed", e)
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Frame processing error", e)
                    } finally {
                        isProcessing.set(false)
                    }
                }
            } else {
                isProcessing.set(false)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error", e)
            isProcessing.set(false)
        } finally {
            imageProxy.close()
        }
    }

    private fun ImageProxy.toBitmap(): Bitmap? {
        return try {
            // Optimized conversion with lower quality for speed
            val yBuffer = planes[0].buffer
            val uBuffer = planes[1].buffer
            val vBuffer = planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)

            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()

            // Lower quality for faster processing
            yuvImage.compressToJpeg(Rect(0, 0, width, height), 75, out)
            val imageBytes = out.toByteArray()

            // Use RGB_565 for faster decoding
            val options = BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.RGB_565
                inSampleSize = 1
            }

            BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)
        } catch (e: Exception) {
            Log.e(TAG, "Bitmap conversion failed", e)
            null
        }
    }

    private fun calculateFPS(currentTime: Long): Float {
        val elapsed = currentTime - lastFrameTime
        lastFrameTime = currentTime

        if (elapsed > 0) {
            val fps = 1000f / elapsed
            fpsBuffer.add(fps.toLong())

            if (fpsBuffer.size > 30) {
                fpsBuffer.removeAt(0)
            }
        }

        return if (fpsBuffer.isNotEmpty()) {
            fpsBuffer.average().toFloat()
        } else {
            0f
        }
    }

    private fun updateOverlay(
        result: GestureResult?,
        fps: Float,
        imageWidth: Int,
        imageHeight: Int,
        rotation: Int,
        useFrontCamera: Boolean
    ) {
        try {
            // Thread-safe landmark access
            val landmarks = synchronized(landmarksLock) {
                currentLandmarks?.copyOf()
            }

            overlayView.updateData(
                result = result,
                landmarks = landmarks,
                fps = fps,
                frameCount = frameCount,
                bufferSize = gestureRecognizer?.getBufferSize() ?: 0,
                handDetected = landmarks != null,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
                rotation = rotation,
                mirrorHorizontal = useFrontCamera
            )
        } catch (e: Exception) {
            Log.e(TAG, "Overlay update failed", e)
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        cameraExecutor?.shutdown()
        gestureRecognizer?.close()
        cameraProvider?.unbindAll()

        Log.d(TAG, "MainActivity destroyed")
    }
}