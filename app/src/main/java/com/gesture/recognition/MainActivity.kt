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

    // Hand tracking state
    private var currentLandmarks: FloatArray? = null

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

        // Preview - 640x480 for better performance
        val preview = Preview.Builder()
            .setTargetResolution(android.util.Size(640, 480))
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

        // Image analysis
        val imageAnalyzer = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(640, 480))
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
        val currentTime = System.currentTimeMillis()
        frameCount++

        try {
            val bitmap = imageProxy.toBitmap()

            if (bitmap != null) {
                lifecycleScope.launch(Dispatchers.Default) {
                    // Process frame with mirroring for front camera
                    val result = gestureRecognizer?.processFrame(bitmap, useFrontCamera)

                    // Get landmarks for overlay
                    currentLandmarks = gestureRecognizer?.getLastLandmarks()

                    // Calculate FPS
                    val fps = calculateFPS(currentTime)

                    // Update UI
                    withContext(Dispatchers.Main) {
                        updateOverlay(result, fps, imageProxy.width, imageProxy.height)
                    }
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun ImageProxy.toBitmap(): Bitmap? {
        return try {
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

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
            val imageBytes = out.toByteArray()
            BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
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

    private fun updateOverlay(result: GestureResult?, fps: Float, imageWidth: Int, imageHeight: Int) {
        overlayView.updateData(
            result = result,
            landmarks = currentLandmarks,
            fps = fps,
            frameCount = frameCount,
            bufferSize = gestureRecognizer?.getBufferSize() ?: 0,
            handDetected = currentLandmarks != null,
            imageWidth = imageWidth,
            imageHeight = imageHeight
        )
    }

    override fun onDestroy() {
        super.onDestroy()

        cameraExecutor?.shutdown()
        gestureRecognizer?.close()
        cameraProvider?.unbindAll()

        Log.d(TAG, "MainActivity destroyed")
    }
}
