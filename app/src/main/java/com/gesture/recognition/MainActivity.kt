package com.gesture.recognition

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
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
import java.io.ByteArrayOutputStream
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

    // Camera
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraExecutor: ExecutorService? = null

    // Gesture Recognition
    private var gestureRecognizer: GestureRecognizer? = null

    // FPS tracking
    private val fpsBuffer = mutableListOf<Long>()
    private var lastFrameTime = System.currentTimeMillis()
    private var frameCount = 0

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

        val preview = Preview.Builder()
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

        val imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor!!) { imageProxy ->
                    processImageProxy(imageProxy)
                }
            }

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            provider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            Log.d(TAG, "Camera bound successfully")

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
                    val result = gestureRecognizer?.processFrame(bitmap)
                    val fps = calculateFPS(currentTime)

                    withContext(Dispatchers.Main) {
                        updateUI(result, fps)
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
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
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

    private fun updateUI(result: GestureResult?, fps: Float) {
        if (result == null) {
            gestureTextView.text = "No hand detected"
            gestureTextView.setTextColor(Color.GRAY)
            confidenceTextView.text = ""
            statusTextView.text = "Waiting for hand..."
            fpsTextView.text = String.format("FPS: %.1f", fps)
            return
        }

        if (result.meetsThreshold()) {
            gestureTextView.text = result.getFormattedGesture()
            gestureTextView.setTextColor(
                if (result.confidence > 0.8f) Color.GREEN
                else if (result.confidence > 0.6f) Color.YELLOW
                else Color.rgb(255, 165, 0)
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

        fpsTextView.text = String.format("FPS: %.1f | Frame: %d", fps, frameCount)
    }

    override fun onDestroy() {
        super.onDestroy()

        cameraExecutor?.shutdown()
        gestureRecognizer?.close()
        cameraProvider?.unbindAll()

        Log.d(TAG, "MainActivity destroyed")
    }
}