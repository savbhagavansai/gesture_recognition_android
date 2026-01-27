package com.gesture.recognition

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * Custom overlay view matching Python predict.py UI
 * - Hand skeleton (21 landmarks + connections)
 * - Top panel (status, buffer, gesture, confidence bar)
 * - Right panel (probability bars for all 11 classes)
 * - Bottom instructions
 * - FPS and frame counter
 */
class GestureOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // Data to display
    private var result: GestureResult? = null
    private var landmarks: FloatArray? = null
    private var fps: Float = 0f
    private var frameCount: Int = 0
    private var bufferSize: Int = 0
    private var handDetected: Boolean = false
    private var imageWidth: Int = 640
    private var imageHeight: Int = 480

    // MediaPipe hand connections (21 landmarks)
    private val handConnections = listOf(
        // Thumb
        0 to 1, 1 to 2, 2 to 3, 3 to 4,
        // Index
        0 to 5, 5 to 6, 6 to 7, 7 to 8,
        // Middle
        0 to 9, 9 to 10, 10 to 11, 11 to 12,
        // Ring
        0 to 13, 13 to 14, 14 to 15, 15 to 16,
        // Pinky
        0 to 17, 17 to 18, 18 to 19, 19 to 20,
        // Palm
        5 to 9, 9 to 13, 13 to 17
    )

    // Paints
    private val landmarkPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val connectionPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        isAntiAlias = true
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    private val smallTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 30f
        isAntiAlias = true
    }

    private val tinyTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 24f
        isAntiAlias = true
    }

    private val backgroundPaint = Paint().apply {
        color = Color.argb(180, 0, 0, 0)  // Semi-transparent black
        style = Paint.Style.FILL
    }

    private val barBackgroundPaint = Paint().apply {
        color = Color.argb(100, 100, 100, 100)
        style = Paint.Style.FILL
    }

    /**
     * Update all data at once
     */
    fun updateData(
        result: GestureResult?,
        landmarks: FloatArray?,
        fps: Float,
        frameCount: Int,
        bufferSize: Int,
        handDetected: Boolean,
        imageWidth: Int,
        imageHeight: Int
    ) {
        this.result = result
        this.landmarks = landmarks
        this.fps = fps
        this.frameCount = frameCount
        this.bufferSize = bufferSize
        this.handDetected = handDetected
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Draw in order: back to front
        drawHandSkeleton(canvas)
        drawTopPanel(canvas)
        drawProbabilityPanel(canvas)
        drawBottomInstructions(canvas)
    }

    /**
     * Draw hand skeleton (21 landmarks + connections)
     */
    private fun drawHandSkeleton(canvas: Canvas) {
        val lm = landmarks ?: return
        if (lm.size != 63) return

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()

        // MediaPipe landmarks are already normalized (0.0 to 1.0)
        // Just scale directly to view dimensions
        val points = mutableListOf<Pair<Float, Float>>()
        for (i in 0 until 21) {
            // Landmarks: [x, y, z] where x, y are in [0, 1] range
            val x = lm[i * 3] * viewWidth
            val y = lm[i * 3 + 1] * viewHeight
            points.add(Pair(x, y))
        }

        // Draw connections first (underneath)
        for ((start, end) in handConnections) {
            val (x1, y1) = points[start]
            val (x2, y2) = points[end]
            canvas.drawLine(x1, y1, x2, y2, connectionPaint)
        }

        // Draw landmarks on top
        for ((x, y) in points) {
            canvas.drawCircle(x, y, 10f, landmarkPaint)
        }
    }

    /**
     * Draw top panel with status, buffer, gesture, confidence
     */
    private fun drawTopPanel(canvas: Canvas) {
        val panelHeight = 250f

        // Semi-transparent background
        canvas.drawRect(20f, 20f, width - 20f, panelHeight, backgroundPaint)

        var y = 60f

        // Hand detection status
        if (handDetected) {
            textPaint.color = Color.GREEN
            canvas.drawText("✓ HAND DETECTED", 40f, y, textPaint)
        } else {
            textPaint.color = Color.RED
            canvas.drawText("✗ NO HAND", 40f, y, textPaint)
        }

        y += 50f

        // Buffer status
        val bufferText = "Buffer: $bufferSize/${Config.SEQUENCE_LENGTH}"
        if (bufferSize >= Config.SEQUENCE_LENGTH) {
            smallTextPaint.color = Color.GREEN
        } else {
            smallTextPaint.color = Color.YELLOW
        }
        canvas.drawText(bufferText, 40f, y, smallTextPaint)

        y += 50f

        // Gesture name
        val res = result
        if (res != null && res.meetsThreshold()) {
            val gestureName = res.getFormattedGesture().uppercase()
            textPaint.color = if (res.confidence > 0.8f) Color.GREEN
                             else Color.YELLOW
            canvas.drawText("GESTURE: $gestureName", 40f, y, textPaint)

            y += 50f

            // Confidence bar
            drawConfidenceBar(canvas, y, res.confidence)
        } else if (bufferSize < Config.SEQUENCE_LENGTH) {
            textPaint.color = Color.GRAY
            val progress = (bufferSize * 100) / Config.SEQUENCE_LENGTH
            canvas.drawText("Collecting frames... $progress%", 40f, y, textPaint)
        } else {
            textPaint.color = Color.GRAY
            canvas.drawText("Low confidence", 40f, y, textPaint)
        }

        // FPS (top right)
        textPaint.color = Color.WHITE
        val fpsText = "FPS: %.1f".format(fps)
        canvas.drawText(fpsText, width - 200f, 60f, textPaint)

        // Frame counter
        smallTextPaint.color = Color.WHITE
        canvas.drawText("Frame: $frameCount", width - 200f, 110f, smallTextPaint)
    }

    /**
     * Draw confidence bar
     */
    private fun drawConfidenceBar(canvas: Canvas, y: Float, confidence: Float) {
        val barX = 40f
        val barWidth = 500f
        val barHeight = 40f

        // Background
        canvas.drawRect(barX, y, barX + barWidth, y + barHeight, barBackgroundPaint)

        // Filled portion
        val fillWidth = barWidth * confidence
        val barColor = when {
            confidence > 0.8f -> Color.GREEN
            confidence > 0.6f -> Color.YELLOW
            else -> Color.rgb(255, 165, 0)  // Orange
        }

        val fillPaint = Paint().apply {
            color = barColor
            style = Paint.Style.FILL
        }
        canvas.drawRect(barX, y, barX + fillWidth, y + barHeight, fillPaint)

        // Border
        val borderPaint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeWidth = 2f
        }
        canvas.drawRect(barX, y, barX + barWidth, y + barHeight, borderPaint)

        // Percentage text
        smallTextPaint.color = Color.WHITE
        canvas.drawText("${(confidence * 100).toInt()}%", barX + barWidth + 20f, y + 30f, smallTextPaint)
    }

    /**
     * Draw probability bars for all 11 classes (right side)
     */
    private fun drawProbabilityPanel(canvas: Canvas) {
        val res = result ?: return
        val probs = res.allProbabilities

        val panelX = width - 350f
        val panelY = 300f
        val panelWidth = 330f
        val panelHeight = (Config.NUM_CLASSES * 45 + 70).toFloat()

        // Background
        canvas.drawRect(panelX, panelY, panelX + panelWidth, panelY + panelHeight, backgroundPaint)

        // Title
        textPaint.color = Color.WHITE
        canvas.drawText("Probabilities:", panelX + 20f, panelY + 40f, textPaint)

        // Draw bars
        var barY = panelY + 70f
        for (i in 0 until Config.NUM_CLASSES) {
            val label = Config.IDX_TO_LABEL[i] ?: "unknown"
            val prob = probs[i]

            // Label
            val labelShort = label.replace('_', ' ').take(12)
            tinyTextPaint.color = Color.WHITE
            canvas.drawText(labelShort, panelX + 20f, barY + 20f, tinyTextPaint)

            // Bar
            val barStartX = panelX + 150f
            val barMaxWidth = 120f
            val barActualWidth = barMaxWidth * prob
            val barPaint = Paint().apply {
                color = if (prob > 0.5f) Color.GREEN else Color.GRAY
                style = Paint.Style.FILL
            }
            canvas.drawRect(barStartX, barY, barStartX + barActualWidth, barY + 25f, barPaint)

            // Percentage
            tinyTextPaint.color = Color.WHITE
            canvas.drawText("${(prob * 100).toInt()}%", barStartX + barMaxWidth + 10f, barY + 20f, tinyTextPaint)

            barY += 45f
        }
    }

    /**
     * Draw bottom instructions
     */
    private fun drawBottomInstructions(canvas: Canvas) {
        val instructions = "Double tap to switch camera  •  Touch controls coming soon"
        tinyTextPaint.color = Color.WHITE
        canvas.drawText(instructions, 40f, height - 40f, tinyTextPaint)
    }
}