package com.example.digitrecognizer

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.view.animation.AnimationUtils
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.digitrecognizer.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var classifier: DigitClassifier? = null

    private val mainHandler = Handler(Looper.getMainLooper())
    private val isInferring = AtomicBoolean(false)

    // 持续推理的 Runnable，每 INFERENCE_INTERVAL_MS 执行一次
    private val inferenceRunnable = object : Runnable {
        override fun run() {
            performInferenceOnPreview()
            mainHandler.postDelayed(this, INFERENCE_INTERVAL_MS)
        }
    }

    // 当前预览帧（由 ImageAnalysis 持续更新）
    @Volatile private var latestBitmap: Bitmap? = null

    // 是否处于"拍照确认"冻结状态
    private var isFrozen = false

    // ──────────────────────────────────────────────
    // 权限请求回调
    // ──────────────────────────────────────────────
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startCamera()
        } else {
            showPermissionDenied()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        // 异步加载模型，避免阻塞主线程
        loadModelAsync()

        binding.btnCapture.setOnClickListener { onCaptureClick() }
        binding.btnGrantPermission.setOnClickListener {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun loadModelAsync() {
        binding.layoutLoading.visibility = View.VISIBLE
        cameraExecutor.execute {
            try {
                classifier = DigitClassifier(this)
                mainHandler.post {
                    binding.layoutLoading.visibility = View.GONE
                    checkCameraPermission()
                }
            } catch (e: Exception) {
                Log.e(TAG, "模型加载失败", e)
                mainHandler.post {
                    binding.layoutLoading.visibility = View.GONE
                    Toast.makeText(this, "模型加载失败: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED -> startCamera()
            shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) ->
                showPermissionDenied()
            else -> requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    // ──────────────────────────────────────────────
    // CameraX 启动
    // ──────────────────────────────────────────────
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            // ImageAnalysis：获取连续帧用于推理
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                updateLatestBitmap(imageProxy)
                imageProxy.close()
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "摄像头绑定失败", e)
            }

            // 开始定时推理
            mainHandler.post(inferenceRunnable)

            // 启动扫描线动画
            val scanAnim = AnimationUtils.loadAnimation(this, R.anim.scan_line)
            binding.scanLine.startAnimation(scanAnim)

        }, ContextCompat.getMainExecutor(this))
    }

    /** 从 ImageProxy 中提取 Bitmap 并缓存，正确处理行步长 */
    private fun updateLatestBitmap(imageProxy: ImageProxy) {
        try {
            val plane = imageProxy.planes[0]
            val buffer = plane.buffer
            val rowStride = plane.rowStride
            val pixelStride = plane.pixelStride
            val width = imageProxy.width
            val height = imageProxy.height

            val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val availableBytes = buffer.remaining()
            val bytes = ByteArray(availableBytes)
            buffer.get(bytes)

            val pixels = IntArray(width * height)
            for (row in 0 until height) {
                for (col in 0 until width) {
                    val idx = row * rowStride + col * pixelStride
                    if (idx + 3 >= availableBytes) break
                    val r = bytes[idx].toInt() and 0xFF
                    val g = bytes[idx + 1].toInt() and 0xFF
                    val b = bytes[idx + 2].toInt() and 0xFF
                    pixels[row * width + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            bmp.setPixels(pixels, 0, width, 0, 0, width, height)

            val rotation = imageProxy.imageInfo.rotationDegrees
            val rotated = if (rotation != 0) rotateBitmap(bmp, rotation) else bmp
            if (rotated !== bmp) bmp.recycle()

            val old = latestBitmap
            latestBitmap = rotated
            old?.recycle()
        } catch (e: Exception) {
            Log.e(TAG, "Bitmap转换失败", e)
        }
    }

    private fun rotateBitmap(src: Bitmap, degrees: Int): Bitmap {
        val matrix = android.graphics.Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
    }

    // ──────────────────────────────────────────────
    // 推理逻辑
    // ──────────────────────────────────────────────
    private fun performInferenceOnPreview() {
        if (isFrozen || isInferring.get()) return
        val bitmap = latestBitmap ?: run {
            Log.d(TAG, "latestBitmap is null, skip")
            return
        }
        val cls = classifier ?: run {
            Log.e(TAG, "classifier is null!")
            mainHandler.post { binding.tvConfidence.text = "模型未加载" }
            return
        }

        // 复制一份 bitmap，避免推理期间被主线程 recycle
        val bitmapCopy = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)

        isInferring.set(true)
        cameraExecutor.execute {
            try {
                val viewfinderRect = getViewfinderRectInBitmap(bitmapCopy)
                Log.d(TAG, "viewfinderRect=$viewfinderRect, bitmap=${bitmapCopy.width}x${bitmapCopy.height}")
                val processed = ImagePreprocessor.process(bitmapCopy, viewfinderRect)
                bitmapCopy.recycle()
                val result = cls.classify(processed)
                processed.recycle()
                Log.d(TAG, "推理结果: digit=${result?.digit}, conf=${result?.confidence}")

                mainHandler.post {
                    if (!isFrozen) updateResultUI(result)
                    isInferring.set(false)
                }
            } catch (e: Exception) {
                Log.e(TAG, "推理异常: ${e.message}", e)
                bitmapCopy.recycle()
                mainHandler.post {
                    isInferring.set(false)
                }
            }
        }
    }

    /**
     * 取 Bitmap 中心正方形区域（占短边的 70%），与屏幕取景框对应。
     */
    private fun getViewfinderRectInBitmap(bitmap: Bitmap): Rect {
        val w = bitmap.width
        val h = bitmap.height
        val side = (minOf(w, h) * 0.55f).toInt()
        val left = (w - side) / 2
        val top  = (h - side) / 2
        return Rect(left, top, left + side, top + side)
    }

    private var lastDisplayedDigit: Int = -1

    private fun updateResultUI(result: DigitClassifier.Result?) {
        if (result == null) {
            binding.tvDigit.text = "?"
            binding.tvConfidence.text = "识别失败"
            return
        }
        // 数字切换时才触发入场动画
        if (result.digit != lastDisplayedDigit) {
            val anim = AnimationUtils.loadAnimation(this, R.anim.digit_enter)
            binding.tvDigit.startAnimation(anim)
            lastDisplayedDigit = result.digit
        }
        binding.tvDigit.text = result.digit.toString()
        binding.tvConfidence.text = getString(R.string.confidence_format, result.confidence)
        binding.confidenceBarView.setScores(result.allScores, result.digit)
    }

    // ──────────────────────────────────────────────
    // 拍照确认按钮
    // ──────────────────────────────────────────────
    private fun onCaptureClick() {
        if (isFrozen) {
            // 解除冻结，恢复实时识别
            isFrozen = false
            binding.frozenFrame.visibility = View.GONE
            binding.frozenFrame.setImageBitmap(null)
            binding.btnCapture.text = getString(R.string.btn_capture)
            binding.tvHint.text = getString(R.string.hint_text)
            // 恢复扫描线动画
            val scanAnim = AnimationUtils.loadAnimation(this, R.anim.scan_line)
            binding.scanLine.startAnimation(scanAnim)
            binding.scanLine.visibility = View.VISIBLE
        } else {
            val bitmap = latestBitmap ?: return
            val cls = classifier ?: return

            // 停止扫描线
            binding.scanLine.clearAnimation()
            binding.scanLine.visibility = View.INVISIBLE

            // 显示冻结帧覆盖在预览上，视觉上"暂停"摄像头
            val frozenBmp = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
            binding.frozenFrame.setImageBitmap(frozenBmp)
            binding.frozenFrame.visibility = View.VISIBLE

            isFrozen = true
            binding.btnCapture.text = getString(R.string.btn_retry)
            binding.tvHint.text = "已锁定 — 点击「重新识别」继续"

            val inferBmp = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
            cameraExecutor.execute {
                try {
                    val rect = getViewfinderRectInBitmap(inferBmp)
                    val processed = ImagePreprocessor.process(inferBmp, rect)
                    inferBmp.recycle()
                    val result = cls.classify(processed)
                    processed.recycle()
                    mainHandler.post { updateResultUI(result) }
                } catch (e: Exception) {
                    Log.e(TAG, "拍照推理异常", e)
                    if (!inferBmp.isRecycled) inferBmp.recycle()
                }
            }
        }
    }

    private fun showPermissionDenied() {
        binding.layoutPermissionDenied.visibility = View.VISIBLE
    }

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacks(inferenceRunnable)
        cameraExecutor.shutdown()
        classifier?.close()
        latestBitmap?.recycle()
    }

    companion object {
        private const val TAG = "DigitRecognizer"
        private const val INFERENCE_INTERVAL_MS = 300L
    }
}
