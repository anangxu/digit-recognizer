package com.example.digitrecognizer

import android.content.Context
import android.graphics.Bitmap
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * DigitClassifier 封装 PyTorch Mobile 模型推理。
 *
 * 使用方式:
 *   val classifier = DigitClassifier(context)
 *   val result = classifier.classify(bitmap28x28Gray)
 *   result?.let { Log.d("TAG", "预测: ${it.digit}, 置信度: ${it.confidence}") }
 */
class DigitClassifier(context: Context) {

    data class Result(
        val digit: Int,
        val confidence: Float,
        val allScores: FloatArray      // softmax 后的 10 个类别概率
    )

    private val module: Module

    init {
        module = LiteModuleLoader.load(assetFilePath(context, MODEL_FILE))
    }

    /**
     * 对 28×28 灰度 Bitmap 执行推理。
     * @param bitmap 必须是 28×28，ARGB_8888 格式
     * @return 识别结果，包含数字 0-9 和置信度；若异常返回 null
     */
    fun classify(bitmap: Bitmap): Result? {
        return try {
            val inputTensor = bitmapToTensor(bitmap)
            val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
            val scores = outputTensor.dataAsFloatArray

            val softmaxScores = softmax(scores)
            val maxIdx = softmaxScores.indices.maxByOrNull { softmaxScores[it] } ?: 0
            Result(
                digit = maxIdx,
                confidence = softmaxScores[maxIdx] * 100f,
                allScores = softmaxScores
            )
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    /**
     * 将 28×28 灰度 Bitmap 转换为 PyTorch Tensor。
     * 归一化参数与训练时一致: mean=0.1307, std=0.3081
     */
    private fun bitmapToTensor(bitmap: Bitmap): Tensor {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val floatArray = FloatArray(width * height)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            // 提取灰度值 (R 通道，因为图像已灰度化，RGB 三通道相同)
            val gray = (pixel shr 16 and 0xFF) / 255.0f
            // MNIST 归一化: (x - 0.1307) / 0.3081
            floatArray[i] = (gray - MEAN) / STD
        }

        return Tensor.fromBlob(floatArray, longArrayOf(1, 1, height.toLong(), width.toLong()))
    }

    private fun softmax(scores: FloatArray): FloatArray {
        val maxScore = scores.max()
        val exp = FloatArray(scores.size) { Math.exp((scores[it] - maxScore).toDouble()).toFloat() }
        val sum = exp.sum()
        return FloatArray(scores.size) { exp[it] / sum }
    }

    fun close() {
        module.destroy()
    }

    companion object {
        private const val MODEL_FILE = "digit_model.ptl"
        private const val MEAN = 0.1307f
        private const val STD = 0.3081f

        /**
         * 将 assets 中的模型文件复制到 app 私有目录，返回文件路径。
         * LiteModuleLoader 需要文件系统路径，不能直接读取 assets 流。
         */
        fun assetFilePath(context: Context, assetName: String): String {
            val file = File(context.filesDir, assetName)
            if (file.exists() && file.length() > 0) return file.absolutePath

            try {
                context.assets.open(assetName).use { inputStream ->
                    FileOutputStream(file).use { outputStream ->
                        val buffer = ByteArray(4 * 1024)
                        var read: Int
                        while (inputStream.read(buffer).also { read = it } != -1) {
                            outputStream.write(buffer, 0, read)
                        }
                        outputStream.flush()
                    }
                }
            } catch (e: IOException) {
                throw RuntimeException("无法复制模型文件 $assetName", e)
            }
            return file.absolutePath
        }
    }
}
