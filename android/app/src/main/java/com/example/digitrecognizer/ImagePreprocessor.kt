package com.example.digitrecognizer

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.Rect

/**
 * 增强版手写数字预处理 pipeline：
 *
 * 摄像头帧
 *   → 裁剪取景框
 *   → 灰度化
 *   → 高斯模糊（去噪）
 *   → 双阈值策略（优先Otsu全局，失败降级自适应局部）
 *   → 形态学闭运算（连接断笔画）
 *   → 连通域分析（面积过滤噪点，合并多域边界框）
 *   → 正方形裁剪 + 20%padding
 *   → 先缩放到 20×20（保留笔画锐度）
 *   → 居中放置到 28×28 黑色画布（模拟MNIST留白分布）
 *   → 送入模型
 */
object ImagePreprocessor {

    private const val TARGET_SIZE = 28
    private const val DIGIT_SIZE  = 20   // 数字内容区域大小，居中放在28×28内

    fun process(fullBitmap: Bitmap, viewfinderRect: Rect): Bitmap {
        // 1. 裁剪取景框
        val cropped = cropSafely(fullBitmap, viewfinderRect)

        // 2. 灰度化
        val gray = toGrayscale(cropped)
        if (cropped !== fullBitmap) cropped.recycle()

        val w = gray.width
        val h = gray.height
        val pixels = IntArray(w * h)
        gray.getPixels(pixels, 0, w, 0, 0, w, h)
        gray.recycle()

        var grayVals = IntArray(pixels.size) { pixels[it] and 0xFF }

        // 3. 高斯模糊（3×3，sigma≈1.0）去除摄像头噪点
        grayVals = gaussianBlur3x3(grayVals, w, h)

        // 4. 双阈值策略：先尝试 Otsu，若前景比例异常则改用自适应阈值
        var foreground = otsuThreshold(grayVals, w, h)
        val fgRatio = foreground.count { it }.toFloat() / foreground.size
        if (fgRatio < 0.01f || fgRatio > 0.6f) {
            // Otsu 失败（全黑/全白或背景太乱），降级到自适应阈值
            foreground = adaptiveThreshold(grayVals, w, h, blockSize = 21, C = 10)
        }

        // 4b. 颜色极性自动纠正：
        // MNIST 是黑底白字。如果 Otsu 判断出来的"前景"（笔画）
        // 实际上是亮像素（白色背景上写黑字 → 前景应该是暗像素）
        // 需要反转，确保前景始终是笔画（暗像素 → 最终渲染为白色）
        // 判断方法：前景像素的平均灰度 vs 背景像素的平均灰度
        // 正常情况：前景（笔画）更暗，平均灰度 < 背景平均灰度 → 无需翻转
        // 异常情况：前景（被当成笔画的像素）比背景更亮 → 需要反转
        run {
            var fgSum = 0L; var bgSum = 0L; var fgCnt = 0; var bgCnt = 0
            for (i in foreground.indices) {
                if (foreground[i]) { fgSum += grayVals[i]; fgCnt++ }
                else { bgSum += grayVals[i]; bgCnt++ }
            }
            if (fgCnt > 0 && bgCnt > 0) {
                val fgMean = fgSum.toFloat() / fgCnt
                val bgMean = bgSum.toFloat() / bgCnt
                // 若前景比背景更亮（即把白色背景当成了前景），则反转
                if (fgMean > bgMean) {
                    for (i in foreground.indices) foreground[i] = !foreground[i]
                }
            }
        }

        // 5. 形态学闭运算（先膨胀再腐蚀，半径=1），连接断开的笔画
        foreground = morphClose(foreground, w, h, radius = 1)

        // 6. 连通域分析：收集所有有效连通域，合并为统一边界框
        val bbox = findMergedBBox(foreground, w, h)

        // 7. 构建二值化 Bitmap（黑底白字）
        val binaryBmp = buildForegroundBitmap(foreground, w, h)

        // 8. 裁剪数字区域（加 padding）
        val digitBitmap: Bitmap
        if (bbox == null || bbox.width() < 4 || bbox.height() < 4) {
            digitBitmap = binaryBmp
        } else {
            val padded = squarePadded(bbox, w, h, ratio = 0.25f)
            digitBitmap = Bitmap.createBitmap(
                binaryBmp, padded.left, padded.top, padded.width(), padded.height()
            )
            binaryBmp.recycle()
        }

        // 9. 缩放到 DIGIT_SIZE×DIGIT_SIZE（用 NEAREST 保留二值锐度）
        val scaled = Bitmap.createScaledBitmap(digitBitmap, DIGIT_SIZE, DIGIT_SIZE, false)
        digitBitmap.recycle()

        // 10. 居中放置到 TARGET_SIZE×TARGET_SIZE 黑色画布（模拟 MNIST 四周留白）
        val result = Bitmap.createBitmap(TARGET_SIZE, TARGET_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        canvas.drawColor(0xFF000000.toInt())
        val offset = (TARGET_SIZE - DIGIT_SIZE) / 2
        canvas.drawBitmap(scaled, offset.toFloat(), offset.toFloat(), null)
        scaled.recycle()

        return result
    }

    // ─── 高斯模糊（3×3 近似） ──────────────────────────────────────────────────

    private fun gaussianBlur3x3(gray: IntArray, w: Int, h: Int): IntArray {
        // 核：[1,2,1 / 2,4,2 / 1,2,1] / 16
        val out = IntArray(gray.size)
        for (y in 0 until h) {
            for (x in 0 until w) {
                var sum = 0; var weight = 0
                for (dy in -1..1) for (dx in -1..1) {
                    val nx = (x + dx).coerceIn(0, w - 1)
                    val ny = (y + dy).coerceIn(0, h - 1)
                    val k = (2 - kotlin.math.abs(dx)) * (2 - kotlin.math.abs(dy))
                    sum += gray[ny * w + nx] * k
                    weight += k
                }
                out[y * w + x] = sum / weight
            }
        }
        return out
    }

    // ─── Otsu 全局阈值 ────────────────────────────────────────────────────────

    private fun otsuThreshold(gray: IntArray, w: Int, h: Int): BooleanArray {
        val hist = IntArray(256)
        for (v in gray) hist[v]++
        val total = gray.size.toFloat()

        var sumAll = 0.0
        for (i in 0..255) sumAll += i * hist[i]

        var sumB = 0.0; var wB = 0; var maxVar = 0.0; var thresh = 128
        for (i in 0..255) {
            wB += hist[i]; if (wB == 0) continue
            val wF = total - wB; if (wF == 0f) break
            sumB += i * hist[i]
            val mB = sumB / wB
            val mF = (sumAll - sumB) / wF
            val between = wB * wF * (mB - mF) * (mB - mF)
            if (between > maxVar) { maxVar = between; thresh = i }
        }
        // 前景 = 暗像素（笔迹）< 阈值
        return BooleanArray(gray.size) { gray[it] < thresh }
    }

    // ─── 自适应局部阈值（降级方案） ───────────────────────────────────────────

    private fun adaptiveThreshold(
        gray: IntArray, w: Int, h: Int,
        blockSize: Int = 21, C: Int = 10
    ): BooleanArray {
        val half = blockSize / 2
        val integral = LongArray((w + 1) * (h + 1))
        for (y in 0 until h) for (x in 0 until w) {
            integral[(y + 1) * (w + 1) + (x + 1)] =
                gray[y * w + x].toLong() +
                integral[y * (w + 1) + (x + 1)] +
                integral[(y + 1) * (w + 1) + x] -
                integral[y * (w + 1) + x]
        }
        val result = BooleanArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val x1 = maxOf(0, x - half); val y1 = maxOf(0, y - half)
            val x2 = minOf(w - 1, x + half); val y2 = minOf(h - 1, y + half)
            val area = ((x2 - x1 + 1) * (y2 - y1 + 1)).toLong()
            val sum = integral[(y2 + 1) * (w + 1) + (x2 + 1)] -
                      integral[y1 * (w + 1) + (x2 + 1)] -
                      integral[(y2 + 1) * (w + 1) + x1] +
                      integral[y1 * (w + 1) + x1]
            result[y * w + x] = gray[y * w + x] < (sum / area).toInt() - C
        }
        return result
    }

    // ─── 形态学闭运算（膨胀 → 腐蚀） ─────────────────────────────────────────

    private fun morphClose(fg: BooleanArray, w: Int, h: Int, radius: Int): BooleanArray {
        return morphErode(morphDilate(fg, w, h, radius), w, h, radius)
    }

    private fun morphDilate(fg: BooleanArray, w: Int, h: Int, r: Int): BooleanArray {
        val out = BooleanArray(fg.size)
        for (y in 0 until h) for (x in 0 until w) {
            var hit = false
            outer@ for (dy in -r..r) for (dx in -r..r) {
                val nx = x + dx; val ny = y + dy
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
                if (fg[ny * w + nx]) { hit = true; break@outer }
            }
            out[y * w + x] = hit
        }
        return out
    }

    private fun morphErode(fg: BooleanArray, w: Int, h: Int, r: Int): BooleanArray {
        val out = BooleanArray(fg.size)
        for (y in 0 until h) for (x in 0 until w) {
            var allFg = true
            outer@ for (dy in -r..r) for (dx in -r..r) {
                val nx = x + dx; val ny = y + dy
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
                if (!fg[ny * w + nx]) { allFg = false; break@outer }
            }
            out[y * w + x] = allFg
        }
        return out
    }

    // ─── 连通域分析：合并所有有效域的边界框 ──────────────────────────────────

    /**
     * 找所有面积在合理范围内的连通域，把它们的 bounding box 合并成一个。
     * 这样即使数字笔画断开成多个连通域，也能正确包围整个数字。
     */
    private fun findMergedBBox(fg: BooleanArray, w: Int, h: Int): Rect? {
        val visited = BooleanArray(fg.size)
        val minArea = maxOf(4, (w * h * 0.002).toInt())   // 至少 0.2%
        val maxArea = (w * h * 0.85).toInt()               // 最多 85%
        val queue = ArrayDeque<Int>()

        var mergedMinX = w; var mergedMaxX = 0
        var mergedMinY = h; var mergedMaxY = 0
        var hasValid = false

        for (start in fg.indices) {
            if (!fg[start] || visited[start]) continue
            queue.clear()
            queue.add(start)
            visited[start] = true
            var size = 0
            var minX = w; var maxX = 0; var minY = h; var maxY = 0

            while (queue.isNotEmpty()) {
                val idx = queue.removeFirst()
                val x = idx % w; val y = idx / w
                size++
                if (x < minX) minX = x; if (x > maxX) maxX = x
                if (y < minY) minY = y; if (y > maxY) maxY = y

                for ((dx, dy) in NEIGHBORS) {
                    val nx = x + dx; val ny = y + dy
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
                    val nIdx = ny * w + nx
                    if (!fg[nIdx] || visited[nIdx]) continue
                    visited[nIdx] = true
                    queue.add(nIdx)
                }
            }

            if (size in minArea..maxArea) {
                if (minX < mergedMinX) mergedMinX = minX
                if (maxX > mergedMaxX) mergedMaxX = maxX
                if (minY < mergedMinY) mergedMinY = minY
                if (maxY > mergedMaxY) mergedMaxY = maxY
                hasValid = true
            }
        }

        return if (hasValid) Rect(mergedMinX, mergedMinY, mergedMaxX + 1, mergedMaxY + 1)
               else null
    }

    // ─── 工具函数 ──────────────────────────────────────────────────────────────

    private val NEIGHBORS = listOf(0 to 1, 0 to -1, 1 to 0, -1 to 0)

    private fun cropSafely(src: Bitmap, rect: Rect): Bitmap {
        val left   = rect.left.coerceIn(0, src.width - 1)
        val top    = rect.top.coerceIn(0, src.height - 1)
        val right  = rect.right.coerceIn(left + 1, src.width)
        val bottom = rect.bottom.coerceIn(top + 1, src.height)
        return Bitmap.createBitmap(src, left, top, right - left, bottom - top)
    }

    private fun toGrayscale(src: Bitmap): Bitmap {
        val result = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        val paint = Paint()
        val cm = ColorMatrix().apply { setSaturation(0f) }
        paint.colorFilter = ColorMatrixColorFilter(cm)
        canvas.drawBitmap(src, 0f, 0f, paint)
        return result
    }

    /** 前景=白(255)，背景=黑(0)，符合MNIST格式 */
    private fun buildForegroundBitmap(fg: BooleanArray, w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val px = IntArray(fg.size) {
            if (fg[it]) 0xFFFFFFFF.toInt() else 0xFF000000.toInt()
        }
        bmp.setPixels(px, 0, w, 0, 0, w, h)
        return bmp
    }

    /** bbox 四周加边距并保持正方形，不超出图像边界 */
    private fun squarePadded(bbox: Rect, w: Int, h: Int, ratio: Float): Rect {
        val side = maxOf(bbox.width(), bbox.height())
        val pad  = (side * ratio).toInt().coerceAtLeast(2)
        var size = (side + pad * 2).coerceAtMost(minOf(w, h)).coerceAtLeast(1)

        val cx = bbox.centerX().coerceIn(size / 2, w - size / 2)
        val cy = bbox.centerY().coerceIn(size / 2, h - size / 2)
        val left = (cx - size / 2).coerceIn(0, w - size)
        val top  = (cy - size / 2).coerceIn(0, h - size)
        return Rect(left, top, left + size, top + size)
    }
}
