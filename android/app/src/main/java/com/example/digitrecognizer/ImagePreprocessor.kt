package com.example.digitrecognizer

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.Rect

/**
 * 标准手写数字预处理 pipeline：
 * 摄像头图像 → 灰度化 → 自适应二值化 → 找最大轮廓 → 裁剪数字区域
 * → resize 28×28 → 反色（黑底白字与MNIST一致） → 送入模型
 */
object ImagePreprocessor {

    private const val TARGET_SIZE = 28

    fun process(fullBitmap: Bitmap, viewfinderRect: Rect): Bitmap {
        // 1. 裁剪取景框区域
        val cropped = cropSafely(fullBitmap, viewfinderRect)

        // 2. 灰度化
        val gray = toGrayscale(cropped)
        if (cropped !== fullBitmap) cropped.recycle()

        val w = gray.width
        val h = gray.height
        val pixels = IntArray(w * h)
        gray.getPixels(pixels, 0, w, 0, 0, w, h)
        gray.recycle()

        val grayVals = IntArray(pixels.size) { pixels[it] and 0xFF }

        // 3. 自适应局部阈值二值化（比全局 Otsu 更能应对不均匀光照）
        //    foreground[i] = true 表示该像素是暗像素（笔迹）
        val foreground = adaptiveThreshold(grayVals, w, h, blockSize = 15, C = 8)

        // 4. 找最大连通区域（轮廓）的 bounding box
        val bbox = findLargestComponentBBox(foreground, w, h)

        // 5. 若未找到有效轮廓，降级为整图缩放
        val digitBitmap: Bitmap
        if (bbox == null || bbox.width() < 5 || bbox.height() < 5) {
            digitBitmap = buildForegroundBitmap(foreground, w, h)
        } else {
            // 加 15% 边距，正方形裁剪
            val padded = squarePadded(bbox, w, h, ratio = 0.2f)
            val full = buildForegroundBitmap(foreground, w, h)
            digitBitmap = Bitmap.createBitmap(full, padded.left, padded.top, padded.width(), padded.height())
            full.recycle()
        }

        // 6. Resize → 28×28
        val scaled = Bitmap.createScaledBitmap(digitBitmap, TARGET_SIZE, TARGET_SIZE, true)
        digitBitmap.recycle()

        // 7. 反色：foreground 已是黑底（0）白字（255），直接返回
        return scaled
    }

    // ─── 自适应阈值 ────────────────────────────────────────────────────────────

    /**
     * 局部均值自适应阈值。
     * 对每个像素，若其灰度值 < 局部均值 - C，则判定为前景（笔迹）。
     * 返回 true = 前景（暗），false = 背景（亮）
     */
    private fun adaptiveThreshold(
        gray: IntArray, w: Int, h: Int,
        blockSize: Int = 15, C: Int = 8
    ): BooleanArray {
        val half = blockSize / 2
        // 用积分图加速局部均值计算
        val integral = LongArray((w + 1) * (h + 1))
        for (y in 0 until h) {
            for (x in 0 until w) {
                integral[(y + 1) * (w + 1) + (x + 1)] =
                    gray[y * w + x].toLong() +
                    integral[y * (w + 1) + (x + 1)] +
                    integral[(y + 1) * (w + 1) + x] -
                    integral[y * (w + 1) + x]
            }
        }

        val result = BooleanArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val x1 = maxOf(0, x - half);        val y1 = maxOf(0, y - half)
                val x2 = minOf(w - 1, x + half);    val y2 = minOf(h - 1, y + half)
                val area = ((x2 - x1 + 1) * (y2 - y1 + 1)).toLong()
                val sum = integral[(y2 + 1) * (w + 1) + (x2 + 1)] -
                          integral[y1 * (w + 1) + (x2 + 1)] -
                          integral[(y2 + 1) * (w + 1) + x1] +
                          integral[y1 * (w + 1) + x1]
                val localMean = (sum / area).toInt()
                result[y * w + x] = gray[y * w + x] < localMean - C
            }
        }
        return result
    }

    // ─── 最大连通区域 ──────────────────────────────────────────────────────────

    /**
     * 用 BFS 找所有连通区域，返回像素数最多的区域的 bounding box。
     * 排除面积过小的噪点和面积过大的背景噪声。
     */
    private fun findLargestComponentBBox(fg: BooleanArray, w: Int, h: Int): Rect? {
        val visited = BooleanArray(fg.size)
        var bestSize = 0
        var bestBbox: Rect? = null
        val minArea = (w * h * 0.005).toInt()   // 至少占总面积 0.5%
        val maxArea = (w * h * 0.9).toInt()      // 不超过总面积 90%（排除整片背景）
        val queue = ArrayDeque<Int>()

        for (start in fg.indices) {
            if (!fg[start] || visited[start]) continue
            // BFS
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

                for ((dx, dy) in listOf(0 to 1, 0 to -1, 1 to 0, -1 to 0)) {
                    val nx = x + dx; val ny = y + dy
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
                    val nIdx = ny * w + nx
                    if (!fg[nIdx] || visited[nIdx]) continue
                    visited[nIdx] = true
                    queue.add(nIdx)
                }
            }

            if (size in minArea..maxArea && size > bestSize) {
                bestSize = size
                bestBbox = Rect(minX, minY, maxX + 1, maxY + 1)
            }
        }
        return bestBbox
    }

    // ─── 工具函数 ──────────────────────────────────────────────────────────────

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

    /** 将前景 BooleanArray 转为 Bitmap：前景=白(255)，背景=黑(0)，符合MNIST */
    private fun buildForegroundBitmap(fg: BooleanArray, w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val px = IntArray(fg.size) {
            if (fg[it]) 0xFFFFFFFF.toInt() else 0xFF000000.toInt()
        }
        bmp.setPixels(px, 0, w, 0, 0, w, h)
        return bmp
    }

    /** 在 bbox 四周加边距并保持正方形 */
    private fun squarePadded(bbox: Rect, w: Int, h: Int, ratio: Float): Rect {
        val side = maxOf(bbox.width(), bbox.height())
        val pad  = (side * ratio).toInt()
        var size = side + pad * 2
        size = minOf(size, w, h).coerceAtLeast(1)

        val cx = bbox.centerX().coerceIn(size / 2, w - size / 2)
        val cy = bbox.centerY().coerceIn(size / 2, h - size / 2)
        val left = (cx - size / 2).coerceIn(0, w - size)
        val top  = (cy - size / 2).coerceIn(0, h - size)
        return Rect(left, top, left + size, top + size)
    }
}
