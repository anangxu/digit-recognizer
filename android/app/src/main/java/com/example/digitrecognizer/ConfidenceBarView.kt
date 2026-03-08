package com.example.digitrecognizer

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * 显示 0-9 每个数字置信度的横向条形图。
 * 调用 setScores(floatArray) 更新，条形长度由 softmax 概率驱动。
 */
class ConfidenceBarView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var scores: FloatArray = FloatArray(10)
    private var predictedDigit: Int = -1

    // 动画相关：每帧插值当前显示值
    private var displayScores: FloatArray = FloatArray(10)
    private var lastDrawTime: Long = 0L
    private val ANIM_SPEED = 8f   // 越大越快（插值系数 per 16ms）

    private val barPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#1AFFFFFF")
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textAlign = Paint.Align.CENTER
    }
    private val valuePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#99FFFFFF")
        textAlign = Paint.Align.RIGHT
    }

    // 霓虹青渐变（活跃条）
    private val activeColorStart = Color.parseColor("#FF00E5FF")
    private val activeColorEnd   = Color.parseColor("#FF00B8D4")
    // 普通条颜色
    private val inactiveColor    = Color.parseColor("#44FFFFFF")

    fun setScores(newScores: FloatArray, predicted: Int) {
        scores = newScores.copyOf()
        predictedDigit = predicted
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()

        // 计算每行高度（10 行 + 间距）
        val rows = 10
        val rowH = h / rows
        val barMaxW = w - 52f    // 左侧留 label，右侧留数值
        val barLeft = 28f
        val barRight = w - 28f
        val cornerR = rowH * 0.22f

        // 动画插值
        val now = System.currentTimeMillis()
        val dt = if (lastDrawTime == 0L) 16L else (now - lastDrawTime).coerceIn(4, 64)
        lastDrawTime = now
        val alpha = (ANIM_SPEED * dt / 1000f).coerceAtMost(1f)
        for (i in 0 until 10) {
            displayScores[i] = displayScores[i] + (scores[i] - displayScores[i]) * alpha
        }

        labelPaint.textSize = rowH * 0.48f
        valuePaint.textSize = rowH * 0.38f

        for (i in 0 until 10) {
            val top  = i * rowH + rowH * 0.1f
            val bot  = (i + 1) * rowH - rowH * 0.1f
            val barH = bot - top
            val centerY = (top + bot) / 2f

            // 背景条
            bgPaint.color = Color.parseColor("#1AFFFFFF")
            canvas.drawRoundRect(barLeft, top, barRight, bot, cornerR, cornerR, bgPaint)

            // 数据条（带渐变）
            val barW = (displayScores[i] * (barRight - barLeft)).coerceAtLeast(cornerR * 2)
            val isActive = (i == predictedDigit)

            barPaint.shader = if (isActive) {
                LinearGradient(
                    barLeft, 0f, barLeft + barW, 0f,
                    activeColorStart, activeColorEnd,
                    Shader.TileMode.CLAMP
                )
            } else {
                null
            }
            barPaint.color = if (isActive) activeColorStart else inactiveColor

            canvas.drawRoundRect(barLeft, top, barLeft + barW, bot, cornerR, cornerR, barPaint)

            // 数字标签
            labelPaint.color = if (isActive) Color.parseColor("#FF00E5FF") else Color.WHITE
            labelPaint.isFakeBoldText = isActive
            canvas.drawText(i.toString(), barLeft - 10f, centerY + labelPaint.textSize * 0.35f, labelPaint)

            // 百分比数值
            val pct = (displayScores[i] * 100).toInt()
            if (pct > 1 || isActive) {
                valuePaint.color = if (isActive) Color.parseColor("#FF00E5FF") else Color.parseColor("#66FFFFFF")
                canvas.drawText("$pct%", barRight - 2f, centerY + valuePaint.textSize * 0.35f, valuePaint)
            }
        }

        // 继续请求绘制以驱动动画（直到收敛）
        val converged = (0 until 10).all { kotlin.math.abs(displayScores[it] - scores[it]) < 0.002f }
        if (!converged) postInvalidateOnAnimation()
    }
}
