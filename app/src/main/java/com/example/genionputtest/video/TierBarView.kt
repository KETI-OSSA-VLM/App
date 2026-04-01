package com.example.genionputtest.video

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.view.View

/**
 * Tier 분포를 3색 바로 그리는 커스텀 뷰.
 * requestLayout() 없이 invalidate()만 호출 — 메인 스레드 부담 최소화.
 */
class TierBarView(context: Context) : View(context) {

    private var ratioT0 = 0f
    private var ratioT1 = 0f
    private var ratioT2 = 0f

    private val paintT0 = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#4CAF50") }
    private val paintT1 = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#FF9800") }
    private val paintT2 = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#F44336") }
    private val paintBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#ECF2F8") }

    private val clipPath = Path()
    private val boundsRect = RectF()
    private val cornerR by lazy { 5f * resources.displayMetrics.density }

    fun update(t0: Int, t1: Int, t2: Int) {
        val total = (t0 + t1 + t2).coerceAtLeast(1).toFloat()
        ratioT0 = t0 / total
        ratioT1 = t1 / total
        ratioT2 = t2 / total
        invalidate()  // layout 재측정 없음 — draw만 다시
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        boundsRect.set(0f, 0f, w.toFloat(), h.toFloat())
        clipPath.reset()
        clipPath.addRoundRect(boundsRect, cornerR, cornerR, Path.Direction.CW)
    }

    override fun onDraw(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()

        canvas.save()
        canvas.clipPath(clipPath)

        // 배경
        canvas.drawRect(0f, 0f, w, h, paintBg)

        if (ratioT0 + ratioT1 + ratioT2 > 0f) {
            val x1 = w * ratioT0
            val x2 = x1 + w * ratioT1

            if (ratioT0 > 0f) canvas.drawRect(0f, 0f, x1, h, paintT0)
            if (ratioT1 > 0f) canvas.drawRect(x1, 0f, x2, h, paintT1)
            if (ratioT2 > 0f) canvas.drawRect(x2, 0f, w, h, paintT2)
        }

        canvas.restore()
    }
}
