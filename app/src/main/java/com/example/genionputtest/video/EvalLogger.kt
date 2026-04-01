package com.example.genionputtest.video

import android.content.Context
import android.os.Environment
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class EvalFrameRecord(
    val frameIndex: Int,
    val timestampMs: Long,
    val tier: String,
    val latencyMs: Double,
    val diffScore: Float
)

class EvalLogger {
    private val records = mutableListOf<EvalFrameRecord>()
    private var frameIndex = 0

    val isEmpty: Boolean get() = records.isEmpty()
    val frameCount: Int get() = records.size

    fun record(result: AdaptiveResult) {
        records.add(
            EvalFrameRecord(
                frameIndex = frameIndex++,
                timestampMs = System.currentTimeMillis(),
                tier = result.tier.name,
                latencyMs = result.inferenceMs,
                diffScore = result.diffScore
            )
        )
    }

    fun clear() {
        records.clear()
        frameIndex = 0
    }

    /** CSV 문자열 반환 */
    fun toCsv(): String = buildString {
        appendLine("frame,timestamp_ms,tier,latency_ms,diff_score")
        for (r in records) {
            appendLine("${r.frameIndex},${r.timestampMs},${r.tier},${"%.1f".format(r.latencyMs)},${"%.4f".format(r.diffScore)}")
        }
    }

    /**
     * CSV 파일을 앱 외부 저장소에 저장.
     * 경로: /sdcard/Android/data/<package>/files/Documents/eval_<label>_YYYYMMDD_HHmmss.csv
     * @return 저장된 파일, 실패 시 null
     */
    fun saveCsv(context: Context, label: String): File? {
        return try {
            val dir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: return null
            if (!dir.exists()) dir.mkdirs()
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val file = File(dir, "eval_${label}_$timestamp.csv")
            file.writeText(toCsv())
            file
        } catch (e: Exception) {
            null
        }
    }

    /** 요약 통계 문자열 (status log용) */
    fun summary(): String {
        if (records.isEmpty()) return "No records."
        val total = records.size
        val byTier = records.groupBy { it.tier }
        val t0 = byTier["ZERO"]?.size ?: 0
        val t1 = byTier["ONE"]?.size ?: 0
        val t2 = byTier["TWO"]?.size ?: 0
        val avgLatency = records.map { it.latencyMs }.average()
        return "Frames: $total | T0 ${pct(t0, total)}% T1 ${pct(t1, total)}% T2 ${pct(t2, total)}% | Avg: ${"%.0f".format(avgLatency)}ms"
    }

    private fun pct(n: Int, total: Int) = if (total == 0) 0 else (n * 100 / total)
}
