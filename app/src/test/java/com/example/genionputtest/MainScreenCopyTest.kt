package com.example.genionputtest

import org.junit.Assert.assertEquals
import org.junit.Test

class MainScreenCopyTest {

    @Test
    fun initialScreenSections_areGroupedByPurpose() {
        val copy = initialScreenCopy()

        assertEquals("MobileCLIP2 Encoder Demo", copy.title)
        assertEquals("Embedding result", copy.resultTitle)
        assertEquals("Benchmark", copy.benchmarkTitle)
        assertEquals("Status log", copy.statusTitle)
    }

    @Test
    fun initialStatus_isShortAndScannable() {
        val copy = initialScreenCopy()

        assertEquals("Model is loading.", copy.initialStatus)
    }
}
