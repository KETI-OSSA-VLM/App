package com.example.genionputtest

import org.junit.Assert.assertEquals
import org.junit.Test

class MainScreenCopyTest {

    @Test
    fun initialScreenSections_areGroupedByPurpose() {
        val copy = initialScreenCopy()

        assertEquals("Edge Vision Model Demo", copy.title)
        assertEquals("Active model", copy.modelPickerLabel)
        assertEquals("Result", copy.resultTitle)
        assertEquals("Benchmark", copy.benchmarkTitle)
        assertEquals("Status log", copy.statusTitle)
    }

    @Test
    fun initialStatus_isShortAndScannable() {
        val copy = initialScreenCopy()

        assertEquals("Model is loading.", copy.initialStatus)
    }

    @Test
    fun previewImageSpec_usesBoundedBoxForDisplayOnly() {
        val spec = previewImageSpec()

        assertEquals(280, spec.maxWidthDp)
        assertEquals(320, spec.maxHeightDp)
        assertEquals(220, spec.minHeightDp)
        assertEquals("FIT_CENTER", spec.scaleTypeName)
    }
}
