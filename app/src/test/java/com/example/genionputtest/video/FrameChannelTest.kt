package com.example.genionputtest.video

import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FrameChannelTest {

    @Test
    fun conflatedChannel_keepsOnlyLatestValue() = runTest {
        val channel = Channel<Int>(Channel.CONFLATED)

        channel.send(1)
        channel.send(2)
        channel.send(3)

        // CONFLATED: 1, 2는 3으로 덮어씌워짐
        assertEquals(3, channel.tryReceive().getOrNull())
        assertNull(channel.tryReceive().getOrNull())
        channel.close()
    }

    @Test
    fun forLoop_terminatesWhenChannelIsClosed() = runTest {
        val channel = Channel<Int>(Channel.CONFLATED)
        val received = mutableListOf<Int>()

        val producerJob = launch {
            channel.send(1)
            channel.send(2)
            channel.close()
        }

        for (value in channel) {
            received.add(value)
        }

        producerJob.join()
        assertTrue("Expected at least one received value", received.isNotEmpty())
    }

    @Test
    fun onUndeliveredElement_calledForEachDroppedFrame() = runTest {
        var droppedCount = 0
        // onUndeliveredElement는 채널 내부에서 교체(드롭)될 때마다 동기 호출됨
        val channel = Channel<Int>(Channel.CONFLATED) { droppedCount++ }

        // send(1): 버퍼에 1 저장
        // send(2): 1이 드롭(droppedCount=1), 버퍼에 2 저장
        // send(3): 2가 드롭(droppedCount=2), 버퍼에 3 저장
        channel.send(1)
        channel.send(2)
        channel.send(3)

        // 소비 전: 1, 2가 드롭됨
        assertEquals(2, droppedCount)

        // Note: close() does not trigger onUndeliveredElement for the retained element (3).
        // cancel() would. This test verifies drop-on-overwrite behavior only.
        channel.close()
    }
}
