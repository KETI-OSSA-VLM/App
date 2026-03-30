package com.example.genionputtest.video

import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FrameChannelTest {

    @Test
    fun `conflated channel keeps only latest value`() = runTest {
        val channel = Channel<Int>(Channel.CONFLATED)

        channel.send(1)
        channel.send(2)
        channel.send(3)

        // CONFLATED: 1, 2는 3으로 덮어씌워짐
        assertEquals(3, channel.tryReceive().getOrNull())
        assertNull(channel.tryReceive().getOrNull())
    }

    @Test
    fun `for loop terminates when channel is closed`() = runTest {
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
        assert(received.isNotEmpty()) { "Expected at least one received value" }
    }

    @Test
    fun `onUndeliveredElement called for each dropped frame`() = runTest {
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

        channel.close()
    }
}
