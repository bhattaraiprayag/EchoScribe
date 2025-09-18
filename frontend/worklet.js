// frontend/worklet.js

/**
 * An AudioWorkletProcessor for downsampling and converting audio to 16-bit PCM.
 * It receives audio from the microphone, processes it, and sends the raw
 * PCM data back to the main thread.
 */
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.targetSampleRate = 16000;
        this.buffer = [];
        this.resampleRatio = 1;
        this.isRecording = false;

        this.port.onmessage = (event) => {
            if (event.data.command === 'start') {
                this.isRecording = true;
                this.resampleRatio = sampleRate / this.targetSampleRate;
            } else if (event.data.command === 'stop') {
                this.isRecording = false;
                this.buffer = []; // Clear buffer on stop
            }
        };
    }

    process(inputs, outputs, parameters) {
        if (!this.isRecording) {
            return true;
        }

        const input = inputs[0]; // Assuming mono input
        if (input.length === 0 || !input[0]) {
            return true;
        }

        const inputData = input[0];

        // Simple downsampling: pick every Nth sample
        for (let i = 0; i < inputData.length; i += this.resampleRatio) {
            this.buffer.push(inputData[Math.floor(i)]);
        }

        // When we have enough samples for a chunk, process and send
        const chunkSize = 512; // Send data in reasonable chunks
        while (this.buffer.length >= chunkSize) {
            const chunk = this.buffer.splice(0, chunkSize);
            const pcmData = this.convertTo16BitPCM(chunk);
            this.port.postMessage(pcmData.buffer, [pcmData.buffer]);
        }

        return true; // Keep processor alive
    }

    convertTo16BitPCM(input) {
        const dataLength = input.length;
        const data = new Int16Array(dataLength);
        for (let i = 0; i < dataLength; i++) {
            // Clamp and convert to 16-bit integer
            data[i] = Math.max(-1, Math.min(1, input[i])) * 32767;
        }
        return data;
    }
}

registerProcessor('audio-processor', AudioProcessor);