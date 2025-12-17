// frontend/worklet.js

/**
 * An AudioWorkletProcessor for downsampling, converting audio to 16-bit PCM,
 * and calculating audio volume with waveform data for visualization.
 */
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.targetSampleRate = 16000;
        this.buffer = [];
        this.resampleRatio = 1;
        this.isRecording = false;

        // Low-pass filter state for anti-aliasing
        this.filterState = 0;
        this.filterCoeff = 0;

        // Waveform visualization data
        this.waveformBuffer = [];
        this.waveformSampleCount = 64; // Number of samples to send for waveform

        this.port.onmessage = (event) => {
            if (event.data.command === 'start') {
                this.isRecording = true;
                this.resampleRatio = sampleRate / this.targetSampleRate;
                // Simple one-pole low-pass filter coefficient
                // Cutoff at approximately half the target sample rate
                this.filterCoeff = 1.0 / this.resampleRatio;
                this.filterState = 0;
            } else if (event.data.command === 'stop') {
                this.isRecording = false;
                this.buffer = [];
                this.waveformBuffer = [];
                this.filterState = 0;
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

        // Collect samples for waveform visualization (from raw input)
        for (let i = 0; i < inputData.length; i += 4) {
            this.waveformBuffer.push(inputData[i]);
        }

        // Downsampling with simple one-pole low-pass filter for anti-aliasing
        const ratio = Math.round(this.resampleRatio);
        for (let i = 0; i < inputData.length; i++) {
            // Apply simple low-pass filter
            this.filterState += this.filterCoeff * (inputData[i] - this.filterState);

            // Decimate: only take every Nth sample
            if (i % ratio === 0) {
                this.buffer.push(this.filterState);
            }
        }

        // When we have enough samples for a chunk, process and send
        const chunkSize = 512; // Send data in reasonable chunks
        while (this.buffer.length >= chunkSize) {
            const chunk = this.buffer.splice(0, chunkSize);
            const pcmData = this.convertTo16BitPCM(chunk);
            const volume = this.calculateRMS(chunk);

            // Get waveform samples for visualization
            const waveformSamples = this.getWaveformSamples();

            this.port.postMessage({
                pcmData: pcmData.buffer,
                volume,
                waveform: waveformSamples
            }, [pcmData.buffer]);
        }

        return true; // Keep processor alive
    }

    getWaveformSamples() {
        // Return a fixed number of samples for consistent visualization
        if (this.waveformBuffer.length === 0) {
            return new Array(this.waveformSampleCount).fill(0);
        }

        const result = [];
        const step = Math.max(1, Math.floor(this.waveformBuffer.length / this.waveformSampleCount));

        for (let i = 0; i < this.waveformSampleCount; i++) {
            const index = Math.min(i * step, this.waveformBuffer.length - 1);
            result.push(this.waveformBuffer[index] || 0);
        }

        // Clear buffer after sampling
        this.waveformBuffer = [];
        return result;
    }

    calculateRMS(input) {
        let sumOfSquares = 0;
        for (let i = 0; i < input.length; i++) {
            sumOfSquares += input[i] * input[i];
        }
        const rms = Math.sqrt(sumOfSquares / input.length);
        return rms;
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