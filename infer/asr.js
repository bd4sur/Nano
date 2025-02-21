
const opencc = OpenCC.Converter({ from: 'tw', to: 'cn' });

const SAMPLE_RATE = 16000;
const FFT_LENGTH = 4096;

window.whisper_worker = null;
window.whisper_worker_busy = false;
let whisper_worker_error_count = 0;


function create_whisper_worker() {

    window.whisper_worker = null;
    window.whisper_worker = new Worker('./whisper_worker.js', { type: 'module' });

    window.whisper_worker.addEventListener('message', e => {

        if (typeof e.data.status == 'string') {
            if (e.data.status == 'progress') {
                // console.log("whisper worker sent download percentage: ", e.data.progress);
            }
            else if (e.data.status == 'ready') {
                console.log("whisper worker sent ready message");
                window.whisper_worker_busy = false;
            }
            else if (e.data.status == 'initiate') {
                console.log("whisper worker sent initiate message");
            }
            else if (e.data.status == 'download') {
                console.log("whisper worker sent download message");
            }
            else if (e.data.status == 'update') {
                if (typeof e.data.data == 'object' && e.data.data != null && e.data.data.length) {
                    $("#log").append(`<p>${e.data.data[0]}</p>`);
                }
            }
            else if (e.data.status == 'complete') {
                window.whisper_worker_busy = false;
                console.log('GOT WHISPER COMPLETE.  e.data.transcript: ', e.data.transcript);

                if (e.data.transcript == null) {
                    console.warn("whisper recognition failed. If this is the first run, that's normal.");
                    
                }
                else if (typeof e.data.transcript != 'undefined') {
                    console.log("whisper returned transcription text: ", e.data.transcript);

                    if (Array.isArray(e.data.transcript)) {
                        console.log("typeof transcription is array");
                    }
                    else if (typeof e.data.transcript == 'object') {
                        // for(chunk of e.data.transcript.chunks) {
                        //     $("#log").append(`<p>${chunk.timestamp[0]} - ${chunk.timestamp[1]} | ${chunk.text}</p>`);
                        // }

                        let text = opencc(e.data.transcript.text);
                        console.info(text);

                        $("#input").val(text);
                        $("#submit").click();
                        // if (typeof text == 'string') {
                        //     $("#log").append(`<p>${text}</p>`);
                        //     console.log("GOT TEXT: ", text);
                        // }
                    }
                }
                else {
                    console.log("transcript was not in whisper e.data");
                }
            }
            else {
                console.log("whisper worker sent a content message");
                window.whisper_worker_busy = false;
                if (e.data.data == null) {
                    console.warn("whisper recognition failed. If this is the first run, that's normal.");
                }
            }
        }
    });

    window.whisper_worker.addEventListener('error', (error) => {
        console.error("ERROR: whisper_worker sent error. terminating!. Error was: ", error, error.message);
        whisper_worker_error_count++;

        window.whisper_worker.terminate();
        window.whisper_worker_busy = false;
        if (typeof error != 'undefined' && whisper_worker_error_count < 10) {
            setTimeout(() => {
                console.log("attempting to restart whisper worker");
                create_whisper_worker();
            }, 1000);
        }
        else {
            console.error("whisper_worker errored out");
        }
    });
}



function whisper_asr(task, language) {
    if (window.whisper_worker_busy) {
        console.error("whisper_asr was called while whisper worker was busy. Aborting.");
        return
    }
    if (typeof task.recorded_audio == 'undefined') {
        console.error("whisper_asr: task did not contain recorded_audio. Aborting.");
        return
    }

    let multilingual = false;
    if (typeof language == 'string') {
        if (language != 'en') {
            multilingual = true;
        }
    }
    const quantized = false;
    // const model = "whisper-base"; // NOTE 仅本地调试用
    const model = "bd4sur/whisper-base-fork";
    // const model = "Xenova/whisper-base";
    const subtask = "transcribe";
    window.whisper_worker.postMessage({
        task: task,
        model,
        multilingual,
        quantized,
        subtask,
        language: multilingual && language !== "auto" ? language : null,
    });
}



let is_audio_initialized = false;
let audioContext;
let mediaStream;
let mediaStreamSource;
let scriptProcessor;
let lowPassFilter;
let audioData = []; // 保存音频数据的数组
let recording = false;

// 初始化音频上下文和相关节点
async function initAudio() {
    if(is_audio_initialized !== false) {
        return;
    }
    audioContext = new AudioContext({sampleRate: SAMPLE_RATE});
    mediaStream = await navigator.mediaDevices.getUserMedia({
        "audio": {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false
        }
    });
    mediaStreamSource = audioContext.createMediaStreamSource(mediaStream);

    // 创建脚本处理器
    scriptProcessor = audioContext.createScriptProcessor(FFT_LENGTH, 1, 1);

    // 连接节点
    mediaStreamSource.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);

    scriptProcessor.onaudioprocess = (event) => {
        if (!recording) return;

        // 获取输入缓冲区

        const inputBuffer = event.inputBuffer;

        let audio;
        if (inputBuffer.numberOfChannels === 2) {
            const SCALING_FACTOR = Math.sqrt(2);

            let left = inputBuffer.getChannelData(0);
            let right = inputBuffer.getChannelData(1);

            audio = new Float32Array(left.length);
            for (let i = 0; i < inputBuffer.length; ++i) {
                audio[i] = SCALING_FACTOR * (left[i] + right[i]) / 2;
            }
        }
        else {
            audio = inputBuffer.getChannelData(0);
        }

        // 复制缓冲区数据
        const outputData = new Float32Array(inputBuffer.length);
        outputData.set(audio);

        audioData.push(outputData); // 保存音频数据
    };

    is_audio_initialized = true;
    console.log("录音上下文初始化完毕。");
}

let ptt_is_pushing = false;

let ptt_timestamp = 0;

$("#ptt").on("mousedown touchstart", async function (event) {
    $("#ptt").addClass("ButtonPTT_Active");
    ptt_timestamp = new Date().getTime();

    if (!audioContext) await initAudio();
    audioData = [];
    recording = true;
    console.log("Recording started...");
    $("#input").val("请按住讲话...");
});

$("#ptt").on("mouseup touchend", async function (event) {
    $("#ptt").removeClass("ButtonPTT_Active");
    let current_timestamp = new Date().getTime();
    if (current_timestamp - ptt_timestamp < 1000) {
        console.log("PTT按住时间短于1秒");
    }

    recording = false;
    console.log("Recording stopped...");
    $("#input").val("正在识别...");

    // 合并音频数据
    const audioBuffer = mergeAudioData(audioData);

    whisper_asr({
        recorded_audio: audioBuffer,
    }, "zh");

    const audioBlob = await createAudioBlob(audioBuffer);
    const audioUrl = URL.createObjectURL(audioBlob);

    let whisper_playback_audio = new Audio();
    whisper_playback_audio.src = audioUrl;
    whisper_playback_audio.play();

});


// 合并音频数据
function mergeAudioData(audioData) {
    const length = audioData.reduce((acc, chunk) => acc + chunk.length, 0);
    const mergedData = new Float32Array(length);
    let offset = 0;

    audioData.forEach((chunk) => {
        mergedData.set(chunk, offset);
        console.log(chunk.length);
        offset += chunk.length;
    });

    return mergedData;
}

// 创建音频 Blob
function createAudioBlob(audioData) {
    audioContext = new AudioContext({sampleRate: SAMPLE_RATE});
    const audioBuffer = audioContext.createBuffer(1, audioData.length, audioContext.sampleRate);
    audioBuffer.copyToChannel(audioData, 0);

    const offlineContext = new OfflineAudioContext(1, audioBuffer.length, audioContext.sampleRate);
    const bufferSource = offlineContext.createBufferSource();
    bufferSource.buffer = audioBuffer;

    bufferSource.connect(offlineContext.destination);
    bufferSource.start();

    return new Promise((resolve) => {
        offlineContext.startRendering().then((renderedBuffer) => {
            const audioBlob = bufferToWav(renderedBuffer);
            resolve(audioBlob);
        });
    });
}

// 将 AudioBuffer 转换为 WAV 文件
function bufferToWav(buffer) {
    const channelData = buffer.getChannelData(0);
    const wavBuffer = new Uint8Array(44 + channelData.length * 2);
    const view = new DataView(wavBuffer.buffer);

    // 写入 WAV 文件头
    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + channelData.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, buffer.sampleRate, true);
    view.setUint32(28, buffer.sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, "data");
    view.setUint32(40, channelData.length * 2, true);

    // 写入音频数据
    let offset = 44;
    for (let i = 0; i < channelData.length; i++) {
        const sample = Math.max(-1, Math.min(1, channelData[i]));
        view.setInt16(offset, sample * 0x7fff, true);
        offset += 2;
    }

    return new Blob([view], { type: "audio/wav" });
}

// 写入字符串到 DataView
function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

$("body").on("click", async () => {
    await initAudio();
});

