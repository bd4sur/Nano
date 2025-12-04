////////////////////////////////////////////////////////////////////////
// PTT based on FunASR https://github.com/alibaba-damo-academy/FunASR
// FunASR (c) 2022-2023 by zhaoming,mali aihealthx.com
// Mio (c) 2024 BD4SUR
////////////////////////////////////////////////////////////////////////

function FunASR(wssip, onText) {

    function FunASR_WebSocket(wssip, onmessage, onclose) {

        let asr_socket;

        this.connect = function () {
            asr_socket = new WebSocket(wssip);
            asr_socket.onopen = (e) => {
                let request = {
                    "chunk_size": new Array(5, 10, 5),
                    "wav_name": "h5",
                    "is_speaking": true,
                    "chunk_interval": 10,
                    "itn": false,
                    "mode": "online", //getAsrMode(),
                    "hotwords": `{"hello world": 40 }` // 关键字 -> 权重
                };
                asr_socket.send(JSON.stringify(request));
                console.log("FunASR连接成功");
            }
            asr_socket.onclose = onclose;
            asr_socket.onmessage = onmessage;
            asr_socket.onerror = (e) => { console.log(`FunASR WS error: ${String(e)}`) };
            return 1;
        };

        this.disconnect = function() {
            if(asr_socket !== undefined) {
                asr_socket.close();
            }
        };

        this.send = function(dataframe) {
            if(asr_socket == undefined) return;
            if(asr_socket.readyState === 1) { // 0:CONNECTING, 1:OPEN, 2:CLOSING, 3:CLOSED
                asr_socket.send(dataframe);
            }
        };

    }

    // 连接; 定义socket连接类对象与语音对象
    let asr_websocket = new FunASR_WebSocket(wssip, onAsrMsg, stop);

    // 录音; 定义录音对象,wav格式
    let recorder = Recorder({
        type: "pcm",
        bitRate: 16,
        sampleRate: 16000,
        onProcess: onRecorderProcess
    });

    let sampleBuf = new Int16Array();

    let rec_text = ""; // for online rec asr result
    let offline_text = ""; // for offline rec asr result

    function reset() {
        sampleBuf = new Int16Array();
        rec_text = "";
        offline_text = "";
    }

    function handleWithTimestamp(tmptext, tmptime)
    {
        console.log("tmptext: " + tmptext);
        console.log("tmptime: " + tmptime);
        if(tmptime === null || tmptime === undefined || tmptext.length <= 0) {
            return tmptext;
        }
        tmptext = tmptext.replace(/。|？|，|、|\?|\.|\ /g, ","); // in case there are a lot of "。"
        let words = tmptext.split(","); // split to chinese sentence or english words
        let jsontime = JSON.parse(tmptime); //JSON.parse(tmptime.replace(/\]\]\[\[/g, "],[")); // in case there are a lot segments by VAD
        let char_index = 0; // index for timestamp
        let text_withtime = "";
        for(let i = 0; i < words.length; i++) {
            if(words[i] === undefined || words[i].length <= 0) {
                continue;
            }
            console.log("words===",words[i]);
            console.log("words: " + words[i] + ", time=" + jsontime[char_index][0] / 1000);
            // if English
            if (/^[a-zA-Z]+$/.test(words[i])) {
                text_withtime = text_withtime + jsontime[char_index][0] / 1000 + ":" + words[i] + "\n";
                char_index = char_index + 1; //for English, timestamp unit is about a word
            }
            else { // if Chinese
                text_withtime = text_withtime + jsontime[char_index][0] / 1000 + ":" + words[i] + "\n";
                char_index = char_index + words[i].length; //for Chinese, timestamp unit is about a char
            }
        }
        return text_withtime;
    }

    // 语音识别结果; 对msg数据解析,将识别结果附加到编辑框中
    function onAsrMsg(msg) {
        let asr_data = JSON.parse(msg.data);
        console.log(asr_data);
        let rectxt = "" + asr_data['text'];
        let asr_mode = asr_data['mode'];
        let timestamp = asr_data['timestamp'];
        if(asr_mode === "2pass-offline" || asr_mode === "offline") {
            offline_text = offline_text + handleWithTimestamp(rectxt, timestamp);
            rec_text = offline_text;
        }
        else {
            rec_text = rec_text + rectxt;
        }

        onText(rec_text);

        console.log( "offline_text: " + asr_mode+","+offline_text);
        console.log( "rec_text: " + rec_text);
    }

    function onRecorderProcess(buffer, powerLevel, bufferDuration, bufferSampleRate, newBufferIdx, asyncEnd) {
        let data_48k = buffer[buffer.length - 1];
        let array_48k = new Array(data_48k);
        let data_16k = Recorder.SampleData(array_48k, bufferSampleRate, 16000).data;
        sampleBuf = Int16Array.from([...sampleBuf, ...data_16k]);
        let chunk_size = 960; // for asr chunk_size [5, 10, 5]
        // info_div.innerHTML=""+bufferDuration/1000+"s";
        while(sampleBuf.length >= chunk_size) {
            let sendBuf = sampleBuf.slice(0, chunk_size);
            sampleBuf = sampleBuf.slice(chunk_size, sampleBuf.length);
            asr_websocket.send(sendBuf);
        }
    }

    this.get_recorder_permission = function() {
        recorder.open(null);
        recorder.stop(null, null);
    };

    // 开始按钮
    this.start_recording = function() {
        recorder.open(() => {
            reset();
            recorder.start();
            console.log("开始录音");
        });
    }

    // 连接
    this.connect = function() {
        reset();
        asr_websocket.connect();
    }

    // 断开连接
    this.disconnect = function() {
        reset();
        asr_websocket.disconnect();
    }

    this.stop_recording = function() {
/*
        // 将缓冲区中剩余的采样发送到ASR
        if(sampleBuf.length > 0) {
            asr_websocket.send(sampleBuf);
            sampleBuf = new Int16Array();
        }
        let stop_request = {
            "chunk_size": new Array(5, 10, 5),
            "wav_name": "h5",
            "is_speaking":  false,
            "chunk_interval": 10,
            "mode": "online", //getAsrMode(),
        };
        asr_websocket.send(JSON.stringify(stop_request));

        // 等待一段时间后断开连接
        setTimeout(() => {
            asr_websocket.disconnect();
        }, 500);
*/
        // 停止录音
        recorder.stop(
            (blob, duration) => { return; },
            (err) => { console.error(err); }
        );
    }
}
