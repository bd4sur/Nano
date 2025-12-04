
// 全局状态：是否启用TTS
let ENABLE_TTS = false;
let TTS_LOADED = false;

let tts_text_queue = [];

const PIPER_MODEL_HF_BASE_PATH = `https://huggingface.co/bd4sur/piper-voices-fork/resolve/main/`;
const PIPER_VOICES_CONFIG = {
    "en_US-amy-medium": {
        "key": "en_US-amy-medium",
        "name": "amy",
        "language": {
            "code": "en_US",
            "family": "en",
            "region": "US",
            "name_native": "English",
            "name_english": "English",
            "country_english": "United States"
        },
        "quality": "medium",
        "num_speakers": 1,
        "speaker_id_map": {},
        "files": {
            "en/en_US/amy/medium/en_US-amy-medium.onnx": {
                "size_bytes": 63201294,
                "md5_digest": "778d28aeb95fcdf8a882344d9df142fc"
            },
            "en/en_US/amy/medium/en_US-amy-medium.onnx.json": {
                "size_bytes": 4882,
                "md5_digest": "7f37dadb26340c90ebc8088e0b252310"
            },
            "en/en_US/amy/medium/MODEL_CARD": {
                "size_bytes": 281,
                "md5_digest": "6fca05ee5bfe8b28211b88b86b47e822"
            }
        },
        "aliases": []
    },
    "en_US-hfc_female-medium": {
        "key": "en_US-hfc_female-medium",
        "name": "hfc_female",
        "language": {
            "code": "en_US",
            "family": "en",
            "region": "US",
            "name_native": "English",
            "name_english": "English",
            "country_english": "United States"
        },
        "quality": "medium",
        "num_speakers": 1,
        "speaker_id_map": {},
        "files": {
            "en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx": {
                "size_bytes": 63201294,
                "md5_digest": "7abec91f1d6e19e913fbc4a333f62787"
            },
            "en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json": {
                "size_bytes": 5033,
                "md5_digest": "c3d00f54dac3b4068f2576c15c5da3bc"
            },
            "en/en_US/hfc_female/medium/MODEL_CARD": {
                "size_bytes": 354,
                "md5_digest": "a4a7b5da65e03e6972e44e9555a59aef"
            }
        },
        "aliases": []
    },
    "zh_CN-huayan-medium": {
        "key": "zh_CN-huayan-medium",
        "name": "huayan",
        "language": {
            "code": "zh_CN",
            "family": "zh",
            "region": "CN",
            "name_native": "简体中文",
            "name_english": "Chinese",
            "country_english": "China"
        },
        "quality": "medium",
        "num_speakers": 1,
        "speaker_id_map": {},
        "files": {
            "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx": {
                "size_bytes": 63201294,
                "md5_digest": "40cdb7930ff91b81574d5f0489e076ea"
            },
            "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json": {
                "size_bytes": 4822,
                "md5_digest": "1fda3ec1d0d3a5a74064397ea8fe0af0"
            },
            "zh/zh_CN/huayan/medium/MODEL_CARD": {
                "size_bytes": 276,
                "md5_digest": "b23255ace0cda4c2e02134d8a70c2e03"
            }
        },
        "aliases": []
    }
};

const piper_worker_url = "./piper_worker.js";
const piperPhonemizeJsUrl = new URL("piper_phonemize.js", document.location).href;
const piperPhonemizeWasmUrl = new URL("piper_phonemize.wasm", document.location).href;
const piperPhonemizeDataUrl = new URL("piper_phonemize.data", document.location).href;

const piper_blob_cache = {};
let piper_worker;
let piper_tts_audio_node = new Audio();

function init_piper_tts(prompt_dom_id) {
    piper_worker?.terminate();

    const voiceFiles = Object.keys(PIPER_VOICES_CONFIG["zh_CN-huayan-medium"].files);
    const modelUrl = `${PIPER_MODEL_HF_BASE_PATH}${voiceFiles.find(path => path.endsWith(".onnx"))}`;
    const modelConfigUrl = `${PIPER_MODEL_HF_BASE_PATH}${voiceFiles.find(path => path.endsWith(".onnx.json"))}`;

    piper_worker = new Worker(piper_worker_url);

    piper_worker.postMessage({
        kind:"init", piper_blob_cache, piperPhonemizeJsUrl, piperPhonemizeWasmUrl, piperPhonemizeDataUrl, modelUrl, modelConfigUrl});

    piper_worker.addEventListener("message", event => {
        const data = event.data;
        switch(data.kind) {
            case "output": {
                append_chat_info("语音转换完成");
                piper_tts_audio_node.src = URL.createObjectURL(data.file);
                piper_tts_audio_node.play();

                if(tts_text_queue.length > 0) {
                    const speakerId = 0;
                    let text = tts_text_queue.shift();
                    console.log(`TTS: ${text}`);
                    piper_worker.postMessage({kind:"tts", input: text, speakerId});
                }

                break;
            }
            case "stderr": {
                console.error(data.message);
                break;
            }
            case "init_finished": {
                TTS_LOADED = true;
                ENABLE_TTS = true;
                $(`#${prompt_dom_id}`).html(`TTS模型读取完毕`);
                break;
            }
            case "fetch": {
                const id = `fetch-${data.id}`;
                if(data.is_done === true) {
                    piper_blob_cache[data.url] = data.blob;
                }
                else {
                    const progress = data.blob ? 1 : (data.total ? data.loaded / data.total : 0);
                    $(`#${prompt_dom_id}`).html(`正在读取TTS模型... ${Math.round(progress * 100)}%`);
                }
                break;
            }
        }
    });
}

function start_tts(text) {
    console.log(`请求TTS：${text}`);
    if(ENABLE_TTS === true) {
        const speakerId = 0;
        tts_text_queue.push(text);
        let t = tts_text_queue.shift();
        piper_worker.postMessage({kind:"tts", input: t, speakerId});
        append_chat_info("正在转语音...");
    }
}

$("#tts_init").on("click", () => {
    init_piper_tts("tts_init");
});

$("#tts_switch").click(function(event) {
    if(ENABLE_TTS === true) {
        ENABLE_TTS = false;
        $("#tts_switch").html(`TTS引擎<span style="color: #bb0000; font-weight: bold;">已关闭</span>，点击启用`);
    }
    else if(ENABLE_TTS === false) {
        if(TTS_LOADED === false) {
            init_piper_tts("tts_switch");
        }
        ENABLE_TTS = true;
        $("#tts_switch").html(`TTS引擎<span style="color: #00bb00; font-weight: bold;">已启用</span>，点击关闭`);
    }
});

