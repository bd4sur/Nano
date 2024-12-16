import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.1';

env.allowRemoteModels = true;
env.allowLocalModels = false;
// NOTE 仅本地调试用
env.remoteHost = "https://192.168.10.52:8443/";
env.remotePathTemplate = "{model}";

addEventListener('message', async (event) => {

    const message = event.data;
    let task = message.task;

    // Do some work...
    // TODO use message data
    try {
        let transcript = await transcribe(
            message.task.recorded_audio,
            message.model,
            message.multilingual,
            message.quantized,
            message.subtask,
            message.language,
        );

        if (transcript === null) {
            console.error("WHISPER WEB WORKER: transcription was null");
        }
        if (typeof transcript === 'undefined') {
            console.error("WHISPER WEB WORKER: transcription was undefined??");
        }

        delete task.recorded_audio;
        task['transcript'] = transcript;

        self.postMessage({
            task: task,
            status: "complete",
            //task: "automatic-speech-recognition",
            transcript: transcript,
        });

    } catch (e) {
        console.error("ERROR: whisper worker: ", e);
    }

});

// Define model factories
// Ensures only one model is created of each type

class PipelineFactory {
    static task = null;
    static model = null;
    static quantized = null;
    static instance = null;

    constructor(tokenizer, model, quantized) {
        this.tokenizer = tokenizer;
        this.model = model;
        this.quantized = quantized;
    }

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            this.instance = pipeline(this.task, this.model, {
                quantized: this.quantized,
                progress_callback,
                dtype: "fp32",
                device: "webgpu",
                // For medium models, we need to load the `no_attentions` revision to avoid running out of memory
                revision: this.model.includes("/whisper-medium") ? "no_attentions" : "main"
            });
        }

        return this.instance;
    }
}


class AutomaticSpeechRecognitionPipelineFactory extends PipelineFactory {
    static task = "automatic-speech-recognition";
    static model = null;
    static quantized = null;
}





const transcribe = async (
    audio,
    model,
    multilingual,
    quantized,
    subtask,
    language,
) => {
    console.log("whisper web worker: in transcribe.  model,multilingual,quantized,subtask,language: ", model, multilingual, quantized, subtask, language);

    let output = null;

    try {
        const isDistilWhisper = model.startsWith("distil-whisper/");

        let modelName = model;

        const p = AutomaticSpeechRecognitionPipelineFactory;
        if (p.model !== modelName || p.quantized !== quantized) {
            // Invalidate model if different
            p.model = modelName;
            p.quantized = quantized;

            if (p.instance !== null) {
                (await p.getInstance()).dispose();
                p.instance = null;
            }
        }

        // Load transcriber model
        let transcriber = await p.getInstance((data) => {
            self.postMessage(data);
        });

        const time_precision =
            transcriber.processor.feature_extractor.config.chunk_length /
            transcriber.model.config.max_source_positions;

        // Storage for chunks to be processed. Initialise with an empty chunk.
        let chunks_to_process = [
            {
                tokens: [],
                finalised: false,
            },
        ];

        // TODO: Storage for fully-processed and merged chunks
        // let decoded_chunks = [];

        const initial_prompt = "以下是汉语普通话，对应的简体中文简体字文本是：";
        const prompt_ids = await transcriber.processor.tokenizer(initial_prompt);

        function chunk_callback(chunk) {
            console.log("in whisper chunk callback. chunk: ", chunk);
            let last = chunks_to_process[chunks_to_process.length - 1];

            // Overwrite last chunk with new info
            Object.assign(last, chunk);
            last.finalised = true;

            // Create an empty chunk after, if it not the last chunk
            if (!chunk.is_last) {
                chunks_to_process.push({
                    tokens: [],
                    finalised: false,
                });
            }
        }



        // Inject custom callback function to handle merging of chunks
        function callback_function(item) {
            //console.log("whisper_worker: COMPLETE?  item: ", item);
            let last = chunks_to_process[chunks_to_process.length - 1];

            // Update tokens of last chunk
            last.tokens = [...item[0].output_token_ids];

            // Merge text chunks
            // TODO optimise so we don't have to decode all chunks every time
            let data = transcriber.tokenizer._decode_asr(chunks_to_process, {
                time_precision: time_precision,
                return_timestamps: true,
                force_full_sequences: false,
            });

            self.postMessage({
                status: "update",
                task: "automatic-speech-recognition",
                data: data,
            });
        }

        // Actually run transcription
        output = await transcriber(audio, {

            // Greedy
            top_k: 0,
            do_sample: false,

            // Sliding window
            chunk_length_s: isDistilWhisper ? 20 : 30,
            stride_length_s: isDistilWhisper ? 3 : 5,

            // Language and task
            language: language,
            task: subtask,

            prompt_ids: prompt_ids.input_ids,

            // Return timestamps
            return_timestamps: true,
            force_full_sequences: false,

            // Callback functions
            callback_function: callback_function, // after each generation step
            chunk_callback: chunk_callback, // after each chunk is processed
        }).catch((error) => {
            console.error("ERROR, actually running whisper failed");
            return null;
        });

        console.log("beyond WHISPER transcribe. output: ", output);

    }
    catch (e) {
        console.error("Whisper worker: error in transcribe function: ", e);
    }


    return output;
};

