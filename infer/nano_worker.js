let memory;
let wasm;

let malloc;
let free;
let init_nano;
let set_sampler;
let encode_external;
let decode_external;
let generate_next_token_external;
let load_lora_external;
let unload_lora_external;
let close_nano;
let HEAPU8;

let IS_RUNNING = false;

console.log("Nano WASM worker start!");

function send_info(message) {
    self.postMessage({
        eventType: "INFO",
        eventData: {
            message: message
        }
    });
}

function on_running(text, num_prompt_tokens, num_generated_tokens, status, tps) {
    self.postMessage({
        eventType: "ON_RUNNING",
        eventData: {
            text: text,
            num_prompt_tokens: num_prompt_tokens,
            num_generated_tokens: num_generated_tokens,
            status: status,
            tps: tps
        }
    });
}

function on_finished(tps) {
    self.postMessage({
        eventType: "ON_FINISHED",
        eventData: {
            tps: tps
        }
    });
}

self.onmessage = function(event) {
    console.log(`Worker RX: ${event.data.eventType}`);

    if(event.data.eventType === "INFER") {
        let prompt = event.data.eventData.prompt;
        let args = event.data.eventData.args;
        generate(prompt, args);
    }

    if(event.data.eventType === "INTERRUPT") {
        IS_RUNNING = false;
    }

    if(event.data.eventType === "MODEL_FILE") {
        send_info("开始初始化LLM");
        let max_seq_len = event.data.eventData.max_seq_len;
        let model_file_buffer = new Uint8Array(event.data.eventData.file_buffer);
        init(model_file_buffer, max_seq_len);
    }

    if(event.data.eventType === "LOAD_LORA") {
        send_info("加载LoRA插件");
        let model_file_buffer = new Uint8Array(event.data.eventData.file_buffer);
        load_lora(model_file_buffer, event.data.eventData.file_name);
    }

    if(event.data.eventType === "UNLOAD_LORA") {
        send_info("卸载LoRA插件");
        unload_lora();
    }

};


function get_uint32(ptr, heap) {
    let v0 = heap[ptr];
    let v1 = heap[ptr+1];
    let v2 = heap[ptr+2];
    let v3 = heap[ptr+3];
    return v0 + (v1 << 8) + (v2 << 16) + (v3 << 24);
}

function set_uint32(ptr, heap, value) {
    let v0 = value & 0x000000ff;
    let v1 = (value & 0x0000ff00) >> 8;
    let v2 = (value & 0x00ff0000) >> 16;
    let v3 = (value & 0xff000000) >> 24;
    heap[ptr + 0] = v0;
    heap[ptr + 1] = v1;
    heap[ptr + 2] = v2;
    heap[ptr + 3] = v3;
}


async function init(model_file_buffer, max_seq_len) {

    memory = new WebAssembly.Memory({ initial: 10, maximum: 65536 });

    wasm = await WebAssembly.instantiateStreaming(
        fetch('nano_infer.wasm'),
        {
            wasi_snapshot_preview1: {
                args_get: ()=>{},
                args_sizes_get: ()=>{},
                environ_get: ()=>{},
                environ_sizes_get: ()=>{},
                clock_time_get: ()=>{},
                clock_res_get: ()=>{},
                proc_exit: ()=>{ throw "End."; },

                fd_advise: ()=>{},
                fd_allocate: ()=>{},
                fd_close: ()=>{},
                fd_datasync: ()=>{},
                fd_fdstat_get: ()=>{},
                fd_fdstat_set_flags: ()=>{},
                fd_fdstat_set_rights: ()=>{},
                fd_filestat_get: ()=>{},
                fd_filestat_set_size: ()=>{},
                fd_filestat_set_times: ()=>{},
                fd_pread: ()=>{},
                fd_prestat_dir_name: ()=>{},
                fd_prestat_get: ()=>{},
                fd_pwrite: ()=>{},
                fd_read: ()=>{},
                fd_readdir: ()=>{},
                fd_renumber: ()=>{},
                fd_seek: ()=>{},
                fd_sync: ()=>{},
                fd_tell: ()=>{},
                fd_write: ()=>{},

                path_filestat_get: ()=>{},
                path_filestat_set_times: ()=>{},
                path_open: ()=>{},
                path_rename: ()=>{},
                path_unlink_file: ()=>{},
                path_create_directory: ()=>{},
            },
            env: { memory: memory }
        }
    );

    malloc = wasm.instance.exports.malloc;
    free = wasm.instance.exports.free;
    init_nano = wasm.instance.exports.init_nano;
    set_sampler = wasm.instance.exports.set_sampler;
    encode_external = wasm.instance.exports.encode_external;
    decode_external = wasm.instance.exports.decode_external;
    generate_next_token_external = wasm.instance.exports.generate_next_token_external;
    load_lora_external = wasm.instance.exports.load_lora_external;
    unload_lora_external = wasm.instance.exports.unload_lora_external;
    close_nano = wasm.instance.exports.close_nano;

    HEAPU8 = new Uint8Array(memory.buffer);




    let buffer_ptr = malloc(model_file_buffer.length);
    HEAPU8 = new Uint8Array(memory.buffer);

    HEAPU8.set(model_file_buffer, buffer_ptr);

    let res = init_nano(buffer_ptr, max_seq_len, Date.now() % 0xffffffff);
    HEAPU8 = new Uint8Array(memory.buffer);

    self.postMessage({
        eventType: "INIT_FINISHED",
        eventData: "LLM initialization finished."
    });

    send_info("语言模型加载完毕，可以开始对话啦！\nCiallo～(∠·ω< )⌒★");
}

function js_string_to_wchar_array(js_str) {
    // 计算所需内存大小（每个字符4字节 + 终止符）
    const bufferSize = (js_str.length + 1) * 4;
    let wchar_array = [];
    let offset = 0;
    for (let i = 0; i < js_str.length; i++) {
        const code = js_str.charCodeAt(i);
        // 处理代理对（Surrogate pairs）
        if (code >= 0xD800 && code <= 0xDBFF) {
            const nextCode = js_str.charCodeAt(i + 1);
            if (nextCode >= 0xDC00 && nextCode <= 0xDFFF) {
                // 组合代理对为完整码点
                const fullCode = (code - 0xD800) * 0x400 + (nextCode - 0xDC00) + 0x10000;
                wchar_array[offset] = fullCode;
                offset++;
                i++; // 跳过已处理的低位代理
                continue;
            }
        }
        // 直接写入 BMP 字符
        wchar_array[offset] = code;
        offset++;
    }
    // 添加4字节终止符
    wchar_array[offset] = 0;
    return wchar_array;
}

async function generate(prompt, args) {

    if(IS_RUNNING === true) {
        return;
    }
    IS_RUNNING = true;

    set_sampler(args.repetition_penalty, args.temperature, args.top_p, args.top_k, Date.now() % 0xffffffff);
    HEAPU8 = new Uint8Array(memory.buffer);

    let input_text_ptr = malloc((prompt.length+1) * 4);
    let n_tokens_ptr = malloc(4);
    let ids_ptr = malloc(args.max_seq_len * 4);
    HEAPU8 = new Uint8Array(memory.buffer);

    let prompt_wchar = js_string_to_wchar_array(prompt);
    console.log(prompt_wchar);

    for(let i = 0; i < prompt_wchar.length; i++) {
        // let ch = prompt_wchar[i].charCodeAt();
        let ch = prompt_wchar[i];
        set_uint32(input_text_ptr + i * 4, HEAPU8, ch);
    }
    set_uint32(input_text_ptr + prompt_wchar.length * 4, HEAPU8, 0);

    let prompt_ptr = encode_external(input_text_ptr, n_tokens_ptr);
    HEAPU8 = new Uint8Array(memory.buffer);
    let num_prompt_tokens = get_uint32(n_tokens_ptr, HEAPU8);

    console.log(`Prompt tokens = ${num_prompt_tokens}`);

    for(let i = 0; i < num_prompt_tokens; i++) {
        let ch = get_uint32(prompt_ptr + i * 4, HEAPU8);
        set_uint32(ids_ptr + i * 4, HEAPU8, ch);
    }

    let output_count = 0;
    let pos = 0;
    let next_token = get_uint32(ids_ptr, HEAPU8);

    let elpased = [];

    while(pos < args.max_seq_len) {

        const t_0 = performance.now();

        let is_prefilling = (pos < num_prompt_tokens - 1) ? 1 : 0;
        let status = (is_prefilling === 1) ? "Pre-filling..." : "Decoding...";

        next_token = generate_next_token_external(ids_ptr, pos, is_prefilling);
        HEAPU8 = new Uint8Array(memory.buffer);

        let output_str = "";
        if(is_prefilling === 0) {
            set_uint32(ids_ptr + num_prompt_tokens * 4 + output_count * 4, HEAPU8, next_token);
            output_count += 1;

            let output_text_ptr = decode_external(ids_ptr + num_prompt_tokens * 4, output_count);
            HEAPU8 = new Uint8Array(memory.buffer);

            let index = 0;
            while(1) {
                let ch = get_uint32(output_text_ptr + index * 4, HEAPU8);
                if(ch == 0) break;
                output_str += String.fromCharCode(ch);
                index += 1;
            }
        }

        pos += 1;

        const t_1 = performance.now();
        elpased.push(1 / (t_1 - t_0) * 1000);
        let tps_now = elpased.slice(-1)[0];
        on_running(output_str, num_prompt_tokens, pos, status, tps_now);

        if(IS_RUNNING !== true) break;
        if (next_token === 0 || next_token === 3 ||
            (is_prefilling === 0 && (next_token === 151643 || next_token === 151645))) break; // 包括Qwen3的<|endoftext|>(151643)或<|im_end|>(151645)

        await new Promise(resolve => setTimeout(resolve, 0));
    }

    IS_RUNNING = false;

    const tps_avg = elpased.reduce((a, b) => a + b) / elpased.length;
    on_finished(tps_avg);
}

function load_lora(lora_file_buffer, lora_file_name) {
    let buffer_ptr = malloc(lora_file_buffer.length);
    HEAPU8 = new Uint8Array(memory.buffer);

    HEAPU8.set(lora_file_buffer, buffer_ptr);

    let res = load_lora_external(buffer_ptr);
    HEAPU8 = new Uint8Array(memory.buffer);

    self.postMessage({
        eventType: "LORA_LOADED",
        eventData: {
            lora_file_name: lora_file_name
        }
    });

    send_info("LoRA插件加载完毕！");
}

function unload_lora() {
    let res = unload_lora_external();
    HEAPU8 = new Uint8Array(memory.buffer);

    self.postMessage({
        eventType: "LORA_UNLOADED",
        eventData: "LoRA module unloaded."
    });
}
