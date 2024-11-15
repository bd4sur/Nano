import { WASI } from "/wasi.js";

let wasi_context;
let memory;
let wasm;

let malloc;
let free;
let init_nano;
let encode_external;
let decode_external;
let generate_next_token_external;
let close_nano;
let HEAPU8;

console.log("worker");

function show(str) {
    self.postMessage({
        eventType: "SHOW",
        eventData: str
    });
}

function report_tps(tps_str) {
    self.postMessage({
        eventType: "TPS",
        eventData: tps_str
    });
}

self.onmessage = function(event) {
    console.log(`Event: ${event}`);
    if(event.data.eventType === "INPUT") {
        let prompt = event.data.eventData;
        console.log(`INPUT: ${prompt}`);
        generate(prompt);
    }
    else if(event.data.eventType === "MODEL_FILE") {
        show("开始初始化LLM");
        let model_file_buffer = new Uint8Array(event.data.eventData);
        init(model_file_buffer);
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


async function init(model_file_buffer) {

    console.log("开始初始化WASM");

    wasi_context = new WASI({
        // args: ['infer', 'model.bin', "-l", "lora.bin", '-i', '人类的本质是什么？'],
        args: ['infer.wasm'],
        stdout: function(out) {
            console.log(out);
        }
    });

    memory = new WebAssembly.Memory({ initial: 10, maximum: 65536 });

    wasm = await WebAssembly.instantiateStreaming(
        fetch('infer.wasm'),
        {
            ...wasi_context.getImportObject(),
            // ...wasi_context.getImports(),
            env: { memory: memory }
        }
    );

    malloc = wasm.instance.exports.malloc;
    free = wasm.instance.exports.free;
    init_nano = wasm.instance.exports.init_nano;
    encode_external = wasm.instance.exports.encode_external;
    decode_external = wasm.instance.exports.decode_external;
    generate_next_token_external = wasm.instance.exports.generate_next_token_external;
    close_nano = wasm.instance.exports.close_nano;

    HEAPU8 = new Uint8Array(memory.buffer);




    let buffer_ptr = malloc(model_file_buffer.length);
    HEAPU8 = new Uint8Array(memory.buffer);

    HEAPU8.set(model_file_buffer, buffer_ptr);

    let res = init_nano(buffer_ptr, Date.now() % 0xffffffff);
    HEAPU8 = new Uint8Array(memory.buffer);

    self.postMessage({
        eventType: "INIT_FINISHED",
        eventData: "LLM initialization finished."
    });
}

function generate(prompt) {

    let input_text_ptr = malloc((prompt.length+1) * 4);
    let n_tokens_ptr = malloc(4);
    let ids_ptr = malloc(512 * 4);
    HEAPU8 = new Uint8Array(memory.buffer);

    for(let i = 0; i < prompt.length; i++) {
        let ch = prompt[i].charCodeAt();
        set_uint32(input_text_ptr + i * 4, HEAPU8, ch);
    }
    set_uint32(input_text_ptr + prompt.length * 4, HEAPU8, 0);

    let prompt_ptr = encode_external(input_text_ptr, n_tokens_ptr);
    HEAPU8 = new Uint8Array(memory.buffer);
    let num_prompt_tokens = HEAPU8[n_tokens_ptr];

    show(`Prompt tokens = ${num_prompt_tokens}`);

    for(let i = 0; i < num_prompt_tokens; i++) {
        let ch = get_uint32(prompt_ptr + i * 4, HEAPU8);
        set_uint32(ids_ptr + i * 4, HEAPU8, ch);
    }

    let output_count = 0;
    let pos = 0;
    let next_token = get_uint32(ids_ptr, HEAPU8);

    let elpased = [];

    while(pos < 512) {

        const t_0 = performance.now();

        let is_prefilling = (pos < num_prompt_tokens - 1) ? 1 : 0;

        next_token = generate_next_token_external(ids_ptr, pos, is_prefilling);
        HEAPU8 = new Uint8Array(memory.buffer);

        if(is_prefilling === 0) {
            set_uint32(ids_ptr + num_prompt_tokens * 4 + output_count * 4, HEAPU8, next_token);
            output_count += 1;

            let output_text_ptr = decode_external(ids_ptr + num_prompt_tokens * 4, output_count);
            HEAPU8 = new Uint8Array(memory.buffer);

            let output_str = "";
            let index = 0;
            while(1) {
                let ch = get_uint32(output_text_ptr + index * 4, HEAPU8);
                if(ch == 0) break;
                output_str += String.fromCharCode(ch);
                index += 1;
            }
            show(output_str);

        }
        // else if(is_prefilling == 1) {
            
        // }

        pos += 1;

        const t_1 = performance.now();
        elpased.push(1 / (t_1 - t_0) * 1000);
        let tps_now = elpased.slice(-1)[0];
        report_tps(tps_now);

        if(next_token == 0 || next_token == 3) break;

    }
}



// let result = wasi_context.start(wasm, { memory: memory });
