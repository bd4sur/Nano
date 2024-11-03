// 
// Nano Language Model - Inference Engine on Web Browser
//
//   BD4SUR 2024-10
//
//   Forked from:
//     - https://github.com/karpathy/llama2.c
//     - https://github.com/epicure/llama2.js
// 

// ===============================================================================
// 全局状态和缓冲区
// ===============================================================================

let LLM = { config: {}, param: {} };
let TOKENIZER = { config: {}, trie: {} };
let LoRA = null;
let FWD_BUFFER;

let is_generating = false;


// ===============================================================================
// 读取并解析模型文件
// ===============================================================================

function load_model(file_buffer) {

    const SIZE_OF_DTYPE = 4;
    const header_length = 256;

    let offset = 0;

    let header = new Int32Array(file_buffer.slice(0, header_length));

    let magic_number_0 = header[0];
    let magic_number_1 = header[1];

    if(magic_number_0 !== 0x42443453 || magic_number_1 !== 0x55524c4d) {
        console.error("Error: Corrupted or wrong model file!");
        return false;
    }

    let major_version = header[2];
    let minor_version = header[3];

    console.info(`Model version: ${major_version}.${minor_version}`);

    let model_type = header[4];
    let config_length = header[5]; // 暂不使用

    console.info(`Model type: ${model_type}`);

    // 读取模型结构参数

    LLM.config = {
        block_size: 0,
        vocab_size: 0,
        n_layer: 0,
        n_embd: 0,
        n_head: 0,
        n_kv_head: 0,
        n_hidden: 0,
        is_shared_classifier: 0
    };

    let cfg_keys = Object.keys(LLM.config);
    header.slice(6, 6 + cfg_keys.length).forEach((v, i) => { LLM.config[cfg_keys[i]] = v; });

    offset += header_length;

    // 读取模型权重

    const cfg = LLM.config;
    const is_shared_weights = cfg.is_shared_classifier > 0 ? 1 : 0;
    const head_dim = ((cfg.n_embd / cfg.n_head)^0);

    offset += 8; // param_count字段占用8个字节，仅用于C实现的推理引擎，这里不读取，直接跳过

    LLM.param = {
        token_embedding: new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * cfg.vocab_size * cfg.n_embd)),
        rms_norm_attn:   new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * cfg.n_layer * cfg.n_embd)),
        wq:              new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * cfg.n_layer * cfg.n_embd * cfg.n_embd)),
        wk:              new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * cfg.n_layer * cfg.n_embd * cfg.n_kv_head * head_dim)),
        wv:              new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * cfg.n_layer * cfg.n_embd * cfg.n_kv_head * head_dim)),
        wo:              new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * cfg.n_layer * cfg.n_embd * cfg.n_embd)),
        rms_norm_ffn:    new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * cfg.n_layer * cfg.n_embd)),
        w1:              new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * cfg.n_layer * cfg.n_embd * cfg.n_hidden)),
        w2:              new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * cfg.n_layer * cfg.n_embd * cfg.n_hidden)),
        w3:              new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * cfg.n_layer * cfg.n_embd * cfg.n_hidden)),
        rms_norm_final:  new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * cfg.n_embd)),
        token_classifier: null,
        freq_cis_real:   new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * cfg.block_size * head_dim / 2)),
        freq_cis_imag:   new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * cfg.block_size * head_dim / 2)),
    };

    LLM.param.token_classifier = is_shared_weights ? LLM.param.token_embedding : offset;

    // 读取词表、构建词元编解码器

    let byte_count = 0;

    let stoi = {};
    let itos = [];
    let special_tokens = {};

    let tokenizer_field_bytes = new Uint32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE))[0];
    let vocab_size = new Uint32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE))[0];

    while(byte_count < tokenizer_field_bytes - 4) { // 不含vocab_size字段的4个字节
        let token_header = new Uint8Array(file_buffer.slice(offset, offset += 4));
        byte_count += 4;
        let token_id     = new Uint32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE))[0];
        byte_count += 4;

        let token_length = token_header[0];
        let is_special   = token_header[1] === 1; // 0-false 1-true
        let reserved_0   = token_header[2]; // 预留
        let reserved_1   = token_header[3]; // 预留

        let token = "";
        for(let i = 0; i < token_length; i++) {
            let unicode = new Uint32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE))[0];
            byte_count += 4;
            token += String.fromCodePoint(unicode);
        }

        stoi[token] = token_id;
        itos[token_id] = token;

        if(is_special) {
            special_tokens[token] = token_id;
        }
    }

    TOKENIZER.config = {
        vocab_size: vocab_size,
        stoi: stoi,
        itos: itos,
        special_tokens: special_tokens
    };

    console.log(TOKENIZER.config);

    TOKENIZER.trie = new TrieTree(TOKENIZER.config.itos);


    // tk_length = new Uint32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE))[0];
    // tokenizer_config_json_base64 = new Uint8Array(file_buffer.slice(offset, offset += tk_length));
    // const text_decoder = new TextDecoder("utf-8");
    // TOKENIZER.config = JSON.parse(window.atob(text_decoder.decode(tokenizer_config_json_base64)));
    // TOKENIZER.trie = new TrieTree(TOKENIZER.config.itos);

    let kv_dim = (cfg.n_embd * cfg.n_kv_head) / cfg.n_head;

    // 构建前向传播数值的缓冲区

    FWD_BUFFER = {
        x:       new Float32Array(cfg.n_embd),   // activation at current time stamp (dim,)
        xb:      new Float32Array(cfg.n_embd),   // same, but inside a residual branch (dim,)
        xb2:     new Float32Array(cfg.n_embd),   // an additional buffer just for convenience (dim,)
        hb:      new Float32Array(cfg.n_hidden), // buffer for hidden dimension in the ffn (hidden_dim,)
        hb2:     new Float32Array(cfg.n_hidden), // buffer for hidden dimension in the ffn (hidden_dim,)
        q:       new Float32Array(cfg.n_embd),   // query (dim,)
    //  k:       new Float32Array(kv_dim),       // key (kv_dim,)
    //  v:       new Float32Array(kv_dim),       // value (kv_dim,)
        k_cache: new Float32Array(cfg.n_layer * cfg.block_size * kv_dim),   // key cache (layer, block_size, kv_dim)
        v_cache: new Float32Array(cfg.n_layer * cfg.block_size * kv_dim),   // value cache (layer, block_size, kv_dim)
        att:     new Float32Array(cfg.n_head * cfg.block_size), // buffer for scores/attention values (n_heads, block_size)
        logits:  new Float32Array(cfg.vocab_size), // output logits
    };

    return true;
}


function load_lora(file_buffer) {

    const SIZE_OF_DTYPE = 4;
    const header_length = 256;

    let offset = 0;

    let header = new Int32Array(file_buffer.slice(0, header_length));

    let magic_number_0 = header[0];
    let magic_number_1 = header[1];

    if(magic_number_0 !== 0x42443453 || magic_number_1 !== 0x55524c4d) {
        console.error("Error: Corrupted or wrong model file!");
        return false;
    }

    let major_version = header[2];
    let minor_version = header[3];

    console.info(`Model version: ${major_version}.${minor_version}`);

    let model_type = header[4];
    let config_length = header[5]; // 暂不使用

    console.info(`Model type: ${model_type}`);
    if(model_type !== 10) {
        console.error(`Error: Not a LoRA module!`);
        return false;
    }

    // 读取LoRA参数

    LoRA = { config: {}, param: {} };

    LoRA.config = {
        lora_rank: 0,
        lora_alpha: 0,
        n_layer: 0,     // 用于校验
        n_embd: 0,     // 用于校验
        n_head: 0,      // 用于校验
        n_kv_head: 0,   // 用于校验
        n_hidden: 0,    // 用于校验
        lora_config: 0  // 预留：用于控制LoRA用到哪些层
    };

    let cfg_keys = Object.keys(LoRA.config);
    header.slice(6, 6 + cfg_keys.length).forEach((v, i) => { LoRA.config[cfg_keys[i]] = v; });

    offset += header_length;

    // 读取LoRA参数

    const llm_cfg  = LLM.config;
    const lora_cfg = LoRA.config;
    const head_dim = ((llm_cfg.n_embd / llm_cfg.n_head)^0);
    const kv_dim = head_dim * llm_cfg.n_kv_head;

    // 校验LoRA模块与基座模型是否匹配
    if (llm_cfg.n_layer !== lora_cfg.n_layer ||
        llm_cfg.n_embd !== lora_cfg.n_embd ||
        llm_cfg.n_head !== lora_cfg.n_head ||
        llm_cfg.n_kv_head !== lora_cfg.n_kv_head ||
        llm_cfg.n_hidden !== lora_cfg.n_hidden) {
        console.error(`Error: LoRA module does not fit the base model.`);
        return false;
    }

    offset += 8; // param_count字段占用8个字节，仅用于C实现的推理引擎，这里不读取，直接跳过

    LoRA.param = {
        wq_lora_a: new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * llm_cfg.n_layer * lora_cfg.lora_rank * llm_cfg.n_embd)),
        wq_lora_b: new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * llm_cfg.n_layer * llm_cfg.n_embd * lora_cfg.lora_rank)),
        wk_lora_a: new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * llm_cfg.n_layer * lora_cfg.lora_rank * llm_cfg.n_embd)),
        wk_lora_b: new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * llm_cfg.n_layer * kv_dim * lora_cfg.lora_rank)),
        wv_lora_a: new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * llm_cfg.n_layer * lora_cfg.lora_rank * llm_cfg.n_embd)),
        wv_lora_b: new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * llm_cfg.n_layer * kv_dim * lora_cfg.lora_rank)),
        wo_lora_a: new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * llm_cfg.n_layer * lora_cfg.lora_rank * llm_cfg.n_embd)),
        wo_lora_b: new Float32Array(file_buffer.slice(offset, offset += SIZE_OF_DTYPE * llm_cfg.n_layer * llm_cfg.n_embd * lora_cfg.lora_rank)),
    };

    // 初始化LoRA数值缓冲区

    FWD_BUFFER.q0 = new Float32Array(lora_cfg.lora_rank);   // query  LoRA branch (lora_cfg.lora_rank,)
    FWD_BUFFER.k0 = new Float32Array(lora_cfg.lora_rank);   // key    LoRA branch (lora_cfg.lora_rank,)
    FWD_BUFFER.v0 = new Float32Array(lora_cfg.lora_rank);   // value  LoRA branch (lora_cfg.lora_rank,)
    FWD_BUFFER.o0 = new Float32Array(lora_cfg.lora_rank);   // output LoRA branch (lora_cfg.lora_rank,)
    FWD_BUFFER.q1 = new Float32Array(llm_cfg.n_embd);           // query  LoRA branch (dim,)
    FWD_BUFFER.k1 = new Float32Array(kv_dim);               // key    LoRA branch (kv_dim,)
    FWD_BUFFER.v1 = new Float32Array(kv_dim);               // value  LoRA branch (kv_dim,)
    FWD_BUFFER.o1 = new Float32Array(llm_cfg.n_embd);           // output LoRA branch (kv_dim,)

    return true;
}


function unload_lora() {
    LoRA = null;
}


// ===============================================================================
// 基础算子
//   所有算子都是C风格的：函数本身不返回值，通过参数引用的buffer来传递计算结果。
// ===============================================================================

function accum(a, b, size) {
    for (let i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

function scale(a, k, size) {
    for (let i = 0; i < size; i++) {
        a[i] *= k;
    }
}

function rms_norm(o, x, weight, size) {
    // calculate sum of squares
    let ss = 0.0;
    for (let j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5;
    ss = 1.0 / Math.sqrt(ss);
    // normalize and scale
    for (let j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

function softmax(x, size) {
    // find max value (for numerical stability)
    let max_val = x[0];
    for (let i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    let sum = 0.0;
    for (let i = 0; i < size; i++) {
        x[i] = Math.exp(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (let i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// 矩阵乘：绝大多数的计算量都花费在这个算子上面
function matmul(xout, x, w, n, d) {
    // W (d,n) @ x (n,) -> xout (d,)
    for (let i = 0; i < d; i++) {
        let val = 0.0;
        for (let j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}


// ===============================================================================
// 核心函数：语言模型前向传播
//   Args:
//     token - I   词元编码（在token_embedding中的列号，或者说词表中的编号）。
//                 NOTE 为什么只输入1个词元？因为过往输入的词元已经被保存在KV-Cache中了。
//     pos   - I   当前词元的位置，从0开始。
//     llm   - I   语言模型对象，包括模型结构参数和权重等。
//     lora  - I   LoRA模块对象。如果为null，则不使用LoRA。
//     buf   - IO  数据缓冲区，通过此缓冲区，张量在各层之间传播。
//   Return:
//     最后一层输出的logits。
// ===============================================================================

function llm_forward(token, pos, llm, lora, buf) {

    let cfg = llm.config;
    let w = llm.param;
    let s = buf;

    // 使用LoRA？
    let use_lora = (lora !== null);
    let a = null;
    let lora_rank = null;
    let lora_alpha = null;
    if(use_lora) {
        a = lora.param;
        lora_rank = lora.config.lora_rank;
        lora_alpha = lora.config.lora_alpha;
    }

    let x = s.x;
    const dim = cfg.n_embd; // Q的维度（每个注意力头的维度*h）
    const kv_dim = dim * (cfg.n_kv_head / cfg.n_head); // KV的维度=每个注意力头的维度*m
    const kv_mul = cfg.n_head / cfg.n_kv_head;
    const hidden_dim = cfg.n_hidden;
    const head_dim = dim / cfg.n_head; // 每个注意力头的维度，对于QKV都是相同的

    // copy the token embedding into x
    x.set(w.token_embedding.subarray(token * dim, (token + 1) * dim));
    
    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    const freq_cis_real_row = w.freq_cis_real.subarray(pos * head_dim / 2, (pos + 1) * head_dim / 2);
    const freq_cis_imag_row = w.freq_cis_imag.subarray(pos * head_dim / 2, (pos + 1) * head_dim / 2);

    // forward all the layers
    for(let l = 0; l < cfg.n_layer; l++) {
        // attention rmsnorm
        rms_norm(s.xb, x, w.rms_norm_attn.subarray(l * dim, (l + 1) * dim), dim);

        // save key,value at this time step (pos) to our kv cache
        const loff = l * cfg.block_size * kv_dim; // kv cache layer offset for convenience
        s.k = s.k_cache.subarray(loff + pos * kv_dim, loff + (pos + 1) * kv_dim);
        s.v = s.v_cache.subarray(loff + pos * kv_dim, loff + (pos + 1) * kv_dim);

        // qkv matmuls for this position
        matmul(s.q, s.xb, w.wq.subarray(l * dim * dim,    (l + 1) * dim * dim),    dim, dim);
        matmul(s.k, s.xb, w.wk.subarray(l * dim * kv_dim, (l + 1) * dim * kv_dim), dim, kv_dim);
        matmul(s.v, s.xb, w.wv.subarray(l * dim * kv_dim, (l + 1) * dim * kv_dim), dim, kv_dim);

        // 计算QKV的低秩分解分支，并将其累加到原来的输出上
        if(use_lora) {
            matmul(s.q0, s.xb, a.wq_lora_a.subarray(l * lora_rank * dim, (l + 1) * lora_rank * dim), dim, lora_rank);
            matmul(s.k0, s.xb, a.wk_lora_a.subarray(l * lora_rank * dim, (l + 1) * lora_rank * dim), dim, lora_rank);
            matmul(s.v0, s.xb, a.wv_lora_a.subarray(l * lora_rank * dim, (l + 1) * lora_rank * dim), dim, lora_rank);

            matmul(s.q1, s.q0, a.wq_lora_b.subarray(l * dim    * lora_rank, (l + 1) * dim    * lora_rank), lora_rank, dim);
            matmul(s.k1, s.k0, a.wk_lora_b.subarray(l * kv_dim * lora_rank, (l + 1) * kv_dim * lora_rank), lora_rank, kv_dim);
            matmul(s.v1, s.v0, a.wv_lora_b.subarray(l * kv_dim * lora_rank, (l + 1) * kv_dim * lora_rank), lora_rank, kv_dim);

            scale(s.q1, (lora_alpha / lora_rank), dim);
            scale(s.k1, (lora_alpha / lora_rank), kv_dim);
            scale(s.v1, (lora_alpha / lora_rank), kv_dim);

            accum(s.q, s.q1, dim);
            accum(s.k, s.k1, kv_dim);
            accum(s.v, s.v1, kv_dim);
        }

        // RoPE旋转位置编码实现方式1：使用模型提供的旋转系数
        for (let h = 0; h < cfg.n_head; h++) {
            const q = s.q.subarray(h * head_dim, (h + 1) * head_dim);
            for (let i = 0; i < head_dim; i += 2) {
                const q0 = q[i];
                const q1 = q[i + 1];
                const fcr = freq_cis_real_row[i / 2];
                const fci = freq_cis_imag_row[i / 2];
                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
            }
        }
        for (let m = 0; m < cfg.n_kv_head; m++) {
            const k = s.k.subarray(m * head_dim, (m + 1) * head_dim);
            for (let i = 0; i < head_dim; i += 2) {
                const k0 = k[i];
                const k1 = k[i + 1];
                const fcr = freq_cis_real_row[i / 2];
                const fci = freq_cis_imag_row[i / 2];
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        /*
        // RoPE旋转位置编码实现方式2：直接计算旋转系数
        for (let i = 0; i < dim; i += 2) {
            let ih = i % head_dim;
            let freq = 1.0 / Math.pow(10000.0, ih / head_dim);
            let val = pos * freq;
            let fcr = Math.cos(val);
            let fci = Math.sin(val);

            if(i < kv_dim) {
                let kr = s.k[i];
                let ki = s.k[i+1];
                s.k[i]   = kr * fcr - ki * fci;
                s.k[i+1] = kr * fci + ki * fcr;
            }
            let qr = s.q[i];
            let qi = s.q[i+1];
            s.q[i]   = qr * fcr - qi * fci;
            s.q[i+1] = qr * fci + qi * fcr;
        }
        */

        // 分组查询多头注意力（GQA-MHA），遍历所有的Q注意力头
        for (let h = 0; h < cfg.n_head; h++) {
            // KV分组注意力头的序号
            let m = ((h / kv_mul)^0);
            // get the query vector for this head
            const qh = s.q.subarray(h * head_dim, (h + 1) * head_dim);
            // attention scores for this head
            const att = s.att.subarray(h * cfg.block_size, (h + 1) * cfg.block_size);
            // 计算因果自注意力，包括当前时间步 iterate over all timesteps, including the current one
            for (let t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                const kh = s.k_cache.subarray(loff + t * kv_dim + m * head_dim, loff + (t + 1) * kv_dim + m * head_dim);
                // calculate the attention score as the dot product of q and k
                let score = 0.0;
                for (let i = 0; i < head_dim; i++) {
                    score += qh[i] * kh[i];
                }
                score /= Math.sqrt(head_dim);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            for (let i = 0; i < head_dim; i++) {
                let val = 0.0;
                for (let t = 0; t <= pos; t++) {
                    const vh = s.v_cache.subarray(loff + t * kv_dim + m * head_dim, loff + (t + 1) * kv_dim + m * head_dim);
                    val += att[t] * vh[i]; // NOTE bad locality
                    // val += att[t] * s.v_cache[loff + t * kv_dim + m * head_dim + i]; // NOTE bad locality
                }
                s.xb[h * head_dim + i] = val;
            }
        }

        // final matmul to get the output of the attention
        matmul(s.xb2, s.xb, w.wo.subarray(l * dim * dim, (l + 1) * dim * dim), dim, dim);

        // 计算output的低秩分解分支，并将其累加到原来的输出上
        if(use_lora) {
            matmul(s.o0, s.xb, a.wo_lora_a.subarray(l * lora_rank * dim, (l + 1) * lora_rank * dim), dim, lora_rank);
            matmul(s.o1, s.o0, a.wo_lora_b.subarray(l * dim    * lora_rank, (l + 1) * dim    * lora_rank), lora_rank, dim);
            scale(s.o1, (lora_alpha / lora_rank), dim);
            accum(s.xb2, s.o1, dim);
        }

        // residual connection back into x
        accum(x, s.xb2, dim);

        // ffn rmsnorm
        rms_norm(s.xb, x, w.rms_norm_ffn.subarray(l * dim, (l + 1) * dim), dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        matmul(s.hb, s.xb, w.w1.subarray(l * dim * hidden_dim, (l + 1) * dim * hidden_dim), dim, hidden_dim);
        matmul(s.hb2, s.xb, w.w3.subarray(l * dim * hidden_dim, (l + 1) * dim * hidden_dim), dim, hidden_dim);

        // SwiGLU non-linearity
        for (let i = 0; i < hidden_dim; i++) {
            let val = s.hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0 / (1.0 + Math.exp(-val)));
            // elementwise multiply with w3(x)
            val *= s.hb2[i];
            s.hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s.xb, s.hb, w.w2.subarray(l * dim * hidden_dim, (l + 1) * dim * hidden_dim), hidden_dim, dim);

        // residual connection
        accum(x, s.xb, dim);
    }

    // final rmsnorm
    rms_norm(x, x, w.rms_norm_final, dim);

    // classifier into logits
    matmul(s.logits, x, w.token_classifier, cfg.n_embd, cfg.vocab_size);

    return s.logits;
}

// ===============================================================================
// 词元编解码、分词器（基于Trie树）
// ===============================================================================

function TrieTree(vocab) {
    this.root = {};
    this.max_token_length = 0;
    this.END_CHAR = "__end__";
    for(let i = 0; i < vocab.length; i++) {
        let word = vocab[i];
        if(word.length > this.max_token_length) {
            this.max_token_length = word.length;
        }
        let current_dict = this.root;
        for(let j = 0; j < word.length; j++) {
            c = word[j];
            if(c in current_dict) {
                current_dict = current_dict[c];
            }
            else {
                current_dict[c] = {};
                current_dict = current_dict[c];
            }
        }
        current_dict[this.END_CHAR] = this.END_CHAR;
    }
}

TrieTree.prototype.match = function(token) {
    let current_dict = this.root;
    for(let j = 0; j < token.length; j++) {
        c = token[j];
        if(c in current_dict !== true) {
            return false;
        }
        current_dict = current_dict[c];
    }
    return (this.END_CHAR in current_dict);
};

TrieTree.prototype.tokenize = function(text) {
    let tokens = [];
    while(text.length > 0) {
        for(let n = this.max_token_length; n > 0; n--) {
            let prefix = text.slice(0, n);
            if(n === 1 || this.match(prefix) === true) {
                tokens.push(prefix);
                text = text.slice(n);
                break;
            }
        }
    }
    return tokens;
};

// 字符串 → 词元编码序列
function encode(text) {
    let tlist = TOKENIZER.trie.tokenize(text);
    let idlist = [];
    let vocab = TOKENIZER.config.stoi;
    for(let i = 0; i < tlist.length; i++) {
        c = tlist[i];
        if(c in vocab) {
            idlist.push(vocab[c]);
        }
        else {
            idlist.push(1); // <|unknown|>
        }
    }
    return idlist;
}

// 词元编码序列 → 字符串
function decode(idlist) {
    let tlist = [];
    for(let i = 0; i < idlist.length; i++) {
        id = idlist[i];
        tlist.push(TOKENIZER.config.itos[id]);
    }
    return tlist.join("");
}


// ===============================================================================
// 采样策略
// ===============================================================================

// 贪心采样：返回概率最大的下标
function sample_argmax(logits, vsize) {
    let max_i = 0;
    let max_p = logits[0];
    for (let i = 1; i < vsize; i++) {
        if (logits[i] > max_p) {
            max_i = i;
            max_p = logits[i];
        }
    }
    return max_i;
}

// 概率采样（香草味的）
function sample_multinomial(prob_dist, n) {
    // sample index from prob_dist, they must sum to 1
    const r = Math.random();
    // const r = 0.5; // TODO
    let cdf = 0.0;
    for (let i = 0; i < n; i++) {
        cdf += prob_dist[i];
        if(cdf > r) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// 概率采样之改进：Top-K采样，只在概率排名前K个词元中采样
function sample_top_k(prob_dist, vsize, k) {
    let probindex = [];
    for (let i = 0; i < vsize; i++) {
        probindex.push({index: i, prob: prob_dist[i]});
    }
    probindex.sort((a, b) => b.prob - a.prob);
    let top_tokens = probindex.slice(0, k);
    // 计算累积概率，用于归一化概率
    let cumulative_prob = 0.0;
    for (let i = 0; i < top_tokens.length; i++) {
        cumulative_prob += top_tokens[i].prob;
    }
    // 在只有前K个词元的列表上执行概率采样
    const r = Math.random() * cumulative_prob;
    let cdf = 0.0;
    for (let i = 0; i < top_tokens.length; i++) {
        cdf += probindex[i].prob;
        if(cdf > r) {
            return probindex[i].index;
        }
    }
    return vsize - 1; // in case of rounding errors
}

// Top-P采样（核采样）：只在累积概率达到p的概率最高的若干个词元中采样
function sample_top_p(probabilities, n, top_p) {
    const cutoff = (1.0 - top_p) / (n - 1);
    let n0 = 0;
    let probindex = [];
    for (let i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex.push({index: i, prob: probabilities[i]});
            n0++;
        }
    }
    probindex.sort((a, b) => b.prob - a.prob);

    // truncate the list where cumulative probability exceeds top_p
    let cumulative_prob = 0.0;
    let last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (let i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > top_p) {
            last_idx = i;
            break; // we've exceeded top_p by including last_idx
        }
    }

    // sample from the truncated list
    const r = Math.random() * cumulative_prob;
    let cdf = 0.0;
    for (let i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if(cdf > r) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

// ===============================================================================
// 文本生成
// ===============================================================================

async function generate(prompt, args, on_ready, on_running, on_finished) {
    if(is_generating) {
        return;
    }
    is_generating = true;

    on_ready();

    let elpased = [];
    let status = "";

    // right now we cannot run for more than cfg.block_size steps
    if (args.max_seq_len <= 0 || args.max_seq_len > LLM.config.block_size) {
        args.max_seq_len = LLM.config.block_size;
    }

    if(args.repetition_penalty <= 0) {
        args.repetition_penalty = 1;
    }

    let idlist = [];
    let prompt_tokens = encode(prompt);
    let next_token = prompt_tokens[0] || 0;
    let pos = 0;

    while (pos < args.max_seq_len) {
        if(is_generating === false) break;

        const t_0 = performance.now();
        llm_forward(next_token, pos, LLM, LoRA, FWD_BUFFER);

        // Pre-fill: if we are still processing the input prompt, force the next prompt token
        if(pos < prompt_tokens.length - 1) {
            status = "Pre-filling...";
            next_token = prompt_tokens[pos + 1];
        }
        // Auto-regressive Decode
        else {
            status = "Decoding...";
            // 复读惩罚：对过往出现过的词元施加惩罚，词元出现得越多，概率越低: ref arxiv:1909.05858
            let tokenset = new Set(idlist);
            for(tk of tokenset.keys()) {
                FWD_BUFFER.logits[tk] /= args.repetition_penalty;
            }

            // 温度采样：当温度设为0时，退化为贪心采样
            if(args.temperature == 0.0) {
                // greedy argmax sampling
                next_token = sample_argmax(FWD_BUFFER.logits, LLM.config.vocab_size);
            }
            else {
                for (let q = 0; q < LLM.config.vocab_size; q++) {
                    FWD_BUFFER.logits[q] /= args.temperature;
                }

                softmax(FWD_BUFFER.logits, LLM.config.vocab_size);

                if(args.top_p > 0 && args.top_p < 1) {
                    next_token = sample_top_p(FWD_BUFFER.logits, LLM.config.vocab_size, args.top_p);
                }
                else if(args.top_k > 0) {
                    next_token = sample_top_k(FWD_BUFFER.logits, LLM.config.vocab_size, args.top_k);
                }
                else {
                    next_token = sample_multinomial(FWD_BUFFER.logits, LLM.config.vocab_size);
                }
            }

            idlist.push(next_token);
        }

        pos++;

        // report achieved tok/s
        const t_1 = performance.now();
        elpased.push(1 / (t_1 - t_0) * 1000);
        let tps_now = elpased.slice(-1)[0];

        let is_continue = on_running(decode(idlist), status, tps_now);
        if(is_continue !== true) break;

        await new Promise(resolve => setTimeout(resolve, 0));
    }

    is_generating = false;

    const tps_avg = elpased.reduce((a, b) => a + b) / elpased.length;
    on_finished(tps_avg);
}

