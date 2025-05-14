# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import struct

# this is a horrible gpt-2 unicode byte encoder hack from https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
# this has poisoned all HF tokenizer configs that use ByteLevel decoder/preprocessor
# as a result we get crazy UTF-8-as-bytes-as-UTF8 in the tokenizer data that we need to convert back
def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def export():

    with open("/home/bd4sur/ai/_model/Qwen3/Qwen3-0.6B/tokenizer.json", "r") as f:
        tokenizer = json.load(f)

    tokens = [""] * 151936
    scores = [0] * 151936

    vocab = tokenizer["model"]["vocab"]

    tokens_gpt2 = not tokenizer["model"].get("byte_fallback", False)

    for t, i in vocab.items():
        tokens[i] = t

    for added in tokenizer["added_tokens"]:
        tokens[added["id"]] = added["content"]

    # compute score as negative merge index so that earlier merges get selected first
    for i, m in enumerate(tokenizer["model"]["merges"]):
        t1, t2 = m[0], m[1] # Qwen3
        ti = vocab[t1 + t2]
        if scores[ti] == 0:
            scores[ti] = -(1 + i)

    # postprocess tokens
    gpt2_decode = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    for i, t in enumerate(tokens):
        if tokens_gpt2:
            b = bytes([gpt2_decode.get(c, 0) for c in t])
        else:
            # t = t.replace('\u2581', ' ') # sentencepiece uses this character as whitespace
            b = t.encode('utf-8')

        b = b.replace(b"\0", b"\7") # replace null bytes with bell characters
        assert b.count(0) == 0 # no null bytes allowed

        tokens[i] = b

    # record the max token length
    max_token_length = max(len(t) for t in tokens)

    # write to a binary file
    # the tokenizer.bin file is the same as .model file, but .bin
    with open("qwen3-tokenizer.bin", 'wb') as f:
        f.write(struct.pack("I", max_token_length))
        for byte, score in zip(tokens, scores):
            f.write(struct.pack("fI", score, len(byte)))
            f.write(byte)

if __name__ == "__main__":
    export()
