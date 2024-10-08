import os
import json
import collections
from typing import Optional
import base64
import regex
import blobfile
from tqdm import tqdm

class BPE_Tokenizer:
    def __init__(self, *, pat_str: str, mergeable_ranks: dict[bytes, int], vocab_size: int) -> None:
        """Creates an Encoding object."""
        # A regex pattern string that is used to split the input text
        self.pat_str = pat_str
        # A dictionary mapping token bytes to their ranks. The ranks correspond to merge priority
        self.mergeable_ranks = mergeable_ranks
        self.vocab_size = vocab_size

        self._decoder = {token: token_bytes for token_bytes, token in mergeable_ranks.items()}
        self._pat = regex.compile(pat_str)

    def encode(self, text: str, visualise: Optional[bool] = False) -> list[int]:
        # Use the regex to split the text into (approximately) words
        words = self._pat.findall(text)
        tokens = []
        for word in words:
            # Turn each word into tokens, using the byte pair encoding algorithm
            word_bytes = word.encode("utf-8")
            word_tokens = bpe_encode(self.mergeable_ranks, word_bytes, visualise=visualise)
            tokens.extend(word_tokens)
        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        return b"".join(self._decoder[token] for token in tokens)

    def decode(self, tokens: list[int]) -> str:
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

    def decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]:
        return [self._decoder[token] for token in tokens]

    def dump(self, bpe_file_path):
        with blobfile.BlobFile(bpe_file_path, "wb") as f:
            f.write(self.vocab_size)
            f.write(self.pat_str)
            for token, rank in sorted(self.mergeable_ranks.items(), key=lambda x: x[1]):
                f.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")

    @staticmethod
    def train(training_data: str, vocab_size: int, pat_str: str):
        mergeable_ranks = bpe_train(data=training_data, vocab_size=vocab_size, pat_str=pat_str, visualise=True)
        return BPE_Tokenizer(pat_str=pat_str, mergeable_ranks=mergeable_ranks, vocab_size=vocab_size)

    @staticmethod
    def load(bpe_file_path):
        with blobfile.BlobFile(bpe_file_path, "rb") as f:
            vocab_size = int(f.readline())
            pat_str = str(f.readline())
            contents = f.read()
            mergeable_ranks = {
                base64.b64decode(token): int(rank)
                for token, rank in (line.split() for line in contents.splitlines() if line)
            }
            return BPE_Tokenizer(pat_str=pat_str, mergeable_ranks=mergeable_ranks, vocab_size=vocab_size)

def bpe_encode(mergeable_ranks: dict[bytes, int], input: bytes, visualise: Optional[bool] = False) -> list[int]:
    parts = [bytes([b]) for b in input]
    while True:
        if visualise:
            visualise_tokens(parts)

        # Iterate over all pairs and find the pair we want to merge the most
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # If there were no pairs we could merge, we're done!
        if min_rank is None:
            break
        assert min_idx is not None

        # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]

    if visualise:
        print()

    tokens = [mergeable_ranks[part] for part in parts]
    return tokens


def bpe_train(data: str, vocab_size: int, pat_str: str, visualise: Optional[bool] = False) -> dict[bytes, int]:
    # First, add tokens for each individual byte value
    if vocab_size < 2**8:
        raise ValueError("vocab_size must be at least 256, so we can encode all bytes")
    ranks = {}
    for i in range(2**8):
        ranks[bytes([i])] = i

    # Splinter up our data into lists of bytes
    # data = "Hello world"
    # words = [
    #     [b'H', b'e', b'l', b'l', b'o'],
    #     [b' ', b'w', b'o', b'r', b'l', b'd']
    # ]
    words: list[list[bytes]] = [
        [bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, data)
    ]

    # Now, use our data to figure out which merges we should make
    while len(ranks) < vocab_size:
        # Find the most common pair. This will become our next token
        stats = collections.Counter()
        for piece in words:
            for pair in zip(piece[:-1], piece[1:]):
                stats[pair] += 1

        most_common_pair = max(stats, key=lambda x: stats[x])
        token_bytes = most_common_pair[0] + most_common_pair[1]
        token = len(ranks)
        # Add the new token!
        ranks[token_bytes] = token

        # Now merge that most common pair in all the words. That is, update our training data
        # to reflect our decision to make that pair into a new token.
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == most_common_pair:
                    # We found our pair! Merge it
                    new_word.append(token_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if i == len(word) - 1:
                new_word.append(word[i])
            new_words.append(new_word)
        words = new_words

        # See the intermediate merges play out!
        if visualise:
            print(f"The current most common pair is {most_common_pair[0]} + {most_common_pair[1]}")
            print(f"So we made {token_bytes} our {len(ranks)}th token")
            print("Now the first fifty words in our training data look like:")
            visualise_tokens([token for word in words[:50] for token in word])
            print("\n")

    return ranks


def visualise_tokens(token_values: list[bytes]) -> None:
    background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]
    # If token boundaries do not occur at unicode character boundaries, it's unclear how best to
    # visualise the token. Here, we'll just use the unicode replacement character to represent some
    # fraction of a character.
    unicode_token_values = [x.decode("utf-8", errors="replace") for x in token_values]

    running_length = 0
    last_color = None
    for token in unicode_token_values:
        color = background[running_length % len(background)]
        if color == last_color:
            color = background[(running_length + 1) % len(background)]
            assert color != last_color
        last_color = color
        running_length += len(token)
        print(color + token, end="")
    print("\u001b[0m")


def train_simple_encoding():
    gpt2_pattern = (
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    with open("/home/bd4sur/ai/Nano/dataset/pretrain-psycho.txt", "r") as f:
        data = f.read()

    enc = BPE_Tokenizer.train(data, vocab_size=4096, pat_str=gpt2_pattern)
    # enc = BPE_Tokenizer.load("/home/bd4sur/ai/Nano/dataset/cl100k_base.txt")

    enc.dump("/home/bd4sur/ai/Nano/dataset/bpe.txt")

    inputstr = """以前（初中、小学）老师形容我们听不懂课或者神游卖呆的状态就用“鸭子听雷”，以前不懂，前几天懂了。这个说法好像分布很广"""
    tokens = enc.encode(inputstr)
    print(len(inputstr))
    print(len(tokens))
    print(tokens)
    print(enc.decode(tokens))

    return enc

class Tokenizer:
    def __init__(self, special_tokens=None):
        # self.bpe_tokenizer = BPE_Tokenizer.load(os.path.join(os.path.dirname(__file__), 'dataset/cl100k_base.txt'))
        # self.bpe_tokenizer = tiktoken.get_encoding("cl100k_base")
        self.config = None
        self.stoi = {}
        self.itos = []
        self.vocab_size = 0 # self.bpe_tokenizer.vocab_size
        self.special_tokens = special_tokens if special_tokens is not None else {
            "<|padding|>": 0,
            "<|unknown|>": 1,
            "<|bos|>": 2,
            "<|eos|>": 3,
            "<|instruct_mark|>": 4,
            "<|response_mark|>": 5,
            "BD4SUR": 6
        }

    # encoder: take a string, output a list of integers
    def encode(self, text):
        # return self.bpe_tokenizer.encode(text)
        placeholder = [
            "\ufff0", "\ufff1", "\ufff2", "\ufff3", "\ufff4", "\ufff5", "\ufff6", "\ufff7",
            "\ufff8", "\ufff9", "\ufffa", "\ufffb", "\ufffc", "\ufffd", "\ufffe", "\uffff"
        ]
        for i,st in enumerate(self.special_tokens):
            text = text.replace(st, placeholder[i])
        tokens = []
        for c in text:
            if c in placeholder:
                tokens.append(ord(c) - 65520) # 65520 == ord("\ufff0")
            elif c in self.stoi:
                tokens.append(self.stoi[c])
            else:
                tokens.append(self.special_tokens["<|unknown|>"])
        return tokens

    # decoder: take a list of integers, output a string
    def decode(self, token_list):
        # return self.bpe_tokenizer.decode(token_list)
        return ''.join([self.itos[i] for i in token_list])

    def load_from_config_file(self, config_path):
        # self.bpe_tokenizer = BPE_Tokenizer.load(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            self.config = config
            self.vocab_size = config["vocab_size"]
            self.stoi = config["stoi"]
            self.itos = config["itos"]
            self.special_tokens = config["special_tokens"]

    def load_from_config_dict(self, config_dict):
        self.config = config_dict
        self.vocab_size = config_dict["vocab_size"]
        self.stoi = config_dict["stoi"]
        self.itos = config_dict["itos"]
        self.special_tokens = config_dict["special_tokens"]

    def dump_config_file(self, config_path):
        with open(config_path, 'w') as f:
            json.dump(self.config, f)

    # 根据已有文本建立编码器，并保存到配置文件
    def build_from_text(self, text, config_path):
        # gpt2_pattern = (r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # self.bpe_tokenizer = BPE_Tokenizer.train(text, vocab_size=300, pat_str=gpt2_pattern)
        # self.bpe_tokenizer.dump(config_path)
        charset = sorted(list(set(text)))

        itos = [0] * len(self.special_tokens)
        for i,c in enumerate(self.special_tokens):
            itos[i] = c

        itos = itos + sorted(list(charset))

        self.vocab_size = len(itos)
        print(f"  Vocab size: {self.vocab_size}")

        self.itos = itos
        self.stoi = { ch:i for i,ch in enumerate(itos) }

        self.config = {
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": self.itos,
            "special_tokens": self.special_tokens
        }
        with open(config_path, 'w') as f:
            json.dump(self.config, f)

    def build_from_files(self, text_path_list, config_path):
        def read_chunk(filepath, chunk_size=65536):
            with open(filepath, mode="r", encoding="utf-8") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        return
                    yield chunk

        charset = set({})
        chunk_size = 100000

        for textpath in text_path_list:
            print(f"Processing `{textpath}`")
            print(f"  Calculating character number ... ", end="")
            with os.popen(f"wc -m {textpath}") as f:
                res = f.readlines()[0]
                char_num = int(res.split(" ")[0])
            print(f"{char_num:,}")

            print(f"  Collecting all unique Unicode characters in `{textpath}` ... ")
            text_iterator = read_chunk(textpath, chunk_size=chunk_size)
            for chunk in tqdm(text_iterator, total=int(char_num/chunk_size)):
                charset = charset.union(set(chunk))

        itos = [0] * len(self.special_tokens)
        for i,c in enumerate(self.special_tokens):
            itos[i] = c

        itos = itos + sorted(list(charset))

        self.vocab_size = len(itos)
        print(f"  Vocab size: {self.vocab_size}")

        self.itos = itos
        self.stoi = { ch:i for i,ch in enumerate(itos) }

        self.config = {
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": self.itos,
            "special_tokens": self.special_tokens
        }
        with open(config_path, 'w') as f:
            json.dump(self.config, f, ensure_ascii=True)


if __name__ == "__main__":
    # train_simple_encoding()

    text_files = [
        "/home/bd4sur/ai/Nano/dataset/pretrain-psycho.txt",
        "/home/bd4sur/ai/Nano/dataset/pretrain-chinese-classic.txt",
        "/home/bd4sur/ai/Nano/dataset/pretrain-amateur-radio.txt",
    ]
    tk = Tokenizer()
    tk.build_from_files(text_files, "/home/bd4sur/ai/Nano/tk.json")

    tokens = tk.encode("<|bos|><|instruct_mark|>人类（包括BD4SUR的操作员）的<|unknown|>本质是<|response_mark|>复读机！<|eos|><|padding|><|padding|>")
    print(tokens)
    txt = tk.decode(tokens)
    print(txt)
