"""
This contains base Tokenizer class with common helper functions
"""

import unicodedata


def merge(ids, pair, idx):
    """
    wherever there is a pair replace with new token index
    """

    newids = []
    i = 0

    while i < len(ids):
        if ids[i] == pair[0] and i + 1 < len(ids) and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(i)
            i += 1

    return newids


def get_stats(ids, counts=None):
    """
    return a dict of counts given a list of integers
    and also allow to update counts
    """
    counts = {} if counts is None else counts

    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts


def replace_control_characters(s: str) -> str:
    # we want to remove control chars as they distort the output
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []

    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")  # escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    s = t.decode("utf-8", errors="replace")
    s = replace_control_characters(s)
    return s


# the base tokenizer class
class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # initializes default vocab size of 256(# of bytes), no merges, no patterns
        self.merges = {}  # (int,int) -> int
        self.special_tokens = {}  # str -> int
        self.vocab = self._build_vocab()  # int -> bytes
        self.pattern = ""  # str

    def train(self, text, vocab_size, debug=False):
        raise NotImplementedError

    def encode(self, text):
        # string to list of integers
        raise NotImplementedError

    def decode(self, ids):
        # list of integers to string
        raise NotImplementedError

    def _build_vocab(self):
        vocab = {i: bytes([i]) for i in range(256)}

        for pair, idx in self.merges.items():
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        return vocab

    def save(self, file_prefix):
        """
        saves file_prefix.model and file_prefix.vocab
        - model file is used for load()
        - vocab file is just for human use only
        """
        model_file = file_prefix + ".model"
        with open(model_file, "w") as f:
            f.write("bpe v1\n")
            f.write(f"{self.pattern}\n")

            # write num of special tokens, and then each one
            f.write(f"{len(self.special_tokens)}\n")

            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

            # merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        # write the vocab file -- readability sake not used
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}

        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():

                s = render_token(token)

                if idx in inverted_merges:

                    idx0, idx1 = inverted_merges[idx]

                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])

                    f.write(f"[{s0}][{s1}] -> [{s}]{idx}\n")
                else:
                    f.write(f"[{s}]{idx}\n")

    def load(self, model_file):
        """Inverse of save() for model file"""

        assert model_file.endswith(".model")

        special_tokens = {}
        merges = {}
        vocab_idx = 256

        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()

            assert version == "bpe v1"

            self.pattern = f.readline().strip()

            num_special = int(f.readline().strip())

            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)

            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = vocab_idx
                vocab_idx += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
