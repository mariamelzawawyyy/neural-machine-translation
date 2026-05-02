import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

# =========================================================
# 1️⃣ SPECIAL TOKENS
# =========================================================
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# =========================================================
# 2️⃣ CLEAN FUNCTION
# =========================================================
def clean_sentence(s):
    s = s.lower().strip()

    s = re.sub(r"!+", " ! ", s)
    s = re.sub(r"\?+", " ? ", s)

    s = re.sub(r"[^a-zA-Z0-9\u0600-\u06FF\s!?]", "", s)

    s = re.sub(r"\s+", " ", s)

    return s.strip()

# =========================================================
# 3️⃣ LOAD DATA (Arabic → English)
# =========================================================
def load_data():
    dataset = load_dataset("opus100", "ar-en", split="train[:20000]")

    pairs = []

    for item in dataset:
        ar = clean_sentence(item["translation"]["ar"])
        en = clean_sentence(item["translation"]["en"])

        if len(ar) > 0 and len(en) > 0:
            pairs.append((ar, en))

    return pairs

# =========================================================
# 4️⃣ VOCAB
# =========================================================
class Vocab:
    def __init__(self):
        self.word2idx = {
            "<PAD>": PAD_IDX,
            "<SOS>": SOS_IDX,
            "<EOS>": EOS_IDX,
            "<UNK>": UNK_IDX
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.count = 4

    def build_vocab(self, pairs):
        for ar, en in pairs:
            for word in (ar + " " + en).split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.count
                    self.idx2word[self.count] = word
                    self.count += 1

# =========================================================
# 5️⃣ ENCODING
# =========================================================
def encode(vocab, sentence):
    return [vocab.word2idx.get(w, UNK_IDX) for w in sentence.split()]

def encode_target(vocab, sentence):
    return [SOS_IDX] + encode(vocab, sentence) + [EOS_IDX]

# =========================================================
# 6️⃣ DATASET
# =========================================================
class TranslationDataset(Dataset):
    def __init__(self, pairs, vocab):
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ar, en = self.pairs[idx]

        src = torch.tensor(encode(self.vocab, ar))
        trg = torch.tensor(encode_target(self.vocab, en))

        return src, trg

# =========================================================
# 7️⃣ COLLATE FUNCTION (FIXED)
# =========================================================
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=PAD_IDX
    )

    trg_batch = pad_sequence(
        trg_batch,
        batch_first=True,
        padding_value=PAD_IDX
    )

    return src_batch, trg_batch

# =========================================================
# 8️⃣ LOADERS
# =========================================================
def get_loaders(batch_size=32):
    pairs = load_data()

    vocab = Vocab()
    vocab.build_vocab(pairs)

    dataset = TranslationDataset(pairs, vocab)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return loader, vocab

 
