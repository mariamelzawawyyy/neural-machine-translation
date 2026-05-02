import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# =========================================================
# 1️⃣ CLEAN FUNCTION
# =========================================================
def clean_sentence(s):
    s = s.lower().strip()

    # normalize punctuation
    s = re.sub(r"!+", " ! ", s)
    s = re.sub(r"\?+", " ? ", s)

    # keep only arabic/english letters + basic punctuation
    s = re.sub(r"[^a-zA-Z0-9\u0600-\u06FF\s!?]", "", s)

    # remove extra spaces
    s = re.sub(r"\s+", " ", s)

    return s.strip()


# =========================================================
# 2️⃣ LOAD DATA (Arabic → English)
# =========================================================
def load_data():
    dataset = load_dataset("opus100", "ar-en", split="train[:20000]")

    pairs = []

    for item in dataset:
        ar = item["translation"]["ar"]
        en = item["translation"]["en"]

        ar = clean_sentence(ar)
        en = clean_sentence(en)

        if len(ar) > 0 and len(en) > 0:
            pairs.append((ar, en))

    return pairs


# =========================================================
# 3️⃣ SPECIAL TOKENS
# =========================================================
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# =========================================================
# 4️⃣ VOCAB CLASS
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
# 5️⃣ ENCODING FUNCTIONS
# =========================================================
def encode(vocab, sentence):
    return [vocab.word2idx.get(word, UNK_IDX) for word in sentence.split()]


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

        x = encode(self.vocab, ar)
        y = encode_target(self.vocab, en)

        return torch.tensor(x), torch.tensor(y)


# =========================================================
# 7️⃣ DATALOADER CREATION
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


# =========================================================
# 8️⃣ COLLATE FUNCTION (IMPORTANT for padding)
# =========================================================
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_max = max([len(x) for x in src_batch])
    tgt_max = max([len(x) for x in tgt_batch])

    src_padded = []
    tgt_padded = []

    for src, tgt in zip(src_batch, tgt_batch):

        src = torch.cat([
            src,
            torch.full((src_max - len(src),), PAD_IDX)
        ])

        tgt = torch.cat([
            tgt,
            torch.full((tgt_max - len(tgt),), PAD_IDX)
        ])

        src_padded.append(src)
        tgt_padded.append(tgt)

    return torch.stack(src_padded), torch.stack(tgt_padded) 



