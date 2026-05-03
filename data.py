import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

# =========================================================
# 1️⃣ TOKENS
# =========================================================
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# =========================================================
# 2️⃣ CLEAN
# =========================================================
def clean_sentence(s):
    s = s.lower().strip()

    s = re.sub(r"!+", " ! ", s)
    s = re.sub(r"\?+", " ? ", s)
    s = re.sub(r"[^a-zA-Z0-9\u0600-\u06FF\s!?]", "", s)
    s = re.sub(r"\s+", " ", s)

    return s.strip()

# =========================================================
# 3️⃣ LOAD DATA
# =========================================================
def load_data(split="train", max_samples=20000):
    dataset = load_dataset("opus100", "ar-en", split=f"{split}[:{max_samples}]")

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
            for w in (ar + " " + en).split():
                if w not in self.word2idx:
                    self.word2idx[w] = self.count
                    self.idx2word[self.count] = w
                    self.count += 1

# =========================================================
# 5️⃣ ENCODING
# =========================================================
def encode(vocab, sentence):
    return [vocab.word2idx.get(w, UNK_IDX) for w in sentence.split()]

def encode_src(vocab, sentence):
    return encode(vocab, sentence) + [EOS_IDX]

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

        src = torch.tensor(encode_src(self.vocab, ar))
        trg = torch.tensor(encode_target(self.vocab, en))

        return src, trg

# =========================================================
# 7️⃣ COLLATE
# =========================================================
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=PAD_IDX)

    return src_batch, trg_batch

# =========================================================
# 8️⃣ LOADERS (TRAIN / VAL / TEST)
# =========================================================
def get_loaders(batch_size=32, max_train=20000, max_test=2000):

    # ================= TRAIN =================
    train_pairs = load_data("train", max_train)

    vocab = Vocab()
    vocab.build_vocab(train_pairs)

    full_dataset = TranslationDataset(train_pairs, vocab)

    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ================= TEST =================
    test_pairs = load_data("test", max_test)
    test_dataset = TranslationDataset(test_pairs, vocab)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, vocab 

 
