import torch
import torch.nn as nn

from model import Encoder, Decoder, Seq2Seq
from data import load_data, Vocab, TranslationDataset, PAD_IDX, collate_fn
from torch.utils.data import DataLoader

# =========================================================
# 1️⃣ DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# memory cleanup
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# =========================================================
# 2️⃣ LOAD DATA + VOCAB
# =========================================================
pairs = load_data()

vocab = Vocab()
vocab.build_vocab(pairs)

dataset = TranslationDataset(pairs, vocab)

# =========================================================
# 3️⃣ DATALOADER
# =========================================================
train_loader = DataLoader(
    dataset,
    batch_size=16,   # مناسب للـ GPU
    shuffle=True,
    collate_fn=collate_fn
)

# =========================================================
# 4️⃣ HYPERPARAMETERS
# =========================================================
vocab_size = len(vocab.word2idx)

embedding_size = 192
hidden_size = 96
attn_dim = 192
num_layers = 1

num_epochs = 10
lr = 0.001

# =========================================================
# 5️⃣ MODEL
# =========================================================
encoder = Encoder(
    vocab_size,
    embedding_size,
    hidden_size,
    num_layers,
    PAD_IDX
)

decoder = Decoder(
    vocab_size,
    embedding_size,
    hidden_size,
    num_layers,
    attn_dim
)

model = Seq2Seq(encoder, decoder).to(device)

# =========================================================
# 6️⃣ LOSS + OPTIMIZER
# =========================================================
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# =========================================================
# 7️⃣ TRAIN LOOP
# =========================================================
for epoch in range(num_epochs):

    # 🔥 Teacher Forcing Schedule (مهم جدًا)
    teacher_forcing_ratio = max(0.5 * (0.95 ** epoch), 0.1)

    model.train()
    total_loss = 0

    torch.cuda.empty_cache()

    for src, trg in train_loader:

        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg, teacher_forcing_ratio)

        # reshape
        output = output[:, 1:].reshape(-1, vocab_size)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)

        loss.backward()

        # 🔥 Gradient Clipping (مهم)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | TF: {teacher_forcing_ratio:.3f}") 
