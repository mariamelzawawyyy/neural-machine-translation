import torch
import torch.nn as nn

from model import Encoder, Decoder, Seq2Seq
from data import (
    get_loaders,
    PAD_IDX
)

# =========================================================
# 1️⃣ DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.cuda.empty_cache()

# =========================================================
# 2️⃣ LOAD DATA (TRAIN / VAL / TEST)
# =========================================================
train_loader, val_loader, test_loader, vocab = get_loaders(
    batch_size=16,
    max_train=20000,
    max_test=2000
)

# =========================================================
# 3️⃣ MODEL PARAMS
# =========================================================
vocab_size = len(vocab.word2idx)

embedding_size = 256
hidden_size = 256
attn_dim = 256
num_layers = 1

num_epochs = 10
lr = 3e-4

# =========================================================
# 4️⃣ MODEL
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
# 5️⃣ LOSS + OPTIMIZER
# =========================================================
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# =========================================================
# 6️⃣ TRAIN FUNCTION
# =========================================================
def train_epoch():
    model.train()
    total_loss = 0

    teacher_forcing_ratio = 0.5

    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg, teacher_forcing_ratio)

        output = output[:, 1:].reshape(-1, vocab_size)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# =========================================================
# 7️⃣ VALIDATION FUNCTION
# =========================================================
def evaluate(loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, teacher_forcing_ratio=0)

            output = output[:, 1:].reshape(-1, vocab_size)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================
# 8️⃣ TRAIN LOOP
# =========================================================
for epoch in range(num_epochs):

    train_loss = train_epoch()
    val_loss = evaluate(val_loader)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

# =========================================================
# 9️⃣ TEST FINAL
# =========================================================
test_loss = evaluate(test_loader)
print(f"\nFinal Test Loss: {test_loss:.4f}") 
