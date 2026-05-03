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

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# =========================================================
# 2️⃣ DATA
# =========================================================
pairs = load_data()

vocab = Vocab()
vocab.build_vocab(pairs)

dataset = TranslationDataset(pairs, vocab)

train_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

# =========================================================
# 3️⃣ HYPERPARAMETERS (IMPROVED)
# =========================================================
vocab_size = len(vocab.word2idx)

embedding_size = 256       
hidden_size = 256          
attn_dim = 256
num_layers = 1

num_epochs = 20            
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
# 6️⃣ TRAIN LOOP
# =========================================================
for epoch in range(num_epochs):

    model.train()
    total_loss = 0

     
    teacher_forcing_ratio = max(0.7 * (0.95 ** epoch), 0.2)

    for src, trg in train_loader:

        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg, teacher_forcing_ratio)

        # remove <SOS>
        output = output[:, 1:].reshape(-1, vocab_size)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)

        loss.backward()

      
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Loss: {avg_loss:.4f} | "
        f"TF: {teacher_forcing_ratio:.3f}"
    ) 
