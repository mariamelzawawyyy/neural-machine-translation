import torch
import torch.nn as nn
import random

# =========================================================
# 1️⃣ ENCODER
# =========================================================
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, padding_idx):
        super().__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x: [B, T]

        embedded = self.embedding(x)  # [B, T, E]

        encoder_outputs, (h, c) = self.lstm(embedded)

        # h, c: [num_layers*2, B, H]

        h_forward = h[-2]
        h_backward = h[-1]
        c_forward = c[-2]
        c_backward = c[-1]

        h = torch.cat((h_forward, h_backward), dim=1)
        c = torch.cat((c_forward, c_backward), dim=1)

        h = torch.tanh(self.fc_hidden(h)).unsqueeze(0)
        c = torch.tanh(self.fc_cell(c)).unsqueeze(0)

        return encoder_outputs, h, c


# =========================================================
# 2️⃣ ATTENTION
# =========================================================
class Attention(nn.Module):
    def __init__(self, enc_dim, dec_hidden, attn_dim):
        super().__init__()

        self.W_h = nn.Linear(enc_dim, attn_dim)
        self.W_s = nn.Linear(dec_hidden, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_mask):

        enc_proj = self.W_h(encoder_outputs)
        dec_proj = self.W_s(decoder_hidden).unsqueeze(1)

        energy = torch.tanh(enc_proj + dec_proj)

        scores = self.v(energy).squeeze(2)
 
        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=1)

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)

        return context, attn_weights


# =========================================================
# 3️⃣ DECODER
# =========================================================
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, attn_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.attention = Attention(hidden_size * 2, hidden_size, attn_dim)

        self.lstm = nn.LSTM(
            embedding_size + hidden_size * 2,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs, src_mask):

        x = x.unsqueeze(1)  # [B] → [B,1]

        emb = self.embedding(x)  # [B,1,E]

        dec_hidden = hidden[-1]

        context, attn = self.attention(dec_hidden, encoder_outputs, src_mask)

        context = context.unsqueeze(1)

        lstm_input = torch.cat((emb, context), dim=2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell, attn


# =========================================================
# 4️⃣ SEQ2SEQ
# =========================================================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx=0):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)

        encoder_outputs, hidden, cell = self.encoder(src)

       
        src_mask = (src != self.pad_idx)

        x = trg[:, 0]  

        for t in range(1, trg_len):

            output, hidden, cell, _ = self.decoder(
                x, hidden, cell, encoder_outputs, src_mask
            )

            outputs[:, t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            x = trg[:, t] if teacher_force else top1

        return outputs 
