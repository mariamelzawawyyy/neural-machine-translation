import torch
import torch.nn as nn
import random


# =========================================================
# 1️⃣ ENCODER
# =========================================================
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, padding_idx):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x: [batch_size, seq_len]

        x = self.embedding(x)

        outputs, (h, c) = self.lstm(x)

        # merge forward + backward
        h = torch.cat([h[-2], h[-1]], dim=1)
        c = torch.cat([c[-2], c[-1]], dim=1)

        h = self.fc_hidden(h).unsqueeze(0)
        c = self.fc_cell(c).unsqueeze(0)

        return outputs, h, c


# =========================================================
# 2️⃣ ATTENTION (Bahdanau)
# =========================================================
class Attention(nn.Module):
    def __init__(self, enc_dim, dec_hidden, attn_dim):
        super().__init__()

        self.W_h = nn.Linear(enc_dim, attn_dim)
        self.W_s = nn.Linear(dec_hidden, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

        self.tanh = nn.Tanh()

    def forward(self, decoder_hidden, encoder_outputs):

        # encoder_outputs: [B, T, enc_dim]
        # decoder_hidden: [B, H]

        enc_proj = self.W_h(encoder_outputs)
        dec_proj = self.W_s(decoder_hidden).unsqueeze(1)

        energy = self.tanh(enc_proj + dec_proj)

        attn_scores = self.v(energy).squeeze(2)

        attn_weights = torch.softmax(attn_scores, dim=1)

        attn_weights = attn_weights.unsqueeze(1)

        context = torch.bmm(attn_weights, encoder_outputs)

        context = context.squeeze(1)

        return context, attn_weights


# =========================================================
# 3️⃣ DECODER
# =========================================================
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, attn_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.attention = Attention(
            enc_dim=hidden_size * 2,
            dec_hidden=hidden_size,
            attn_dim=attn_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_size + hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):

        # x: [B]
        x = x.unsqueeze(1)

        emb = self.embedding(x)

        dec_hidden = hidden[-1]

        context, attn = self.attention(dec_hidden, encoder_outputs)

        context = context.unsqueeze(1)

        lstm_input = torch.cat((emb, context), dim=2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell, attn


# =========================================================
# 4️⃣ SEQ2SEQ MODEL
# =========================================================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        x = trg[:, 0]  # <SOS>

        for t in range(1, trg_len):

            output, hidden, cell, _ = self.decoder(
                x, hidden, cell, encoder_outputs
            )

            outputs[:, t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            x = trg[:, t] if teacher_force else top1

        return outputs