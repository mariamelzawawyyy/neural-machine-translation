import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, padding_idx, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        
        self.gru = nn.GRU(
            embedding_size, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        encoder_outputs, hidden = self.gru(embedded)
        
        B = x.size(0)
        hidden = hidden.view(self.num_layers, 2, B, self.hidden_size)
        combined_hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=2)
        decoder_init_hidden = torch.tanh(self.fc_hidden(combined_hidden))
        
        return encoder_outputs, decoder_init_hidden

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        super().__init__()
        self.W_h = nn.Linear(enc_hidden_dim, attn_dim)
        self.W_s = nn.Linear(dec_hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_mask=None):
        enc_proj = self.W_h(encoder_outputs)
        dec_proj = self.W_s(decoder_hidden).unsqueeze(1)
        
        energy = torch.tanh(enc_proj + dec_proj)
        attn_scores = self.v(energy).squeeze(2)
        
        if src_mask is not None:
            attn_scores = attn_scores.masked_fill(src_mask == 0, float('-inf'))
            
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, num_layers, attn_dim, padding_idx, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_size * 2, hidden_size, attn_dim)
        
        self.gru = nn.GRU(
            embedding_size + hidden_size * 2, hidden_size,
            num_layers=num_layers, batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_size + hidden_size * 2 + embedding_size, vocab_size)

    def forward(self, x, hidden, encoder_outputs, src_mask=None):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        
        context, attn_weights = self.attention(hidden[-1], encoder_outputs, src_mask)
        
        gru_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, hidden = self.gru(gru_input, hidden)
        
        output = output.squeeze(1)
        context = context
        embedded = embedded.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        
        return prediction, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, bos_idx=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bos_idx = bos_idx

    def create_mask(self, src):
        return (src != 0)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        B = src.size(0)
        trg_len = trg.size(1)
        outputs = torch.zeros(B, trg_len, self.decoder.fc_out.out_features).to(src.device)
        
        encoder_outputs, hidden = self.encoder(src)
        src_mask = self.create_mask(src)
        
        x = torch.full((B,), self.bos_idx, device=src.device)
        
        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(x, hidden, encoder_outputs, src_mask)
            outputs[:, t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            x = trg[:, t] if teacher_force else output.argmax(1)
            
        return outputs  
