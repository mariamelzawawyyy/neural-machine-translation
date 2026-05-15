import os
import torch
import sentencepiece as spm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from model_seq2seq import Encoder, Decoder, Seq2Seq
from data_preprocessing import VOCAB_SIZE, PAD_IDX, full_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp = spm.SentencePieceProcessor()
sp.load("/content/opus_nmt_data/spm.model")

embedding_size = 128
hidden_size = 128
num_layers = 2
attention_dim = 128

encoder = Encoder(VOCAB_SIZE, embedding_size, hidden_size, num_layers, PAD_IDX, dropout=0.5).to(device)
decoder = Decoder(embedding_size, VOCAB_SIZE, hidden_size, num_layers, attention_dim, PAD_IDX, dropout=0.5).to(device)
model = Seq2Seq(encoder, decoder).to(device)

MODEL_PATH = "/content/drive/MyDrive/nmt-seq2seq-attention/nmt_best_model.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ Successfully loaded the best model from Drive!")
else:
    model.load_state_dict(torch.load("nmt_best_model.pth", map_location=device))
    print("⚠️ Drive model not found, loaded local best model.")

model.eval()

def translate_sentence(sentence, max_len=50, beam_width=3):
    tokens = sp.encode(sentence.lower())
    src = torch.tensor([1] + tokens + [2]).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)

    sequences = [([1], 0.0, hidden)]

    for _ in range(max_len):
        all_candidates = []
        
        for seq, score, h in sequences:
            if seq[-1] == 2:
                all_candidates.append((seq, score, h))
                continue
                
            trg_tensor = torch.tensor([seq[-1]]).to(device)

            with torch.no_grad():
                output, new_hidden, _ = model.decoder(trg_tensor, h, encoder_outputs)

            probs = torch.log_softmax(output, dim=1)
            topk = torch.topk(probs, beam_width)

            for i in range(beam_width):
                token = topk.indices[0][i].item()
                new_score = score + topk.values[0][i].item()
                all_candidates.append((seq + [token], new_score, new_hidden))

        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    best_seq = sequences[0][0]
    clean_tokens = [t for t in best_seq if t not in [0, 1, 2]]
    return sp.decode(clean_tokens)

def evaluate_bleu(n_samples=100):
    smoothie = SmoothingFunction().method4
    scores = []

    print(f"📊 Calculating BLEU Score for {n_samples} samples...")
    for i in range(n_samples):
        _, _, src_text, trg_text = full_dataset[i]
        
        pred_text = translate_sentence(src_text)

        reference = [trg_text.split()]
        candidate = pred_text.split()

        score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        scores.append(score)
        
        if i % 20 == 0:
            print(f"   [Sample {i}] Src: {src_text} | Pred: {pred_text}")

    print(f"\n🔥 Final Avg BLEU Score: {(sum(scores)/len(scores)) * 100:.2f}%")

if __name__ == "__main__":
    evaluate_bleu()