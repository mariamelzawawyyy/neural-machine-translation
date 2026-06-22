import os
import re
import codecs
import sentencepiece as spm
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

SAVE_DIR = "/content/opus_nmt_data"
os.makedirs(SAVE_DIR, exist_ok=True)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"    

print("⏳ Loading Opus100 dataset...")
  
dataset = load_dataset("opus100", "ar-en", split='train')

num_samples = min(50000, len(dataset)) 
raw_samples = dataset.select(range(num_samples))

temp_raw_file = os.path.join(SAVE_DIR, "opus_raw.txt")

with codecs.open(temp_raw_file, 'w', encoding='utf-8') as f:
    for item in raw_samples:
        en = item['translation']['en'].replace('\n', ' ').strip()
        ar = item['translation']['ar'].replace('\n', ' ').strip()
        f.write(f"{en}\t{ar}\n")

pairs = []
with codecs.open(temp_raw_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            en = parts[0].lower().strip()
            ar = parts[1].strip()

            if en and ar and re.search(r'[\u0600-\u06FF]', ar) and 3 < len(en.split()) < 20:
                pairs.append(f"{en}\n{ar}")

pairs.sort(key=lambda x: len(x.split('\n')[0].split()))

tokenizer_train_file = os.path.join(SAVE_DIR, "tokenizer_input.txt")
with open(tokenizer_train_file, "w", encoding="utf-8") as f:
    f.write("\n".join(pairs))

print("🎓 Training Tokenizer...")
spm.SentencePieceTrainer.train(
    input=tokenizer_train_file,
    model_prefix=os.path.join(SAVE_DIR, "spm"),
    vocab_size=9000,
    model_type="unigram",
    byte_fallback=True, 
    pad_id=0, bos_id=1, eos_id=2, unk_id=3
)

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(SAVE_DIR, "spm.model"))

VOCAB_SIZE = sp.get_piece_size()
PAD_IDX = 0

class TranslationDataset(Dataset):
    def __init__(self, data_list):
        self.samples = []
        for item in data_list:
            parts = item.split('\n')
            if len(parts) == 2:
                self.samples.append((parts[0], parts[1]))

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        en, ar = self.samples[idx]
        
        MAX_LEN = 50 
        src_tokens = sp.encode(en)[:MAX_LEN]
        trg_tokens = sp.encode(ar)[:MAX_LEN]
        
        src = torch.tensor([1] + src_tokens + [2])
        trg = torch.tensor([1] + trg_tokens + [2])
        
        return src, trg, en, ar

def collate_fn(batch):
    src, trg, en, ar = zip(*batch)

    return (
        pad_sequence(src, batch_first=True, padding_value=PAD_IDX),
        pad_sequence(trg, batch_first=True, padding_value=PAD_IDX),
        en,
        ar
    )

full_dataset = TranslationDataset(pairs)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_data, val_data = random_split(
    full_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

print(f"✅ Ready! Data: {len(full_dataset)} | Vocab Size: {VOCAB_SIZE}") 
