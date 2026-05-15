import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc  
import shutil 

from model_seq2seq import Encoder, Decoder, Seq2Seq
from data_preprocessing import train_loader, val_loader, VOCAB_SIZE, PAD_IDX, sp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DRIVE_PATH = "/content/drive/MyDrive/nmt-seq2seq-attention/"
os.makedirs(DRIVE_PATH, exist_ok=True)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

embedding_size = 128  
hidden_size = 128     
num_layers = 2             
attention_dim = 128       
learning_rate = 4e-4      
num_epochs = 40           
early_stop_patience = 12   
clip = 1.0                

encoder = Encoder(VOCAB_SIZE, embedding_size, hidden_size, num_layers, PAD_IDX, dropout=0.5).to(device)
decoder = Decoder(embedding_size, VOCAB_SIZE, hidden_size, num_layers, attention_dim, PAD_IDX, dropout=0.5).to(device)
model = Seq2Seq(encoder, decoder).to(device) 

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.15)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2) 
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

def run_epoch(model, loader, train=True, epoch=0):
    model.train() if train else model.eval()
    total_loss = 0
    
    if train:
        tf_ratio = max(0.1, 0.5 - (epoch - 1) * 0.03)
    else:
        tf_ratio = 0  

    for batch_idx, (src, trg, en_text, ar_text) in enumerate(loader):
        src, trg = src.to(device), trg.to(device)
        
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == "cuda")):
                output = model(src, trg, teacher_forcing_ratio=tf_ratio)
                output_dim = output.shape[-1]
                loss = criterion(
                    output[:, 1:].contiguous().view(-1, output_dim), 
                    trg[:, 1:].contiguous().view(-1)
                )

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % 100 == 0:
                print(f"   Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}", end='\r')

        total_loss += loss.detach().item()
        
        if train and batch_idx % 400 == 0:
             torch.cuda.empty_cache()
             
    if not train and epoch % 1 == 0:
        print(f"\n--- 🌟 Live Sample Test (Epoch {epoch}) ---")
        print(f"English: {en_text[0]}")
        print(f"Target Arabic: {ar_text[0]}")
        pred_tokens = output[0].argmax(dim=-1).tolist()
        pred_clean = [t for t in pred_tokens if t not in [0, 1, 2]]
        predicted_text = sp.decode(pred_clean)
        print(f"Model Translation: {predicted_text}")
        print("-" * 40)
    
    return total_loss / len(loader), tf_ratio

best_val = float("inf")
counter = 0

print(f"🚀 Last Run Starting on {device}...")

for epoch in range(1, num_epochs + 1):
    torch.cuda.empty_cache()
    gc.collect()
    
    start_time = time.time()
    
    train_loss, current_tf = run_epoch(model, train_loader, train=True, epoch=epoch)
    val_loss, _ = run_epoch(model, val_loader, train=False, epoch=epoch)

    scheduler.step(val_loss)
    
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "nmt_best_model.pth")
        try:
            shutil.copy("nmt_best_model.pth", os.path.join(DRIVE_PATH, "nmt_best_model.pth"))
            status = "⭐ Saved Best"
        except:
            status = "⭐ Saved Local"
        counter = 0
    else:
        counter += 1
        status = f"Patience: {counter}/{early_stop_patience}"

    try:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
        shutil.copy(f"checkpoint_epoch_{epoch}.pth", os.path.join(DRIVE_PATH, "checkpoint_last_run.pth"))
    except:
        pass

    duration = int(time.time() - start_time)
    print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {duration}s | {status}")

    if counter >= early_stop_patience:
        print(f"\n🛑 Early Stopping! Best Val Loss: {best_val:.4f}")
        break  
