#!/usr/bin/env python3
"""
adversarial_forest_trainer_youtube_chunked.py

Adversarial forest sound detection:
- Main CRNN predicts forest sound classes from audio chunks
- Validator predicts correctness of main CRNN predictions using YouTube metadata
- Both models train alternately, improving automatically
- Downloads random forest YouTube videos for training dynamically
- Long audio files are split into multiple chunks
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import soundfile as sf
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

import yt_dlp

from trainer import CRNN, collate_fn, N_MELS, BATCH_SIZE, LR, WEIGHT_DECAY, DEFAULT_SAMPLE_RATE

# ----------------- Utilities -----------------
def encode_metadata(video_info, labelset):
    """One-hot encode if label appears in title/tags (max 10 features)"""
    feats = []
    text = (video_info.get("title","") + " " + video_info.get("description","") + " " + " ".join(video_info.get("tags",[]))).lower()
    for l in labelset:
        feats.append(1.0 if l in text else 0.0)
    feats = torch.tensor(feats[:10], dtype=torch.float32)
    if len(feats) < 10:
        feats = F.pad(feats, (0, 10 - len(feats)))
    return feats.unsqueeze(0)

def download_youtube_audio(label:str, output_dir:str, num_results=1):
    """Download forest-related videos from YouTube and extract audio"""
    os.makedirs(output_dir, exist_ok=True)
    query = f"ytsearch{num_results}:{label} forest sound"
    ydl_opts = {
        "quiet": True,
        "format": "bestaudio/best",
        "ffmpeg_location": r"D:\ffmpeg-2026-01-26-git-fe0813d6e2-essentials_build\ffmpeg-2026-01-26-git-fe0813d6e2-essentials_build\bin",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=True)
    videos = []
    if "entries" in info:
        for e in info["entries"]:
            videos.append(e)
    return videos

def chunk_audio(file_path, clip_sec=5.0, sample_rate=DEFAULT_SAMPLE_RATE):
    """Load audio and split into multiple chunks of length clip_sec"""
    wav, sr = sf.read(file_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != sample_rate:
        wav = torch.from_numpy(wav).float().unsqueeze(0)
        wav = torchaudio.functional.resample(wav, sr, sample_rate).squeeze(0).numpy()
    total_len = len(wav)
    clip_len = int(sample_rate * clip_sec)
    chunks = []
    start = 0
    while start < total_len:
        end = min(start + clip_len, total_len)
        chunk = wav[start:end]
        # pad if last chunk is smaller than clip_len
        if len(chunk) < clip_len:
            chunk = F.pad(torch.tensor(chunk), (0, clip_len - len(chunk))).numpy()
        chunks.append(chunk)
        start += clip_len  # optionally could use overlap here
    return chunks

def audio_to_mel(wav_chunk, sample_rate=DEFAULT_SAMPLE_RATE):
    """Convert audio chunk to normalized Mel-spectrogram"""
    wav_tensor = torch.tensor(wav_chunk).unsqueeze(0).float()
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=N_MELS)(wav_tensor)
    mel = torchaudio.transforms.AmplitudeToDB()(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel

# ----------------- Validator -----------------
class ValidatorCRNN(nn.Module):
    """
    Validator CRNN predicts correctness probability for main model predictions
    """
    def __init__(self, n_mels, n_classes):
        super().__init__()
        self.crnn = CRNN(n_mels, n_classes)
        self.scorer = nn.Sequential(
            nn.Linear(n_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        logits = self.crnn(x)
        probs = torch.sigmoid(logits.max(1)[0])
        score = self.scorer(probs)
        return score

# ----------------- Main Training Loop -----------------
def adversarial_train_youtube(labels_file, out_dir, epochs=50,
                              batch_size=BATCH_SIZE, lr=LR, device='cuda',
                              lambda_adv=0.1, clip_sec=5.0):
    os.makedirs(out_dir, exist_ok=True)

    with open(labels_file, 'r') as f:
        labels = [l.strip() for l in f if l.strip()]
    n_classes = len(labels)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    main_crnn = CRNN(n_mels=N_MELS, n_classes=n_classes).to(device)
    opt_main = optim.Adam(main_crnn.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    validator_crnn = ValidatorCRNN(n_mels=N_MELS, n_classes=n_classes).to(device)
    opt_val = optim.Adam(validator_crnn.parameters(), lr=lr*0.5, weight_decay=WEIGHT_DECAY)

    main_ckpt_path = os.path.join(out_dir, "main_crnn_latest.pth")
    validator_ckpt_path = os.path.join(out_dir, "validator_crnn_latest.pth")
    try:
        if os.path.exists(main_ckpt_path):
            ckpt = torch.load(main_ckpt_path, map_location=device)
            main_crnn.load_state_dict(ckpt["model_state"])
            opt_main.load_state_dict(ckpt.get("optimizer_state", opt_main.state_dict()))
    except: pass
    try:
        if os.path.exists(validator_ckpt_path):
            ckpt = torch.load(validator_ckpt_path, map_location=device)
            validator_crnn.load_state_dict(ckpt["model_state"])
            opt_val.load_state_dict(ckpt.get("optimizer_state", opt_val.state_dict()))
    except: pass

    criterion_class = nn.BCEWithLogitsLoss()
    criterion_val = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        # --- Download videos ---
        videos = download_youtube_audio("forest", output_dir=os.path.join(out_dir, "yt_cache"), num_results=1)
        mels_list, labels_list, meta_list = [], [], []

        for v in videos:
            try:
                audio_file = Path(os.path.join(out_dir, "yt_cache", v['id']+'.wav'))
                if not audio_file.exists(): 
                    continue
                chunks = chunk_audio(audio_file, clip_sec=clip_sec)
                # Metadata & target labels
                target = torch.zeros(n_classes)
                text = (v.get("title","")+" "+v.get("description","")+" "+" ".join(v.get("tags",[]))).lower()
                for i,l in enumerate(labels):
                    if l in text: target[i] = 1
                meta_feat = encode_metadata(v, labels).to(device)

                for chunk in chunks:
                    mel = audio_to_mel(chunk).to(device)
                    mels_list.append(mel)
                    labels_list.append(target.to(device))
                    meta_list.append(meta_feat)
            except Exception as e:
                print("Skipping video:", e)

        if len(mels_list) == 0:
            print("No audio chunks loaded this epoch, skipping...")
            continue

        mels = torch.stack(mels_list)
        y_true = torch.stack(labels_list)
        meta_feats = torch.cat(meta_list)

        # --- Main CRNN step ---
        logits_main = main_crnn(mels)
        probs_main = torch.sigmoid(logits_main.max(1)[0])
        validator_scores = validator_crnn(mels)
        adv_loss = -validator_scores.mean()
        class_loss = criterion_class(logits_main.max(1)[0], y_true)
        loss_main = class_loss + lambda_adv * adv_loss

        opt_main.zero_grad()
        loss_main.backward()
        opt_main.step()

        # --- Validator step ---
        with torch.no_grad():
            logits_main_detach = main_crnn(mels)
            probs_main_detach = torch.sigmoid(logits_main_detach.max(1)[0])
        correct_labels = ((probs_main_detach > 0.5).float() * y_true).sum(dim=1)/(y_true.sum(dim=1)+1e-6)
        correct_labels = correct_labels.clamp(0,1).unsqueeze(1)
        validator_scores = validator_crnn(mels)
        loss_val = criterion_val(validator_scores, correct_labels)

        opt_val.zero_grad()
        loss_val.backward()
        opt_val.step()

        print(f"Epoch {epoch} | Main Loss: {loss_main.item():.4f} | Validator Loss: {loss_val.item():.4f}")

        # Save checkpoints
        torch.save(main_crnn.state_dict(), os.path.join(out_dir, "main_crnn_latest.pth"))
        torch.save(validator_crnn.state_dict(), os.path.join(out_dir, "validator_crnn_latest.pth"))

# ----------------- CLI -----------------
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', required=True)
    parser.add_argument('--out_dir', default='adversarial_models')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lambda_adv', type=float, default=0.1)
    parser.add_argument('--clip_sec', type=float, default=5.0)
    args = parser.parse_args()

    adversarial_train_youtube(args.labels, args.out_dir,
                              epochs=args.epochs,
                              batch_size=args.batch,
                              lr=args.lr,
                              device=args.device,
                              lambda_adv=args.lambda_adv,
                              clip_sec=args.clip_sec)
