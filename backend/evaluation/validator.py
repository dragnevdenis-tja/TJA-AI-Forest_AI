#!/usr/bin/env python3
"""
adversarial_forest_trainer.py

Adversarial forest sound detection:
- Main CRNN predicts forest sound classes
- Validator CRNN predicts correctness of main CRNN predictions
- Both models train alternately, improving automatically
"""

import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import yt_dlp
import ffmpeg

from backend.training.trainer import UnifiedSoundDataset, CRNN, collate_fn, N_MELS, BATCH_SIZE, LR, WEIGHT_DECAY, DEFAULT_SAMPLE_RATE

def encode_metadata(video_info, labelset):
    """
    Very simple: one-hot encode if label appears in title/tags
    """
    feats = []
    text = (video_info.get("title","") + " " + video_info.get("description","") + " " + " ".join(video_info.get("tags",[]))).lower()
    for l in labelset:
        feats.append(1.0 if l in text else 0.0)
    feats = torch.tensor(feats[:10], dtype=torch.float32)  # truncate/pad to 10
    if len(feats)<10:
        feats = nn.functional.pad(feats,(0,10-len(feats)))
    return feats.unsqueeze(0)


def download_youtube_audio(label:str, output_dir:str, num_results=1):
    os.makedirs(output_dir, exist_ok=True)
    query = f"ytsearch{num_results}:{label} forest sound"
    ydl_opts = {"quiet":True,"format":"bestaudio/best",
                "outtmpl":os.path.join(output_dir,"% (id)s.%(ext)s"),
                "postprocessors":[{"key":"FFmpegExtractAudio","preferredcodec":"wav"}]}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=True)
    videos=[]
    if "entries" in info:
        for e in info["entries"]:
            videos.append(e)
    return videos

# ----------------- Adversarial Trainer -----------------

class ValidatorCRNN(nn.Module):
    """
    Validator CRNN predicts correctness probability for main model predictions
    """
    def __init__(self, n_mels, n_classes):
        super().__init__()
        # Reuse main CRNN architecture
        self.crnn = CRNN(n_mels, n_classes)
        # final score: probability prediction for correctness
        self.scorer = nn.Sequential(
            nn.Linear(n_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.crnn(x)
        # Use max over time dimension as aggregated prediction
        probs = torch.sigmoid(logits.max(1)[0])
        score = self.scorer(probs)
        return score

# ----------------- Training Loop -----------------
def adversarial_train(manifest, labels_file, out_dir, epochs=999999999,
                      batch_size=BATCH_SIZE, lr=LR, device='cuda', lambda_adv=0.1):
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset
    dataset = UnifiedSoundDataset(manifest, labels_file)
    n_classes = len(dataset.labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    main_ckpt_path = os.path.join(out_dir,"main_crnn_latest.pth")
    validator_ckpt_path = os.path.join(out_dir,"validator_crnn_latest.pth")
    main_crnn = CRNN(n_mels=N_MELS, n_classes=n_classes).to(device)
    opt_main = optim.Adam(main_crnn.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    validator_crnn = ValidatorCRNN(n_mels=N_MELS, n_classes=n_classes).to(device)
    opt_val = optim.Adam(validator_crnn.parameters(), lr=lr*0.5, weight_decay=WEIGHT_DECAY)  # validator slower
    try:
        if os.path.exists(main_ckpt_path):
            ckpt = torch.load(main_ckpt_path, map_location=device)
            main_crnn.load_state_dict(ckpt["model_state"])
            opt_main.load_state_dict(ckpt.get("optimizer_state", opt_main.state_dict()))
    except:
        pass

    try:
        if os.path.exists(validator_ckpt_path):
            ckpt = torch.load(validator_ckpt_path, map_location=device)
            validator_crnn.load_state_dict(ckpt["model_state"])
            opt_val.load_state_dict(ckpt.get("optimizer_state", opt_val.state_dict()))
    except:
        pass

    criterion_class = nn.BCEWithLogitsLoss()
    criterion_val = nn.BCELoss()

    best_metric = -1

    for epoch in range(1, epochs+1):
        main_crnn.train()
        validator_crnn.train()
        total_loss_main = 0.0
        total_loss_val = 0.0

        for mel, y, _ in tqdm(loader, desc=f"Epoch {epoch}"):
            mel = mel.to(device)
            y = y.to(device)

            # ---------------- Main CRNN Step ----------------
            logits_main = main_crnn(mel)
            probs_main = torch.sigmoid(logits_main.max(1)[0])

            # Validator scores the main prediction
            validator_scores = validator_crnn(mel)
            # Generator (main CRNN) wants high validator score for correct predictions
            adv_loss = -validator_scores.mean()  # maximize validator confidence

            class_loss = criterion_class(logits_main.max(1)[0], y)
            loss_main = class_loss + lambda_adv * adv_loss

            opt_main.zero_grad()
            loss_main.backward()
            opt_main.step()
            total_loss_main += loss_main.item()*mel.size(0)

            # ---------------- Validator Step ----------------
            with torch.no_grad():
                logits_main_detach = main_crnn(mel)
                probs_main_detach = torch.sigmoid(logits_main_detach.max(1)[0])
            # label validator as 1 if main prediction correct, 0 otherwise
            correct_labels = ((probs_main_detach>0.5).float() * y).sum(dim=1) / (y.sum(dim=1)+1e-6)
            correct_labels = correct_labels.clamp(0,1).unsqueeze(1)

            validator_scores = validator_crnn(mel)
            loss_val = criterion_val(validator_scores, correct_labels)

            opt_val.zero_grad()
            loss_val.backward()
            opt_val.step()
            total_loss_val += loss_val.item()*mel.size(0)

        print(f"Epoch {epoch} | Main Loss: {total_loss_main/len(dataset):.4f} | Validator Loss: {total_loss_val/len(dataset):.4f}")

        # ---------------- Save Checkpoints ----------------
        torch.save({
            "model_state": main_crnn.state_dict(),
            "optimizer_state": opt_main.state_dict(),
            "labels": dataset.labels
        }, os.path.join(out_dir, "main_crnn_latest.pth"))
        torch.save({
            "model_state": validator_crnn.state_dict(),
            "optimizer_state": opt_val.state_dict(),
            "labels": dataset.labels
        }, os.path.join(out_dir, "validator_crnn_latest.pth"))

# ----------------- CLI -----------------
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--out_dir', default='adversarial_models')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lambda_adv', type=float, default=0.1)
    args = parser.parse_args()

    adversarial_train(args.manifest, args.labels, args.out_dir,
                      epochs=args.epochs, batch_size=args.batch,
                      lr=args.lr, device=args.device, lambda_adv=args.lambda_adv)
