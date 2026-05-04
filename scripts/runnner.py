import argparse
import os
from glob import glob

import numpy as np

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

# NEW: use soundfile for robust audio reading on Windows
import soundfile as sf

# --------------------------- Config / Hyperparams ---------------------------
DEFAULT_SAMPLE_RATE = 32000
N_MELS = 64
WIN_SEC = 0.025
CLIP_SEC = 5.0
HOP_SEC = 0.1
N_FFT = 1024
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-5

class PCEN(nn.Module):
    def __init__(self, n_mels, alpha=0.98, delta=2.0, r=0.5, s=0.025, eps=1e-6):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.log(torch.tensor([alpha])))
        self.log_delta = nn.Parameter(torch.log(torch.tensor([delta])))
        self.log_r = nn.Parameter(torch.log(torch.tensor([r])))
        self.s = s
        self.eps = eps

    def forward(self, x):
        alpha = torch.exp(self.log_alpha)
        delta = torch.exp(self.log_delta)
        r = torch.exp(self.log_r)
        smoothed = x.clone()
        for t in range(1, x.size(-1)):
            smoothed[..., t] = (1 - self.s) * smoothed[..., t-1] + self.s * x[..., t]
        pcen = (x / (smoothed + self.eps).pow(alpha) + delta).pow(r) - delta.pow(r)
        return pcen

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)

class CRNN(nn.Module):
    def __init__(self, n_mels: int, n_classes: int):
        super().__init__()
        self.pcen = PCEN(n_mels)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(input_size=(n_mels // 4) * 128, hidden_size=256, batch_first=True, bidirectional=True)
        self.attention = Attention(256 * 2)
        self.classifier = nn.Linear(256 * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.pcen(x)
        z = self.cnn(x)
        C, Mprime, Tprime = z.shape[1], z.shape[2], z.shape[3]
        z = z.permute(0, 3, 1, 2).contiguous().view(B, Tprime, C * Mprime)
        r, _ = self.rnn(z)
        # Global context via attention
        context = self.attention(r)
        logits = self.classifier(context)
        # Also return r (frame-level features) for time-localized detection
        # The frame-level classifier uses the same weights but applied to each frame
        frame_logits = self.classifier(r)
        return frame_logits, logits


def logits_to_events_adaptive(
    logits: np.ndarray,
    labels: list,
    global_min_thresh: float = 0.05,
    on_ratio: float = 0.25,
    off_ratio: float = 0.5,
    hop_seconds: float = 0.01,
    min_dur: float = 0.15,
    merge_gap: float = 0.1
):
    """
    Adaptive/hysteresis event detector.

    - Computes probs = sigmoid(logits)
    - For each class c: picks class_threshold = max(global_min_thresh, on_ratio * max_prob_c)
      - On-threshold = class_threshold
      - Off-threshold = class_threshold * off_ratio  (hysteresis)
    - Walks frames with hysteresis to make event segments
    - Enforces min_dur and merges events separated by <= merge_gap
    - Returns list of events with fields: class_idx, start, end, score (max prob in the segment), label
    """

    probs = 1.0 / (1.0 + np.exp(-logits))
    Tframes, C = probs.shape
    events = []

    max_per_class = probs.max(axis=0)
    # avoid zero thresholds: use global_min_thresh baseline
    class_on_thresh = np.maximum(global_min_thresh, on_ratio * max_per_class)
    class_off_thresh = class_on_thresh * off_ratio

    for c in range(C):
        p = probs[:, c]
        on_th = class_on_thresh[c]
        off_th = class_off_thresh[c]
        if max_per_class[c] < global_min_thresh:
            # skip classes with no signal at all (optional)
            continue

        i = 0
        active = False
        seg_start = None
        seg_max_score = 0.0
        while i < Tframes:
            val = p[i]
            if not active:
                # waiting for on trigger
                if val >= on_th:
                    active = True
                    seg_start = i
                    seg_max_score = val
                i += 1
            else:
                # currently active; we keep until we go below off_th
                seg_max_score = max(seg_max_score, val)
                if val < off_th:
                    seg_end = i  # exclusive
                    # convert to seconds
                    start_s = seg_start * hop_seconds
                    end_s = seg_end * hop_seconds
                    dur = end_s - start_s
                    if dur >= min_dur:
                        events.append({
                            'class_idx': c,
                            'start': round(start_s, 6),
                            'end': round(end_s, 6),
                            'score': float(seg_max_score),
                            'label': labels[c]
                        })
                    active = False
                    seg_start = None
                    seg_max_score = 0.0
                i += 1

        # if still active at end, close it
        if active and seg_start is not None:
            seg_end = Tframes
            start_s = seg_start * hop_seconds
            end_s = seg_end * hop_seconds
            dur = end_s - start_s
            if dur >= min_dur:
                events.append({
                    'class_idx': c,
                    'start': round(start_s, 6),
                    'end': round(end_s, 6),
                    'score': float(seg_max_score),
                    'label': labels[c]
                })

    # merge events of same class that are very close
    merged = []
    events = sorted(events, key=lambda e: (e['class_idx'], e['start']))
    for e in events:
        if not merged:
            merged.append(e)
            continue
        last = merged[-1]
        if e['class_idx'] == last['class_idx'] and e['start'] - last['end'] <= merge_gap:
            # merge
            new_end = max(last['end'], e['end'])
            new_score = max(last['score'], e['score'])
            merged[-1] = {
                'class_idx': last['class_idx'],
                'start': last['start'],
                'end': new_end,
                'score': new_score,
                'label': labels[last['class_idx']]
            }
        else:
            merged.append(e)

    return merged


def run_inference(audio_path: str, checkpoint: str, device: str = 'cuda'):
    # Load checkpoint
    ck = torch.load(checkpoint, map_location='cpu')
    labels = ck['labels']
    state = ck['model_state']
    model = CRNN(n_mels=N_MELS, n_classes=len(labels))
    model.load_state_dict(state)
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
    model.to(dev)
    print("Loaded model")

    # Load audio
    audio, sr = sf.read(audio_path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    wav = torch.from_numpy(audio).unsqueeze(0)  # (1, N)
    if sr != DEFAULT_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, DEFAULT_SAMPLE_RATE)
    # # pad/trim
    # clip_len = int(CLIP_SEC * DEFAULT_SAMPLE_RATE)
    # if wav.shape[1] < clip_len:
    #     wav = torch.nn.functional.pad(wav, (0, clip_len - wav.shape[1]))
    # else:
    #     wav = wav[:, :clip_len]

    print("Loaded Audio")

    # Mel transform
    hop_length = int(HOP_SEC * DEFAULT_SAMPLE_RATE)
    mel_tf = T.MelSpectrogram(sample_rate=DEFAULT_SAMPLE_RATE, n_fft=N_FFT,
                              hop_length=hop_length, win_length=int(WIN_SEC*DEFAULT_SAMPLE_RATE),
                              n_mels=N_MELS)
    amp_to_db = T.AmplitudeToDB()

    mel = mel_tf(wav)
    mel = amp_to_db(mel)

    # Use the same normalization as training
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = mel.unsqueeze(0).to(dev)  # (1,1,n_mels,T)

    # Predict
    print("Predicting")
    with torch.no_grad():
        frame_logits, global_logits = model(mel)  # returns (T, C) and (C,)
    
    # We use frame_logits for time-localized event detection
    logits = frame_logits.squeeze(0).cpu().numpy()  # (T, C)


    print("Start mapping logits")
    time_downsample = mel.shape[-1] // logits.shape[0]
    hop_seconds = (hop_length / DEFAULT_SAMPLE_RATE) * time_downsample

    # Map logits to events
    events = logits_to_events_adaptive(
        logits,
        labels=ck['labels'],
        global_min_thresh=0.05,   # baseline minimum
        on_ratio=0.25,
off_ratio=0.5,
            # off threshold = on_threshold * 0.5 (hysteresis)
        hop_seconds=hop_seconds,
        min_dur=0.01,
        merge_gap=0.01
    )

    return events


def run_multi_inference(audio_dir: str, checkpoint: str, device: str = 'cuda'):
    # Find all audio files in the directory
    audio_paths = []
    for ext in ('wav', 'flac', 'mp3', 'ogg'):
        audio_paths.extend(glob(os.path.join(audio_dir, f'**/*.{ext}'), recursive=True))
    if not audio_paths:
        print(f"No audio files found in {audio_dir}")
        return

    all_events = {}
    sound_counts = {}
    unique_sounds = set()

    print("Processing devices:")
    for idx, audio_path in enumerate(sorted(audio_paths)):
        print(f"\nDevice {idx + 1} ({os.path.basename(audio_path)}):")
        events = run_inference(audio_path, checkpoint, device)
        status = []
        device_sounds = set()
        for e in events:
            label = e['label']
            start = e['start']
            end = e['end']
            score = e['score']
            status.append(f"{label} from {start:.2f}s to {end:.2f}s (score: {score:.4f})")
            device_sounds.add(label)
            unique_sounds.add(label)
            sound_counts[label] = sound_counts.get(label, 0) + 1
        if status:
            for s in status:
                print(f"  - {s}")
        else:
            print("  No significant sounds detected.")
        all_events[audio_path] = events

    # Generate one-sentence summary
    if not unique_sounds:
        summary = "No sounds were detected across all devices in the forest grid, indicating a quiet environment with nothing notable occurring."
    else:
        top_sounds = sorted(sound_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join([f"{k} ({v} occurrences)" for k, v in top_sounds])
        illegal_sounds = {'chainsaw', 'gun_shot', 'explosion'}  # Example illegal sounds; adjust based on dataset
        has_illegal = any(s in illegal_sounds for s in unique_sounds)
        if has_illegal:
            summary = f"Across {len(audio_paths)} devices in the forest grid, detected sounds including {top_str}, with potential illegal activities noted."
        else:
            summary = f"Across {len(audio_paths)} devices in the forest grid, detected normal sounds such as {top_str}, with no illegal activities apparent."

    print("\nOverall Summary:")
    print(summary)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', type=str, default='experiments\\run7', help='output directory for manifests and models')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--run_infer', type=str, default=None, help='if provided, run inference on this file after training using best_checkpoint.pth')
    p.add_argument('--run_multi_infer', type=str, default=None, help='if provided, run multi-device inference on audio files in this directory')
    return p.parse_args()

def main():
    print("Starting")
    args = parse_args()

    if args.run_multi_infer:
        run_multi_inference(
            args.run_multi_infer,
            os.path.join(args.out_dir, 'best_checkpoint.pth'),
            device=args.device
        )

    elif args.run_infer:
        events = run_inference(
            args.run_infer,
            os.path.join(args.out_dir, 'best_checkpoint.pth'),
            device=args.device
        )
        # print(events)
        events = [e for e in events if e['score'] > 0.5][:10]  # filter low confidence
        for e in events:
            print(f"{e['label']} {e['start']:.2f}-{e['end']:.2f}s score={e['score']:.4f}")
        
        print("Finished")


if __name__ == "__main__":
    main()
