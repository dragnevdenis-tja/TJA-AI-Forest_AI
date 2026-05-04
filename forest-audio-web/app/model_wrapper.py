import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
import io

DEFAULT_SAMPLE_RATE = 32000
N_MELS = 64
WIN_SEC = 0.025
CLIP_SEC = 5.0
HOP_SEC = 0.1
N_FFT = 1024

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
        self.pcen_bn = nn.BatchNorm2d(1)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=128*(n_mels//4), hidden_size=256, batch_first=True, bidirectional=True)
        self.attention = Attention(256*2)
        self.classifier = nn.Linear(256*2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.pcen(x)
        x = self.pcen_bn(x)
        z = self.cnn(x)
        C, M, T = z.shape[1], z.shape[2], z.shape[3]
        z = z.permute(0, 3, 1, 2).contiguous().view(B, T, C*M)
        r, _ = self.rnn(z)
        # Use attention instead of max pooling for better sound separation
        context = self.attention(r)
        logits = self.classifier(context)
        # Repeat logits over time if frame-level targets are needed, 
        # but here we return global logits for simplicity or use the 'r' for frame-level
        return logits, r # Return both for flexibility

class ForestAudioAI:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.labels = ck['labels']
        self.model = CRNN(n_mels=N_MELS, n_classes=len(self.labels))
        self.model.load_state_dict(ck['model_state'])
        self.model.eval()
        self.model.to(self.device)
        
        self.mel_tf = T.MelSpectrogram(sample_rate=DEFAULT_SAMPLE_RATE, n_fft=N_FFT,
                                      hop_length=int(HOP_SEC * DEFAULT_SAMPLE_RATE), 
                                      win_length=int(WIN_SEC*DEFAULT_SAMPLE_RATE),
                                      n_mels=N_MELS)

    def predict(self, audio_bytes):
        # Load audio from bytes
        audio, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
        return self.predict_pcm(audio, sr)

    def predict_pcm(self, audio, sr):
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        wav = torch.from_numpy(audio).unsqueeze(0)  # (1, N)
        
        if sr != DEFAULT_SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, DEFAULT_SAMPLE_RATE)
            
        mel = self.mel_tf(wav)
        # Ensure strictly positive for PCEN stability as in trainer.py
        mel = mel + 1e-9
        mel = mel.unsqueeze(0).to(self.device)  # (1,1,n_mels,T)
        
        with torch.no_grad():
            logits, _ = self.model(mel)
            
        # Global probabilities
        probs = torch.sigmoid(logits[0]).cpu().numpy()
        
        # Get top predictions
        top_indices = np.where(probs > 0.3)[0] # Increased threshold for more confident results
        results = []
        for idx in top_indices:
            results.append({
                "label": self.labels[idx],
                "confidence": float(probs[idx])
            })
            
        return sorted(results, key=lambda x: x['confidence'], reverse=True)

