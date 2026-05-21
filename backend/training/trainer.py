#!/usr/bin/env python3
"""
master_sound_trainer.py

Production-ready forest sound trainer using:
- PyTorch 3.11+
- Robust audio IO via soundfile
- UnifiedSoundDataset with augmentation & frame-level targets
- CRNN model
- Automatic manifest building and YouTube dataset integration
"""
import argparse
import json
import os
import random
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from tqdm import tqdm

from backend.utils.seed import set_seed

# Optional sklearn metrics
try:
    from sklearn.metrics import average_precision_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# --------------------------- Hyperparams / Config ---------------------------
DEFAULT_SAMPLE_RATE = 32000
N_MELS = 64
WIN_SEC = 0.025
HOP_SEC = 0.1
CLIP_SEC = 5.0
N_FFT = 1024
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-5

# --------------------------- YouTube Download Functions ---------------------
import yt_dlp
import static_ffmpeg

def download_youtube_audio_for_label(label: str, output_dir: str, num_results: int = 5, skip_ids: List[str] = []) -> List[str]:
    # Ensure ffmpeg/ffprobe are available in the path
    static_ffmpeg.add_paths()
    
    os.makedirs(output_dir, exist_ok=True)
    downloaded = []
    query = f"ytsearch{num_results}:{label} sound"
    
    # Custom match filter to skip already downloaded IDs
    def match_filter(info_dict, *, incomplete):
        if info_dict.get('id') in skip_ids:
            return 'Already downloaded this video in a previous session'
        return None

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "max_filesize": 100 * 1024 * 1024, # Limit to 100 MiB
        "match_filter": match_filter,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=True)
        if "entries" in info:
            for e in info["entries"]:
                if not e:
                    continue
                wav_path = os.path.join(output_dir, f"{e['id']}.wav")
                if os.path.exists(wav_path):
                    downloaded.append(wav_path)
    return downloaded

def build_youtube_dataset(data_root: str, labels_file: str, num_per_label: int = 1, for_eval: bool = False) -> None:
    with open(labels_file, "r", encoding="utf-8") as f:
        labels = [l.strip() for l in f if l.strip()]
    subfolder = "youtube_eval" if for_eval else "youtube_train"
    youtube_root = os.path.join(data_root, subfolder)
    os.makedirs(youtube_root, exist_ok=True)
    for label in labels:
        try:
            label_dir = os.path.join(youtube_root, label)
            print(f"Downloading samples for label: {label}")
            download_youtube_audio_for_label(label, label_dir, num_per_label)
        except Exception as e:
            print(e)
    print(f"Dataset built at: {youtube_root}")

# --------------------------- Manifest builder ------------------------------

def parse_esc50(esc_root: str) -> List[dict]:
    meta = os.path.join(esc_root, 'meta', 'esc50.csv')
    out = []
    if os.path.exists(meta):
        import csv
        with open(meta, newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                fname = os.path.join(esc_root, 'audio', row['filename'])
                if os.path.exists(fname):
                    out.append({'path': fname, 'labels': [row['category'].strip().lower()], 'events': []})
    return out

def parse_generic_folder(root: str) -> List[dict]:
    # root/<label>/*.wav or .flac
    out = []
    if not os.path.exists(root):
        return out
    for labdir in sorted(os.listdir(root)):
        p = os.path.join(root, labdir)
        if not os.path.isdir(p):
            continue
        # skip meta folders
        if labdir.lower() in ('meta', 'annotations'):
            continue
        for ext in ('wav', 'flac', 'mp3', 'ogg'):
            for wav in glob(os.path.join(p, f'**/*.{ext}'), recursive=True):
                out.append({'path': wav, 'labels': [labdir.strip().lower()], 'events': []})
    return out

def parse_dcase_synth(dcase_root: str) -> List[dict]:
    """
    Robust parser for DCASE synthetic CSV annotations.

    Handles:
    - normal CSV with headers filename,onset,offset,event_label
    - TSV where each row might be read as a single field like:
      "filename\tonset\toffset\tevent_label": "184.wav\t5.51\t5.83\tAlarm_bell_ringing"
      (this was your observed problem)
    - various header name variants: file, audio, label, event_label, category
    - groups rows by filename and returns items like:
        {'path': fullpath, 'labels': [...], 'events': [{'label':..., 'onset':..., 'offset':...}, ...]}
    """
    out = []
    if not os.path.exists(dcase_root):
        return out

    csv_paths = glob(os.path.join(dcase_root, '**', '*.csv'), recursive=True)
    parsed_rows = []

    for csv_path in csv_paths:
        # read entire file first (helps detect if it's tab-separated mashed into single column)
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as fh:
            text = fh.read()

        # detect delimiter heuristically: prefer tab if many tabs present, else comma
        n_tabs = text.count('\t')
        n_commas = text.count(',')
        delim = '\t' if n_tabs > n_commas else ','

        # now parse lines
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue

        header = lines[0]
        header_fields = header.split(delim)
        # normalize header fields (lowercase, strip)
        header_fields = [h.strip().lower() for h in header_fields]

        # expected columns candidates
        filename_keys = {'filename'}
        label_keys = {'event_label'}
        onset_keys = {'onset'}
        offset_keys = {'offset'}

        # compute indices for the expected columns where possible
        def find_index(candidates):
            for cand in candidates:
                if cand in header_fields:
                    return header_fields.index(cand)
            return None

        idx_fname = find_index(filename_keys)
        idx_label = find_index(label_keys)
        idx_onset = find_index(onset_keys)
        idx_offset = find_index(offset_keys)

        # If header was not split correctly (common when csv module produced a single-field dict) try fallback:
        # If header contains '\t' but delim was ',', try splitting header by '\t' anyway
        if len(header_fields) == 1 and '\t' in header:
            header_fields = header.split('\t')
            header_fields = [h.strip().lower() for h in header_fields]
            idx_fname = find_index(filename_keys)
            idx_label = find_index(label_keys)
            idx_onset = find_index(onset_keys)
            idx_offset = find_index(offset_keys)
            delim = '\t'

        # iterate over data lines (skip header)
        for ln in lines[1:]:
            parts = ln.split(delim)
            # print(parts)
            # if parts length equals 1 but it contains tabs (another odd case), try splitting by tab
            if len(parts) == 1 and '\t' in parts[0]:
                parts = parts[0].split('\t')

            # guard against short rows
            if len(parts) < 1:
                continue

            # extract fields using indices if available, otherwise try heuristics
            fname = None
            label = None
            onset = None
            offset = None

            if idx_fname is not None and idx_fname < len(parts):
                fname = parts[idx_fname].strip()
            if idx_label is not None and idx_label < len(parts):
                label = parts[idx_label].strip()
            if idx_onset is not None and idx_onset < len(parts):
                try:
                    onset = float(parts[idx_onset].strip())
                except Exception:
                    onset = None
            if idx_offset is not None and idx_offset < len(parts):
                try:
                    offset = float(parts[idx_offset].strip())
                except Exception:
                    offset = None

            # Heuristic fallbacks: if header columns not found, try common ordering: filename,onset,offset,label
            if fname is None or fname == '':
                if len(parts) >= 1:
                    fname = parts[0].strip()
                if len(parts) >= 2 and onset is None:
                    try:
                        onset = float(parts[1].strip())
                    except Exception:
                        pass
                if len(parts) >= 3 and offset is None:
                    try:
                        offset = float(parts[2].strip())
                    except Exception:
                        pass
                if len(parts) >= 4 and label is None:
                    label = parts[3].strip()

            # final sanity: skip rows without a filename
            if not fname:
                continue

            # normalize label
            if label:
                label = label.strip().lower()
            # default numeric onsets/offsets to 0.0 if missing
            if onset is None:
                onset = 0.0
            if offset is None:
                offset = 0.0

            # attempt to resolve full path: path may be relative to dcase_root or absolute
            possible_path = os.path.join(dcase_root, fname)
            if os.path.exists(possible_path):
                path = possible_path
            elif os.path.exists(fname):
                path = fname
            else:
                # try looking in subfolders (sometimes filenames are just '184.wav' and live in a 'audio' folder)
                found = None
                for ext_search in ('**/' + fname, '**/' + os.path.basename(fname)):
                    matches = glob(os.path.join(dcase_root, ext_search), recursive=True)
                    if matches:
                        found = matches[0]
                        break
                if found:
                    path = found
                else:
                    # file not found; skip this row but continue parsing others
                    # (we don't raise, because datasets sometimes reference files not included)
                    # print a small debug message for visibility
                    # (you can comment this out if noisy)
                    # print(f"Warning: referenced file not found for CSV row: {fname} (searched under {dcase_root})")
                    continue

            parsed_rows.append({'path': path, 'label': label, 'onset': onset, 'offset': offset})
            # print(parsed_rows[-1])

    # group parsed rows by path into manifest entries
    if not parsed_rows:
        # fallback: parse by folder structure
        return parse_generic_folder(dcase_root)

    grouped = {}
    for r in parsed_rows:
        p = r['path']
        if p not in grouped:
            grouped[p] = {'path': p, 'labels': [], 'events': []}
        if r['label']:
            grouped[p]['labels'].append(r['label'])
            grouped[p]['events'].append({'label': r['label'], 'onset': r['onset'], 'offset': r['offset']})

    # deduplicate labels/events per file
    out = []
    for p, info in grouped.items():
        info['labels'] = sorted(list(set([l for l in info['labels'] if l])))
        # optionally, you can sort events by onset
        info['events'] = sorted(info['events'], key=lambda x: x.get('onset', 0.0))
        out.append(info)

    return out

def parse_fsc22(fsc_root: str) -> List[dict]:
    meta_paths = glob(os.path.join(fsc_root, '**', '*Metadata*.csv'), recursive=True)
    out = []
    if meta_paths:
        import csv
        meta = meta_paths[0]  # assume first one
        with open(meta, newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                fname_key = 'Dataset File Name' if 'Dataset File Name' in row else 'dataset_file_name'
                class_key = 'Class Name' if 'Class Name' in row else 'class_name'
                fname = os.path.join(fsc_root, 'Audio Wise V1.0', row[fname_key])  # adjust based on user path
                if os.path.exists(fname):
                    out.append({'path': fname, 'labels': [row[class_key].strip().lower()], 'events': []})
    else:
        # fallback to generic if no meta
        out = parse_generic_folder(os.path.join(fsc_root, 'Audio Wise V1.0'))
    return out

def build_manifest(data_root: str, out_manifest: str, out_labels: str) -> None:
    # datasets we try to auto-detect
    candidates = {
        'ESC50': os.path.join(data_root, 'ESC50'),
        'ARCA23K': os.path.join(data_root, 'ARCA23K'),
        'DBR': os.path.join(data_root, 'DBR'),
        'DCASE2019_synth': os.path.join(data_root, 'DCASE2019_synth'),
        'FSC22': os.path.join(data_root, 'FSC22'),
        'WildFire': os.path.join(data_root, 'Forest Wild Fire Sound Dataset'),
        'Ambience': os.path.join(data_root, 'ambience-audio')  # assume downloaded locally
    }
    items = []
    # ESC50
    if os.path.exists(candidates['ESC50']):
        items += parse_esc50(candidates['ESC50'])
    # ARCA23K & DBR as generic folders
    if os.path.exists(candidates['ARCA23K']):
        items += parse_generic_folder(candidates['ARCA23K'])
    if os.path.exists(candidates['DBR']):
        items += parse_generic_folder(candidates['DBR'])
    # DCASE synth
    if os.path.exists(candidates['DCASE2019_synth']):
        items += parse_dcase_synth(candidates['DCASE2019_synth'])
    # FSC22
    if os.path.exists(candidates['FSC22']):
        items += parse_fsc22(candidates['FSC22'])
    # WildFire - assign fixed label 'wildfire' to all files
    if os.path.exists(candidates['WildFire']):
        wildfire_root = candidates['WildFire']
        for ext in ('wav', 'flac', 'mp3', 'ogg'):
            for p in glob(os.path.join(wildfire_root, f'**/*.{ext}'), recursive=True):
                items.append({'path': p, 'labels': ['wildfire'], 'events': []})
    # Ambience (assume user downloads from HF to this folder with folder structure)
    if os.path.exists(candidates['Ambience']):
        items += parse_generic_folder(candidates['Ambience'])
    # Optional: load HF dataset if not local
    try:
        from datasets import load_dataset
        ds = load_dataset("igorriti/ambience-audio")
        # Assume ds['train'] has 'audio' with 'path', but if not, skip or download
        ambience_dir = candidates['Ambience']
        os.makedirs(ambience_dir, exist_ok=True)
        for idx, example in enumerate(ds['train']):
            if 'audio' in example:
                audio_data = example['audio']['array']
                sr = example['audio']['sampling_rate']
                label = example.get('title', 'ambience').strip().lower()  # use title as label or 'ambience'
                fname = os.path.join(ambience_dir, f"{label}_{idx}.wav")
                sf.write(fname, audio_data, sr)
                items.append({'path': fname, 'labels': [label], 'events': []})
    except ImportError:
        print("datasets library not found; skipping ambience-audio download.")
    except Exception as e:
        print(f"Error loading ambience-audio: {e}")

    # YouTube datasets
    if os.path.exists(os.path.join(data_root, 'youtube_train')):
        items += parse_generic_folder(os.path.join(data_root, 'youtube_train'))
    if os.path.exists(os.path.join(data_root, 'youtube_eval')):
        items += parse_generic_folder(os.path.join(data_root, 'youtube_eval'))
    if os.path.exists(os.path.join(data_root, 'Harvested')):
        items += parse_generic_folder(os.path.join(data_root, 'Harvested'))

    # As a last effort, scan the entire data_root for audio files and try to assign labels by parent folder
    if not items:
        print('No specific dataset scaffolding found; scanning data_root for audio files...')
        for ext in ('wav', 'flac', 'mp3', 'ogg'):
            for p in glob(os.path.join(data_root, f'**/*.{ext}'), recursive=True):
                # label = parent folder name
                label = Path(p).parent.name.lower()
                items.append({'path': p, 'labels': [label], 'events': []})

    # normalize labels and deduplicate
    for it in items:
        labs = []
        for l in it.get('labels', []) or []:
            if not l:
                continue
            labs.append(l.strip().lower())
        it['labels'] = sorted(list(set(labs)))
        if 'events' not in it:
            it['events'] = []
        else:
            # ensure event labels normalized
            for e in it['events']:
                e['label'] = e.get('label', '').strip().lower()

    # global labelset
    labelset = sorted({lab for it in items for lab in it['labels']})
    if not labelset:
        print('Warning: no labels detected. Exiting.')
        sys.exit(1)

    # write labels and manifest
    with open(out_labels, 'w') as f:
        for l in labelset:
            f.write(l + '\n')

    with open(out_manifest, 'w') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')

    print(f'Built manifest {out_manifest} with {len(items)} items and {len(labelset)} labels')
import pandas as pd

# --------------------------- Self-Learning & Harvesting ---------------------

class SoundHarvester:
    """
    Handles downloading and managing internet audio data.
    Ensures data is stored with 'tabular' metadata.
    Tracks downloaded IDs to avoid duplicates.
    """
    def __init__(self, data_root: str, labels: List[str]):
        self.harvest_root = os.path.join(data_root, "Harvested")
        os.makedirs(self.harvest_root, exist_ok=True)
        self.labels = labels
        self.metadata_file = os.path.join(self.harvest_root, "harvested_metadata.csv")
        self.download_history_file = os.path.join(self.harvest_root, "downloaded_ids.txt")
        self._init_metadata()
        self.downloaded_ids = self._load_history()

    def _init_metadata(self):
        if not os.path.exists(self.metadata_file):
            df = pd.DataFrame(columns=["path", "label", "confidence", "source_url", "duration_sec", "timestamp", "video_id"])
            df.to_csv(self.metadata_file, index=False)

    def _load_history(self) -> List[str]:
        if os.path.exists(self.download_history_file):
            with open(self.download_history_file, "r") as f:
                return [line.strip() for line in f if line.strip()]
        return []

    def _save_id(self, video_id: str):
        if video_id not in self.downloaded_ids:
            self.downloaded_ids.append(video_id)
            with open(self.download_history_file, "a") as f:
                f.write(video_id + "\n")

    def log_harvest(self, path: str, label: str, confidence: float, source_url: str, duration: float, video_id: str):
        df = pd.read_csv(self.metadata_file)
        new_row = {
            "path": os.path.abspath(path),
            "label": label,
            "confidence": float(confidence),
            "source_url": source_url,
            "duration_sec": float(duration),
            "timestamp": pd.Timestamp.now().isoformat(),
            "video_id": video_id
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.metadata_file, index=False)
        self._save_id(video_id)

    def search_and_download(self, label: str, count: int = 5) -> List[dict]:
        label_dir = os.path.join(self.harvest_root, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Diversify search queries
        queries = [
            f"{label} sound effect",
            f"pure {label} recording",
            f"natural {label} sounds",
            f"{label} in nature"
        ]
        
        harvested_info = []
        for q in queries:
            if len(harvested_info) >= count: break
            try:
                # Use the improved download function with skip_ids
                files = download_youtube_audio_for_label(q, label_dir, num_results=3, skip_ids=self.downloaded_ids)
                for f in files:
                    vid_id = os.path.splitext(os.path.basename(f))[0]
                    harvested_info.append({"path": f, "id": vid_id})
            except Exception as e:
                print(f"Error harvesting for {label} with query {q}: {e}")
        
        return harvested_info

class SelfLearner:
    """
    Orchestrates the self-learning loop:
    1. Identify weak labels.
    2. Harvest data.
    3. Filter/Pseudo-label.
    4. Update manifest.
    """
    def __init__(self, model_wrapper, harvester: SoundHarvester, manifest_path: str):
        self.model = model_wrapper
        self.harvester = harvester
        self.manifest_path = manifest_path

    def run_cycle(self, performance_map: dict, min_count: int = 10):
        """
        Identify labels with low confidence/score and fetch more data.
        """
        weak_labels = [label for label, score in performance_map.items() if score < 0.6]
        print(f"🤖 Self-Learning: Identified {len(weak_labels)} weak labels: {weak_labels}")
        
        for label in weak_labels:
            print(f"🔍 Harvesting more data for '{label}'...")
            raw_data = self.harvester.search_and_download(label, count=5)
            
            for item in raw_data:
                f = item["path"]
                vid_id = item["id"]
                try:
                    # Pseudo-labeling / Filtering
                    audio, sr = sf.read(f)
                    if len(audio) < sr * 5: continue
                    
                    # Process first minute
                    found_any = False
                    for i in range(0, min(len(audio), sr*60), sr*5):
                        segment = audio[i:i+sr*5]
                        if len(segment) < sr * 5: break
                        
                        # Use model to verify
                        preds = self.model.predict_pcm(segment, sr)
                        match = next((p for p in preds if p['label'] == label), None)
                        
                        if match and match['confidence'] > 0.4:
                            # Keep this high-quality segment
                            seg_name = f"{label}_harvested_{vid_id}_{i}.wav"
                            seg_path = os.path.join(os.path.dirname(f), seg_name)
                            sf.write(seg_path, segment, sr)
                            
                            self.harvester.log_harvest(seg_path, label, match['confidence'], f"https://youtube.com/watch?v={vid_id}", 5.0, vid_id)
                            self._append_to_manifest(seg_path, label)
                            print(f"✅ Saved high-quality segment for {label} (conf: {match['confidence']:.2f})")
                            found_any = True

                    # Clean up the large original file to save space
                    if os.path.exists(f):
                        os.remove(f)
                        
                except Exception as e:
                    print(f"Failed to process harvested file {f}: {e}")

    def _append_to_manifest(self, path: str, label: str):
        entry = {"path": os.path.abspath(path), "labels": [label], "events": []}
        with open(self.manifest_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

# --------------------------- Dataset ---------------------------------------

class UnifiedSoundDataset(Dataset):
    def __init__(self, manifest: str, labels: str, sample_rate: int = DEFAULT_SAMPLE_RATE,
                 clip_sec: float = CLIP_SEC, augment: bool = True, mix_prob: float = 0.6,
                 frame_targets: bool = True):
        with open(manifest, 'r') as f:
            self.items = [json.loads(l) for l in f if l.strip()]
        with open(labels, 'r') as f:
            self.labels = [l.strip() for l in f if l.strip()]
        self.lab2idx = {l: i for i, l in enumerate(self.labels)}
        self.sample_rate = sample_rate
        self.clip_sec = clip_sec
        self.clip_len = int(clip_sec * sample_rate)
        self.augment = augment
        self.mix_prob = mix_prob
        self.frame_targets = frame_targets

        hop_length = int(HOP_SEC * sample_rate)
        win_length = int(WIN_SEC * sample_rate)
        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=N_FFT,
                                              hop_length=hop_length, win_length=win_length, n_mels=N_MELS)
        
        # SpecAugment
        self.spec_augment = nn.Sequential(
            T.FrequencyMasking(freq_mask_param=10),
            T.TimeMasking(time_mask_param=30)
        )
        
        self.hop_seconds = hop_length / sample_rate

        self.segments = []
        for it in self.items:
            try:
                info = sf.info(it['path'])
                duration_sec = info.duration
                num_samples = int(duration_sec * self.sample_rate + 0.5)
                num_segments = max(1, int(np.ceil(num_samples / self.clip_len)))
            except Exception as e:
                print(f"Warning: could not get info for {it['path']}: {e}")
                num_segments = 1
            for seg in range(num_segments):
                start_sample = seg * self.clip_len
                seg_item = {
                    'path': it['path'],
                    'start_sample': start_sample,
                    'clip_len': self.clip_len,
                    'labels': it['labels'],
                    'events': it.get('events', [])
                }
                self.segments.append(seg_item)

    def __len__(self):
        return len(self.segments)

    def load_wave(self, path: str, start_sample: int = 0, clip_len: Optional[int] = None) -> torch.Tensor:
        info = sf.info(path)
        orig_sr = info.samplerate
        total_frames = info.frames
        
        # Optimization: Only read exactly what we need
        if clip_len is None:
            start_frame = 0
            frames_to_read = total_frames
        else:
            start_sec = start_sample / self.sample_rate
            start_frame = int(start_sec * orig_sr)
            # Duration in frames relative to original sample rate
            frames_to_read = min(int((clip_len / self.sample_rate) * orig_sr), total_frames - start_frame)
        
        audio, _ = sf.read(path, start=start_frame, frames=frames_to_read, always_2d=False)
        
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        
        # Skip resampling if sample rates match - HUGE CPU SAVING
        if orig_sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_sr, self.sample_rate)
            
        if clip_len is not None and wav.size(1) < clip_len:
            pad = clip_len - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad))
        return wav

    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(wav)
        # PCEN expects linear scale energy. amp_to_db and normalization here
        # would make values negative, causing NaNs in PCEN's pow(r) operation.
        if self.augment and self.training:
            mel = self.spec_augment(mel)
        # Ensure strictly positive for PCEN stability
        mel = mel + 1e-9
        return mel

    def make_multihot(self, labels: List[str]) -> torch.Tensor:
        y = torch.zeros(len(self.labels), dtype=torch.float32)
        for lab in labels:
            if lab in self.lab2idx:
                y[self.lab2idx[lab]] = 1.0
        return y

    def __getitem__(self, idx):
        seg = self.segments[idx]
        wav = self.load_wave(seg['path'], start_sample=seg['start_sample'], clip_len=seg['clip_len'])
        
        # Correct label calculation: Only assign labels present in this time window
        if seg['events']:
            start_sec = seg['start_sample'] / self.sample_rate
            end_sec = start_sec + (seg['clip_len'] / self.sample_rate)
            current_labels = set()
            for ev in seg['events']:
                # Overlap condition: event starts before segment ends AND event ends after segment starts
                if ev['onset'] < end_sec and ev['offset'] > start_sec:
                    current_labels.add(ev['label'])
            y = self.make_multihot(list(current_labels))
        else:
            # Fallback for datasets without fine-grained events (e.g., ESC50)
            y = self.make_multihot(seg['labels'])
            
        mel = self.wav_to_mel(wav)
        return mel, y, None

# --------------------------- Model -----------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class PCEN(nn.Module):
    def __init__(self, n_mels, alpha=0.98, delta=2.0, r=0.5, s=0.025, eps=1e-6):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.log(torch.tensor([alpha])))
        self.log_delta = nn.Parameter(torch.log(torch.tensor([delta])))
        self.log_r = nn.Parameter(torch.log(torch.tensor([r])))
        self.s = s
        self.eps = eps

    def forward(self, x):
        # x: (B, 1, F, T)
        alpha = torch.exp(self.log_alpha)
        delta = torch.exp(self.log_delta)
        r = torch.exp(self.log_r)
        
        # Simple EMA for smoothing over time (T dim)
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
        # x: (B, T, hidden_size)
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

# --------------------------- Training Loop ---------------------------------

def collate_fn(batch):
    mels = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    return mels, ys, None

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_loop(manifest: str, labels_file: str, out_dir: str,
               epochs: int = 9999, batch_size: int = BATCH_SIZE,
               lr: float = LR, device: str = 'cuda', patience: int = 10):
    ds = UnifiedSoundDataset(manifest, labels_file, augment=True)
    # Hack to set 'training' flag for SpecAugment logic
    ds.training = True 
    
    n_classes = len(ds.labels)
    
    # Stratified Split: Group indices by label
    label_to_idxs = {l: [] for l in ds.labels}
    for i in range(len(ds)):
        # Take the first label as the primary category for splitting
        main_label = ds.segments[i]['labels'][0]
        label_to_idxs[main_label].append(i)
    
    train_idx, val_idx = [], []
    split_ratio = 0.8 # Let's use 80/20 for better data utilization
    
    for label, idxs in label_to_idxs.items():
        random.shuffle(idxs)
        num_train = int(len(idxs) * split_ratio)
        # Ensure at least one sample in each set if possible
        if len(idxs) >= 2 and num_train == len(idxs):
            num_train -= 1
        if len(idxs) >= 2 and num_train == 0:
            num_train = 1
            
        train_idx.extend(idxs[:num_train])
        val_idx.extend(idxs[num_train:])

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    
    print(f"📊 Stratified Split: {len(train_idx)} train, {len(val_idx)} val samples across {n_classes} labels.")
    
    # Validation dataset (no augmentation)
    ds_val = UnifiedSoundDataset(manifest, labels_file, augment=False)
    ds_val.training = False

    # Optimized workers for laptop CPU
    num_cpus = os.cpu_count() or 1
    # On Windows, num_workers > 0 can eat RAM. In WSL2/Linux, it is much more efficient.
    # We use roughly half the CPUs for training and a quarter for validation.
    train_workers = max(1, num_cpus // 2)
    val_workers = max(1, num_cpus // 4)

    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx), 
                              batch_size=batch_size,
                              shuffle=True, 
                              collate_fn=collate_fn, 
                              num_workers=train_workers,
                              persistent_workers=True if train_workers > 0 else False,
                              pin_memory=False) # pin_memory=False for CPU-only training saves RAM

    val_loader = DataLoader(torch.utils.data.Subset(ds_val, val_idx), 
                            batch_size=batch_size,
                            shuffle=False, 
                            collate_fn=collate_fn, 
                            num_workers=val_workers,
                            persistent_workers=True if val_workers > 0 else False,
                            pin_memory=False)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CRNN(n_mels=N_MELS, n_classes=n_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    # Use Focal Loss
    criterion = FocalLoss(gamma=2.0)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    os.makedirs(out_dir, exist_ok=True)
    best_metric = -1; wait = 0
    start_epoch = 0

    # --------------------- RESUME CHECK ---------------------
    last_ckpt = os.path.join(out_dir, "last_checkpoint.pth")
    if os.path.exists(last_ckpt):
        print("💾 Resuming from last_checkpoint.pth ...")
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        best_metric = ck.get("best_metric", best_metric)
        start_epoch = ck.get("epoch", 1)
        print(f"Resumed at epoch {start_epoch}, best_metric={best_metric:.4f}")

    for epoch in range(start_epoch+1, epochs+1):
        if wait>=patience:
            print("⛔ Early stopping triggered"); break
        model.train()
        total_loss=0
        
        for mel, y, _ in tqdm(train_loader, desc=f"Train ep{epoch}"):
            mel = mel.to(device); y = y.to(device)
            
            opt.zero_grad()
            
            # Mixed Precision & Mixup
            if scaler:
                with torch.amp.autocast('cuda'):
                    if random.random() < 0.5: # 50% chance of mixup
                        mel, y_a, y_b, lam = mixup_data(mel, y)
                        logits, _ = model(mel)
                        # No more manual pooling needed, Attention handled it
                        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                    else:
                        logits, _ = model(mel)
                        loss = criterion(logits, y)
                
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                # CPU fallback
                logits, _ = model(mel)
                loss = criterion(logits, y)
                loss.backward()
                opt.step()
                
            total_loss += loss.item()*mel.size(0)
            
        print(f"Epoch {epoch} avg loss={total_loss/len(train_loader.dataset):.6f}")
        
        # validation
        model.eval(); all_truth, all_scores = [], []
        with torch.no_grad():
            for mel, y, _ in val_loader:
                mel = mel.to(device); logits, _ = model(mel)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_scores.append(probs); all_truth.append(y.numpy())
        all_truth = np.vstack(all_truth); all_scores = np.vstack(all_scores)
        
        # Per-class metrics
        perf_map = {}
        for i in range(n_classes):
            if all_truth[:,i].sum() > 0:
                if SKLEARN_AVAILABLE:
                    c_metric = average_precision_score(all_truth[:,i], all_scores[:,i])
                else:
                    c_metric = (all_truth[:,i] * all_scores[:,i]).mean()
                perf_map[ds.labels[i]] = float(c_metric)
        
        if SKLEARN_AVAILABLE:
            metric = np.nanmean(list(perf_map.values()))
        else:
            metric = np.array(list(perf_map.values())).mean()
        
        print(f"Validation metric={metric:.4f}")
        
        # Save checkpiont
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "epoch": epoch + 1,
            "best_metric": best_metric,
            "labels": ds.labels
        }
        torch.save(state, last_ckpt)

        # save best
        if metric > best_metric:
            best_metric = metric; wait=0
            torch.save({"model_state": model.state_dict(), "labels": ds.labels},
                       os.path.join(out_dir, "best_checkpoint.pth"))
            print("🔥 New best model saved!")
        else:
            wait+=1
        
            
    return perf_map

# --------------------------- CLI & Main ------------------------------------

# Add current directory to path so forest_audio_web can be imported
sys.path.append(os.getcwd())

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='../DATA', help='root folder containing datasets')
    p.add_argument('--out_dir', type=str, default='experiments/run4', help='output directory for manifests and models')
    p.add_argument('--manifest', type=str, help='optional: supply manifest.jsonl to skip building')
    p.add_argument('--labels', type=str, default=None, help='optional: labels.txt (one per line)')
    p.add_argument('--epochs', type=int, default=9999)
    p.add_argument('--batch', type=int, default=BATCH_SIZE)
    p.add_argument('--lr', type=float, default=LR)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--self_learn', action='store_true', help='enable automated harvesting loop')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = os.path.join(args.out_dir, 'all_manifest.jsonl')
    labels_file = args.labels or os.path.join(args.out_dir, 'labels.txt')

    if not manifest or not args.labels:
        build_manifest(args.data_root, manifest, labels_file)

    perf_map = train_loop(manifest, labels_file, args.out_dir, epochs=args.epochs,
                          batch_size=args.batch, lr=args.lr, device=args.device)
    
    if args.self_learn:
        print("\n🚀 Starting Self-Learning Cycle...")
        # Load best model for harvesting
        import importlib.util
        wrapper_path = os.path.join(os.getcwd(), "backend", "services", "audio_service", "model_wrapper.py")
        
        if os.path.exists(wrapper_path):
            spec = importlib.util.spec_from_file_location("model_wrapper", wrapper_path)
            mw = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mw)
            ForestAudioAI = mw.ForestAudioAI
            
            best_ckpt = os.path.join(args.out_dir, "best_checkpoint.pth")
            if os.path.exists(best_ckpt):
                model_wrapper = ForestAudioAI(best_ckpt)
                harvester = SoundHarvester(args.data_root, model_wrapper.labels)
                learner = SelfLearner(model_wrapper, harvester, manifest)
                
                learner.run_cycle(perf_map)
                
                print("🔄 Self-Learning cycle complete. Manifest updated. Restarting training with new data...")
                
                # Update args to use the newly updated manifest and restart
                args.manifest = manifest
                args.epochs = 50 # Run fewer epochs for the refinement stage
                
                # Clear memory if possible
                del model_wrapper
                import gc
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Re-run training loop
                train_loop(manifest, labels_file, args.out_dir, epochs=args.epochs,
                           batch_size=args.batch, lr=args.lr, device=args.device)
                
                print("✨ Full Self-Learning pipeline completed.")
            else:
                print(f"⚠️ Best checkpoint not found at {best_ckpt}, skipping self-learning.")
        else:
            print(f"⚠️ Could not find model_wrapper.py at {wrapper_path}")

if __name__ == "__main__":
    main()
