import os
import shutil
import sys
import subprocess

try:
    import yt_dlp
except ImportError:
    print("Please install yt-dlp: pip install yt-dlp")
    sys.exit(1)

# Try to import static_ffmpeg if available
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

DATA_ROOT = "Data"
TRAIN_DIR = os.path.join(DATA_ROOT, "youtube_train")
EVAL_DIR = os.path.join(DATA_ROOT, "youtube_eval")

# Illegal / Dangerous sounds to add
ILLEGAL_CLASSES = [
    "gunshot",
    "rifle_shot",
    "shotgun",
    "cap_gun",
    "chainsaw",
    "axe_chopping_wood",
    "vehicle_engine",
    "diesel_generator",
    "trap_snap",
    "footsteps_on_leaves", # Poachers walking
    "human_shout",
    "human_whisper"
]

def check_ffmpeg():
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return True
    return False

def download_youtube_audio_for_label(label: str, output_dir: str, num_results: int = 5) -> None:
    os.makedirs(output_dir, exist_ok=True)
    query = f"ytsearch{num_results}:{label} sound"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}],
        "ignoreerrors": True, # Skip if download fails
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(query, download=True)

def main():
    print("Checking for ffmpeg...")
    if not check_ffmpeg():
        print("WARNING: ffmpeg or ffprobe not found in PATH.")
        print("Please install ffmpeg (e.g. via 'winget install ffmpeg' or 'choco install ffmpeg')")
        print("or 'pip install static-ffmpeg'.")
        print("Without ffmpeg, audio extraction will fail.")
        # Proceeding anyway just in case static-ffmpeg works magically or user fixed it.
    
    print("Downloading Illegal/Dangerous sound classes...")
    
    for label in ILLEGAL_CLASSES:
        print(f"\n--- Processing {label} ---")
        
        # Train: 20 samples
        train_label_dir = os.path.join(TRAIN_DIR, label)
        if not os.path.exists(train_label_dir) or len(os.listdir(train_label_dir)) < 5:
            print(f"Downloading train samples for {label}...")
            download_youtube_audio_for_label(label, train_label_dir, num_results=20)
        else:
            print(f"Train samples for {label} already exist.")

        # Eval: 5 samples
        eval_label_dir = os.path.join(EVAL_DIR, label)
        if not os.path.exists(eval_label_dir) or len(os.listdir(eval_label_dir)) < 2:
            print(f"Downloading eval samples for {label}...")
            download_youtube_audio_for_label(label, eval_label_dir, num_results=5)
        else:
            print(f"Eval samples for {label} already exist.")

    print("\nDone! You can now run trainer.py to rebuild the manifest and train.")

if __name__ == "__main__":
    main()
