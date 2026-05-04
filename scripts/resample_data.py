import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

# Config
DATA_DIR = "Data"
TARGET_SR = 32000
# Using the static_ffmpeg found in the venv
FFMPEG_PATH = r".venv\Scripts\static_ffmpeg.exe"

def get_audio_files(root_dir):
    extensions = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')
    audio_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files

def resample_file(file_path):
    try:
        # Check current sample rate to avoid unnecessary work
        info = sf.info(file_path)
        if info.samplerate == TARGET_SR and file_path.lower().endswith('.wav'):
            return "skipped"
        
        temp_file = file_path + ".tmp.wav"
        
        # ffmpeg command: -y (overwrite), -i (input), -ar (sample rate), -ac 1 (mono)
        cmd = [
            FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
            "-i", file_path,
            "-ar", str(TARGET_SR),
            "-ac", "1",
            temp_file
        ]
        
        subprocess.run(cmd, check=True)
        
        # Replace original with resampled wav
        # Note: We change extension to .wav for consistency if it wasn't already
        final_path = os.path.splitext(file_path)[0] + ".wav"
        
        # If the original was not a .wav, we remove it
        if file_path != final_path:
            os.remove(file_path)
            
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(temp_file, final_path)
        
        return "resampled"
    except Exception as e:
        return f"error: {str(e)}"

def main():
    print(f"🔍 Scanning {DATA_DIR} for audio files...")
    files = get_audio_files(DATA_DIR)
    print(f"Found {len(files)} files. Starting parallel resampling...")

    # Use all CPU cores
    num_workers = os.cpu_count()
    
    results = {"resampled": 0, "skipped": 0, "error": 0}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(executor.map(resample_file, files), total=len(files)):
            if res == "resampled":
                results["resampled"] += 1
            elif res == "skipped":
                results["skipped"] += 1
            else:
                results["error"] += 1
                # print(res) # Uncomment for debugging

    print("\n✅ Resampling Complete!")
    print(f"   - Resampled: {results['resampled']}")
    print(f"   - Already correct: {results['skipped']}")
    print(f"   - Errors: {results['error']}")

if __name__ == "__main__":
    main()
