import os
import sys
import importlib.util

# New infrastructure imports
from backend.utils.config import AUDIO_MODEL_PATH, config

# Add the root directory to sys.path
root_dir = os.getcwd()
sys.path.append(root_dir)

def test():
    checkpoint = os.path.join(root_dir, AUDIO_MODEL_PATH)
    if not os.path.exists(checkpoint):
        print(f"Checkpoint {checkpoint} not found")
        return

    try:
        # Import ForestAudioAI
        from backend.services.audio_service.model_wrapper import ForestAudioAI
        
        model = ForestAudioAI(checkpoint)
        print("Model loaded successfully")
        print(f"Labels: {model.labels}")
        
        # Test with one of the mp3 files
        samples_dir = config.paths.get("audio_samples_path", "audio_samples")
        test_file = os.path.join(root_dir, samples_dir, "AuenwaldWasser.mp3")
        
        if os.path.exists(test_file):
            with open(test_file, "rb") as f:
                audio_bytes = f.read()
            
            print(f"Predicting for {test_file}...")
            results = model.predict(audio_bytes)
            print("Results:")
            for res in results:
                print(f"  {res['label']}: {res['confidence']:.4f}")
        else:
            print(f"Test file {test_file} not found")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
