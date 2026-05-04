import os
import sys
import importlib.util

# Add the directory containing 'forest-audio-web' to sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

def test():
    checkpoint = "best_checkpoint.pth"
    if not os.path.exists(checkpoint):
        print(f"Checkpoint {checkpoint} not found")
        return

    try:
        # Import ForestAudioAI using importlib because of hyphen in folder name
        spec = importlib.util.spec_from_file_location(
            "model_wrapper", 
            os.path.join(root_dir, "forest-audio-web", "app", "model_wrapper.py")
        )
        model_wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_wrapper)
        ForestAudioAI = model_wrapper.ForestAudioAI
        
        model = ForestAudioAI(checkpoint)
        print("Model loaded successfully")
        print(f"Labels: {model.labels}")
        
        # Test with one of the mp3 files
        test_file = "AuenwaldWasser.mp3"
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
