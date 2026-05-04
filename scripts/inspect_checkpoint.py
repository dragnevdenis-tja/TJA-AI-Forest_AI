import torch

def inspect():
    checkpoint = "best_checkpoint.pth"
    try:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        print(f"Keys in checkpoint: {ck.keys()}")
        if 'model_state' in ck:
            print("\nModel state dict keys:")
            keys = list(ck['model_state'].keys())
            for k in keys:
                print(f"  {k}")
            print(f"\nTotal keys: {len(keys)}")
            
            # Check for specific keys
            print("\nChecking for PCEN/Attention keys:")
            search_keys = ["pcen.log_alpha", "pcen_bn.weight", "attention.attn.weight"]
            for sk in search_keys:
                found = sk in ck['model_state']
                print(f"  {sk}: {'Found' if found else 'NOT FOUND'}")
                
            if not any(sk in ck['model_state'] for sk in search_keys):
                print("\nSuggestion: Check if model was trained with an older version of trainer.py")
        
        if 'labels' in ck:
            print(f"\nLabels: {ck['labels']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
