import random
import numpy as np
import torch
from .config import RANDOM_SEED

def set_seed(seed: int = None):
    if seed is None:
        seed = RANDOM_SEED
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Deterministic seed set to: {seed}")

if __name__ == "__main__":
    set_seed()
