import argparse
import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from backend.utils.config import config, RANDOM_SEED
from backend.pipelines.data_generation.generate_dataset import DataGenerator
from backend.pipelines.data_generation.labeling import assign_risk_label

def main():
    parser = argparse.ArgumentParser(description="Generate structured training data for Forest Audio AI.")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="datasets/structured/v1.csv", help="Output CSV path.")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting generation of {args.n_samples} samples with seed {args.seed}...")
    
    generator = DataGenerator(seed=args.seed)
    data = []
    
    for _ in tqdm(range(args.n_samples), desc="Generating rows"):
        row = generator.generate_row()
        row["risk_label"] = assign_risk_label(row)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save CSV
    df.to_csv(args.output, index=False)
    print(f"✅ Dataset saved to {args.output}")
    
    # Print class distribution
    print("\n📊 Class Distribution:")
    dist = df["risk_label"].value_counts(normalize=True) * 100
    for label, percent in dist.items():
        count = df["risk_label"].value_counts()[label]
        print(f"   {label:7}: {percent:5.1f}% ({count} samples)")
    
    # Save metadata sidecar
    meta_path = args.output.replace(".csv", "_meta.json")
    metadata = {
        "n_samples": args.n_samples,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "features": list(df.columns),
        "class_distribution": df["risk_label"].value_counts().to_dict(),
        "region_distribution": df["region"].value_counts().to_dict(),
        "season_distribution": df["season"].value_counts().to_dict()
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"📝 Metadata saved to {meta_path}")

if __name__ == "__main__":
    main()
