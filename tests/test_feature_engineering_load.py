import pandas as pd
from pathlib import Path
import sys, os

# Add repo root to path
repo_root = os.path.abspath(".")
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src.feature_engineering import engineer_all_features

def test_pipeline():
    print("Testing feature engineering pipeline...")
    
    # Paths (matching what's now in the notebook)
    train_path = Path("data/interim/training_encoded.csv")
    test_path = Path("data/interim/test_encoded.csv")
    
    if not train_path.exists() or not test_path.exists():
        print("Error: Data files not found in data/interim/")
        return

    # Load data
    try:
        train_encoded = pd.read_csv(train_path)
        test_encoded = pd.read_csv(test_path)
        print(f"Loaded data: Train {train_encoded.shape}, Test {test_encoded.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Run pipeline (subset for speed/verification)
    try:
        # Just checking if the function runs without error on a sample
        train_sample = train_encoded.head(100).copy()
        train_enriched = engineer_all_features(train_sample)
        print(f"Pipeline successful. Enriched shape: {train_enriched.shape}")
        
        # Check for expected new columns
        expected_cols = ["PBI", "PVAS_h1n1", "HKI"]
        for col in expected_cols:
            if col in train_enriched.columns:
                print(f"Verified column: {col}")
            else:
                print(f"Missing column: {col}")
                
    except Exception as e:
        print(f"Error running pipeline: {e}")

if __name__ == "__main__":
    test_pipeline()
