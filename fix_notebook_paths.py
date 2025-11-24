import json
from pathlib import Path

notebook_path = Path("notebooks/sprint2_feature_engineering.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Iterate through cells to find the one loading data
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        modified = False
        for line in cell["source"]:
            if 'train_path = Path("../data/processed/training_encoded.csv")' in line:
                new_source.append(line.replace("processed", "interim"))
                modified = True
            elif 'test_path = Path("../data/processed/test_encoded.csv")' in line:
                new_source.append(line.replace("processed", "interim"))
                modified = True
            else:
                new_source.append(line)
        
        if modified:
            cell["source"] = new_source
            print("Updated paths in notebook cell.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook saved.")
