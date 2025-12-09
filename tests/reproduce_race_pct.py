
import pandas as pd
import matplotlib.pyplot as plt

# Simulate loading data (or load actual data if path is correct)
try:
    df_train = pd.read_csv("c:/Projects/Flushot/data/interim/training_imputed.csv")
    print("Data loaded successfully.")
    
    # The missing definition
    race_pct = df_train['race'].value_counts(normalize=True) * 100
    print("race_pct defined:")
    print(race_pct)
    
    # Simulate the plotting usage
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        race_pct.values,
        labels=race_pct.index,
        autopct="%.1f%%",
        startangle=90,
        counterclock=False
        # colors omitted for simplicity as COLOR_PALETTE might be missing
    )
    print("Pie chart created successfully.")

except Exception as e:
    print(f"Error: {e}")
