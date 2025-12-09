from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from lightgbm import LGBMClassifier

# Define the custom configuration for the plots
cm_config = {
    "h1n1_vaccine": {
        "color": "#58ADC5",
        "title": "Normalized CM, H1N1",
        "filename": "h1n1_lgbm_norm_updated.png"
    },
    "seasonal_vaccine": {
        "color": "#EA5B25",
        "title": "Normalized CM, Seasonal Flu",
        "filename": "seasonal_lgbm_norm_updated.png"
    }
}

# Ensure the output directory exists
# ARTIFACTS_COMPARISON is defined in the notebook setup
ARTIFACTS_COMPARISON.mkdir(parents=True, exist_ok=True)

# Loop through each target to train the best model (LightGBM) and generate the plot
for target in TARGET_COLS:
    print(f"Processing {target}...")
    
    # 1. Select the specific target data
    y_train_curr = y_train[target]
    y_val_curr = y_val[target]
    
    # 2. Retrain the LightGBM model (Best Model)
    # Re-initializing to ensure a fresh fit for each target
    # Using parameters consistent with the notebook's LightGBM configuration
    model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train_curr)
    
    # 3. Predict on validation set
    y_pred = model.predict(X_val)
    
    # 4. Compute Normalized Confusion Matrix
    # normalize='true' normalizes over the true labels (rows)
    cm_norm = confusion_matrix(y_val_curr, y_pred, normalize='true')
    
    # 5. Create Custom Colormap (White -> Target Color)
    target_config = cm_config[target]
    custom_cmap = LinearSegmentedColormap.from_list(
        f"cmap_{target}", ["#ffffff", target_config["color"]]
    )
    
    # 6. Plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt=".2f", 
        cmap=custom_cmap, 
        cbar=True,
        square=True
    )
    
    plt.title(target_config["title"], fontsize=12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # 7. Save
    save_path = ARTIFACTS_COMPARISON / target_config["filename"]
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved normalized confusion matrix to: {save_path}")
