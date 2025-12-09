
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
import os
from pathlib import Path

def save_custom_shap_plots(models, X_data, output_dir):
    """
    Generates and saves custom SHAP bar plots for H1N1 and Seasonal Flu.
    
    Args:
        models (dict): Dictionary of trained models {target_name: model_object}.
        X_data (pd.DataFrame): Feature data used for SHAP calculation.
        output_dir (str or Path): Directory to save the plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Configuration for each target
    configs = {
        'h1n1_vaccine': {
            'color': '#F08C66',
            'title': 'SHAP Feature Importance (H1N1)'
        },
        'seasonal_vaccine': {
            'color': '#8AC6D6',
            'title': 'SHAP Feature Importance (Seasonal Flu)'
        }
    }

    results = {}

    for target, model in models.items():
        if target not in configs:
            print(f"Skipping {target} (no config found)")
            continue
            
        print(f"Generating SHAP plot for {target}...")
        config = configs[target]
        
        # Calculate SHAP values
        # Note: TreeExplainer is efficient for tree-based models (LGBM, RF, XGB)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data)
        
        # Determine if binary classification (list of arrays) or regression/single output
        # For binary, shap_values is usually [class_0_values, class_1_values]
        # We want the positive class (index 1)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            vals = shap_values[1]
        elif isinstance(shap_values, list) and len(shap_values) == 1:
             vals = shap_values[0]
        else:
            vals = shap_values

        # Calculate Mean Absolute SHAP values
        # Global importance of each feature
        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        
        # Create a DataFrame for easy handling
        if isinstance(X_data, pd.DataFrame):
            feature_names = X_data.columns
        else:
            feature_names = [f"Feature {i}" for i in range(X_data.shape[1])]
            
        df_shap = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        })
        
        # Sort and select top 10
        # We sort ascending because barh plots from bottom to top
        df_top10 = df_shap.sort_values('importance', ascending=True).tail(10)

        # Plotting
        plt.figure(figsize=(6, 6))
        
        # Horizontal bar plot
        plt.barh(df_top10['feature'], df_top10['importance'], color=config['color'])
        
        # Title and Axis Labels
        plt.title(config['title'], fontsize=14, pad=20)
        plt.xlabel("mean(|SHAP value|)", fontsize=12)
        
        # Hide Y-axis as requested
        ax = plt.gca()
        # ax.get_yaxis().set_visible(False) # Removed to show feature names
        
        # Optional: Remove spines for a cleaner look since axis is hidden
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Hide standard ticks but keep labels
        ax.tick_params(axis='y', length=0)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"shap_importance_{target}.png"
        save_file = output_path / filename
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot to {save_file}")
        results[target] = str(save_file)
        
    return results
