# pyrefly: ignore [missing-import]
import shap
import joblib
import os

def explain_model_shap(model, X_train, X_test, output_path):
    """
    Generate SHAP explanations for the model.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    
    # Save explainer for later use
    joblib.dump(explainer, output_path)
    print(f"SHAP explainer saved to {output_path}")
    
    return shap_values

def plot_top_features(shap_values, feature_names, save_path):
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    import matplotlib.pyplot as plt
    plt.savefig(save_path)
    plt.close()
