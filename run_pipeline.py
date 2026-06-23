import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocess import clean_text
from src.features import extract_tfidf_features
from src.models import train_logistic_regression, train_random_forest, create_ensemble, save_model
from src.evaluate import evaluate_model
from src.explain import explain_model_shap

def run_pipeline():
    # 1. Load and Preprocess
    print("Step 1: Preprocessing...")
    raw_data_path = 'data/raw/customer_reviews.csv'
    df = pd.read_csv(raw_data_path)
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # 2. Split Data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    # 3. Feature Extraction
    print("Step 2: Feature Extraction...")
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    # Fit on all cleaned text for simplicity, or just train
    vectorizer_features, vectorizer = extract_tfidf_features(df, 'cleaned_review', save_path=vectorizer_path)
    X_train = vectorizer.transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # 4. Model Training
    print("Step 3: Training Models and Creating Ensemble...")
    log_reg = train_logistic_regression(X_train, y_train)
    rf_clf = train_random_forest(X_train, y_train)
    
    ensemble = create_ensemble([
        ('lr', log_reg),
        ('rf', rf_clf)
    ])
    ensemble.fit(X_train, y_train)
    
    save_model(ensemble, 'models/ensemble_voting.pkl')
    # Also save as the primary model for deployment
    save_model(ensemble, 'models/logistic_best.pkl') 
    
    # 5. Evaluation
    print("Step 4: Evaluating Ensemble...")
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred, y_prob, 'reports')
    
    # 6. Interpretability (Optional/Simplified for pipeline)
    print("Step 5: Generating Interpretability Reports...")
    # explain_model_shap(model, X_train.toarray(), X_test.toarray(), 'models/shap_explainer.pkl')
    
    print("Pipeline completed successfully!")

    # Automatically open dashboard in default web browser
    try:
        import webbrowser
        dashboard_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dashboard.html'))
        if os.path.exists(dashboard_path):
            print(f"🚀 Automatically opening dashboard: {dashboard_path}")
            webbrowser.open(f"file:///{dashboard_path}")
        else:
            local_dashboard = os.path.abspath('dashboard.html')
            if os.path.exists(local_dashboard):
                print(f"🚀 Automatically opening dashboard: {local_dashboard}")
                webbrowser.open(f"file:///{local_dashboard}")
    except Exception as e:
        print(f"Could not automatically open dashboard: {e}")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    run_pipeline()
