# train_sentiment.py
import argparse
import joblib
import os
import time
import logging
import json
import re
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

# Optional: imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced Sentiment Analysis Training & Inference")
    # Data arguments
    parser.add_argument('--data', type=str, help='Path to CSV file for training')
    parser.add_argument('--text-col', type=str, default='text', help='Column containing text')
    parser.add_argument('--target-col', type=str, default='sentiment', help='Column containing label')
    # Model & vectorizer
    parser.add_argument('--model-type', type=str, choices=['lr', 'rf', 'dt', 'svc'], default='lr')
    parser.add_argument('--vectorizer', type=str, choices=['tfidf', 'count'], default='tfidf')
    parser.add_argument('--max-features', type=int, default=5000)
    parser.add_argument('--ngram-range', type=int, nargs=2, default=[1, 2])
    # Preprocessing
    parser.add_argument('--use-stopwords', action='store_true', help='Remove English stopwords')
    parser.add_argument('--use-stemming', action='store_true', help='Apply Porter stemming')
    # Training
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--output-dir', '--output', type=str, default='output', dest='output_dir')
    parser.add_argument('--handle-imbalance', action='store_true',
                        help='Apply SMOTE (requires imbalanced-learn)')
    # Inference mode
    parser.add_argument('--predict', type=str, help='Single text string to predict (requires trained model)')
    parser.add_argument('--model-dir', type=str, help='Directory containing saved model and label encoder')
    return parser.parse_args()


def preprocess_text(text: str, use_stopwords: bool = False, use_stemming: bool = False) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if use_stopwords or use_stemming:
        try:
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer
            try:
                stop_words = set(stopwords.words('english'))
            except LookupError:
                import nltk
                nltk.download('stopwords')
                stop_words = set(stopwords.words('english'))
        except ImportError:
            logger.warning("NLTK not installed. Stopwords/stemming disabled.")
            return text

        tokens = text.split()
        if use_stopwords:
            tokens = [t for t in tokens if t not in stop_words]
        if use_stemming:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(t) for t in tokens]
        text = ' '.join(tokens)
    return text


def create_vectorizer(vectorizer_type: str, max_features: int, ngram_range: tuple, use_stopwords: bool):
    stop_words = 'english' if use_stopwords else None
    if vectorizer_type == 'tfidf':
        return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                               stop_words=stop_words, lowercase=True)
    else:
        return CountVectorizer(max_features=max_features, ngram_range=ngram_range,
                               stop_words=stop_words, lowercase=True)


def create_model(model_type: str, random_state: int, class_weight: Optional[str] = None):
    if model_type == 'lr':
        return LogisticRegression(random_state=random_state, max_iter=1000,
                                  class_weight=class_weight, n_jobs=-1)
    elif model_type == 'rf':
        return RandomForestClassifier(random_state=random_state, class_weight=class_weight, n_jobs=-1)
    elif model_type == 'dt':
        return DecisionTreeClassifier(random_state=random_state, class_weight=class_weight)
    elif model_type == 'svc':
        return SVC(random_state=random_state, class_weight=class_weight, probability=True)
    else:
        raise ValueError(f"Unsupported model: {model_type}")


def get_param_grid(model_type: str, vectorizer_type: str) -> Dict[str, List[Any]]:
    param_grid = {}
    # Vectorizer params
    if vectorizer_type == 'tfidf':
        param_grid['vec__use_idf'] = [True, False]
        param_grid['vec__smooth_idf'] = [True, False]
    param_grid['vec__max_df'] = [0.75, 1.0]
    param_grid['vec__min_df'] = [1, 2]

    # Classifier params
    if model_type == 'lr':
        param_grid['clf__C'] = [0.1, 1.0, 10.0]
        param_grid['clf__penalty'] = ['l2']
        param_grid['clf__solver'] = ['lbfgs', 'liblinear']
    elif model_type == 'rf':
        param_grid['clf__n_estimators'] = [50, 100, 200]
        param_grid['clf__max_depth'] = [None, 10, 20]
        param_grid['clf__min_samples_split'] = [2, 5]
        param_grid['clf__min_samples_leaf'] = [1, 2]
    elif model_type == 'dt':
        param_grid['clf__max_depth'] = [None, 5, 10, 20]
        param_grid['clf__min_samples_split'] = [2, 5, 10]
        param_grid['clf__criterion'] = ['gini', 'entropy']
    elif model_type == 'svc':
        param_grid['clf__C'] = [0.1, 1.0, 10.0]
        param_grid['clf__kernel'] = ['linear', 'rbf']
        param_grid['clf__gamma'] = ['scale', 'auto']
    return param_grid


def plot_confusion_matrix(y_true, y_pred, target_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_feature_importance(pipeline, save_path, top_n=20):
    model = pipeline.named_steps['clf']
    vectorizer = pipeline.named_steps['vec']
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Feature importance plot saved to {save_path}")
    elif hasattr(model, 'coef_'):
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        top_pos = np.argsort(coef)[-top_n:][::-1]
        top_neg = np.argsort(coef)[:top_n]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.barh(range(len(top_pos)), coef[top_pos], align='center')
        ax1.set_yticks(range(len(top_pos)))
        ax1.set_yticklabels([feature_names[i] for i in top_pos])
        ax1.set_title('Top positive features')
        ax1.invert_yaxis()
        ax2.barh(range(len(top_neg)), coef[top_neg], align='center')
        ax2.set_yticks(range(len(top_neg)))
        ax2.set_yticklabels([feature_names[i] for i in top_neg])
        ax2.set_title('Top negative features')
        ax2.invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Top coefficient plot saved to {save_path}")
    else:
        logger.warning("Model does not provide feature importances or coefficients.")


def plot_roc_curve(best_pipeline, X_test, y_test, save_path):
    try:
        if hasattr(best_pipeline, 'predict_proba'):
            y_prob = best_pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            logger.info(f"ROC curve saved to {save_path}")
        else:
            logger.warning("Classifier does not support probability predictions. ROC curve skipped.")
    except Exception as e:
        logger.warning(f"Could not plot ROC curve: {e}")



def train(args: argparse.Namespace) -> Tuple[Pipeline, List[str], LabelEncoder]:
    """Main training pipeline."""
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    
    # Smart column fallback logic to prevent crashes
    if args.text_col not in df.columns:
        for alternative in ['review', 'cleaned_review', 'reviews', 'text', 'comment', 'sentence']:
            if alternative in df.columns:
                logger.info(f"Column '{args.text_col}' not found. Auto-falling back to text column: '{alternative}'")
                args.text_col = alternative
                break
                
    if args.target_col not in df.columns:
        for alternative in ['sentiment', 'label', 'target', 'class', 'rating']:
            if alternative in df.columns:
                logger.info(f"Column '{args.target_col}' not found. Auto-falling back to target column: '{alternative}'")
                args.target_col = alternative
                break

    if args.text_col not in df.columns or args.target_col not in df.columns:
        raise ValueError(f"Columns '{args.text_col}' and/or '{args.target_col}' not found. Available columns in CSV: {list(df.columns)}")

    df = df.dropna(subset=[args.text_col, args.target_col])
    logger.info(f"Dataset shape after dropping missing: {df.shape}")

    # Preprocessing
    logger.info("Applying text cleaning...")
    df['cleaned_text'] = df[args.text_col].apply(
        lambda x: preprocess_text(x, args.use_stopwords, args.use_stemming)
    )
    df = df[df['cleaned_text'].str.strip() != '']
    logger.info(f"After removing empty texts: {df.shape}")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[args.target_col])
    target_names = [str(c) for c in label_encoder.classes_.tolist()]
    X_text = df['cleaned_text'].values

    logger.info(f"Classes: {target_names}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Class weight or SMOTE
    use_smote = args.handle_imbalance and SMOTE_AVAILABLE
    if args.handle_imbalance and not SMOTE_AVAILABLE:
        logger.warning("SMOTE requested but imbalanced-learn not installed. Falling back to class_weight='balanced'.")

    if use_smote:
        logger.info("Using SMOTE for imbalance handling.")
        # Pipeline with SMOTE after vectorization is complex; we'll apply SMOTE on the vectorized features manually during grid search.
        # For simplicity, we build a pipeline without SMOTE for tuning, then train final model with SMOTE if needed.
        # Here we use class_weight='balanced' + SMOTE inside the final model (see below).
        class_weight = None
    else:
        class_weight = 'balanced' if args.model_type in ['lr', 'svc', 'rf', 'dt'] else None

    # Base pipeline
    vectorizer = create_vectorizer(args.vectorizer, args.max_features,
                                   tuple(args.ngram_range), args.use_stopwords)
    classifier = create_model(args.model_type, args.random_seed, class_weight)
    pipeline = Pipeline([('vec', vectorizer), ('clf', classifier)])

    # Hyperparameter tuning
    param_grid = get_param_grid(args.model_type, args.vectorizer)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_seed)

    logger.info(f"Starting GridSearchCV for {args.model_type.upper()}")
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info(f"GridSearch completed in {elapsed:.2f}s")

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    logger.info(f"Best parameters: {best_params}")

    # If SMOTE is requested, we rebuild the best pipeline with SMOTE after vectorization
    if use_smote:
        logger.info("Applying SMOTE to the training set (after vectorization).")
        # Transform training data with the best vectorizer
        best_vec = best_pipeline.named_steps['vec']
        X_train_vec = best_vec.transform(X_train)
        smote = SMOTE(random_state=args.random_seed)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)
        # Train the same classifier on resampled data
        best_clf = best_pipeline.named_steps['clf']
        best_clf.fit(X_train_resampled, y_train_resampled)
        # Replace classifier in pipeline
        best_pipeline.named_steps['clf'] = best_clf
        logger.info("SMOTE applied successfully.")

    # Cross-validation on training data
    cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    logger.info(f"Training CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Test evaluation
    y_pred = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=target_names))

    # Save artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{args.model_type}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    joblib.dump(best_pipeline, os.path.join(run_dir, 'sentiment_pipeline.pkl'))
    joblib.dump(label_encoder, os.path.join(run_dir, 'label_encoder.pkl'))

    # Metrics JSON
    metrics = {
        'best_params': best_params,
        'train_cv_mean_accuracy': float(np.mean(cv_scores)),
        'train_cv_std_accuracy': float(np.std(cv_scores)),
        'test_accuracy': test_acc,
        'training_time_seconds': elapsed,
        'classification_report': classification_report(y_test, y_pred, target_names=target_names, output_dict=True),
        'class_distribution': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        'preprocessing': {
            'use_stopwords': args.use_stopwords,
            'use_stemming': args.use_stemming,
            'max_features': args.max_features,
            'ngram_range': args.ngram_range
        }
    }
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Plots
    plot_confusion_matrix(y_test, y_pred, target_names, os.path.join(run_dir, 'confusion_matrix.png'))
    plot_feature_importance(best_pipeline, os.path.join(run_dir, 'feature_importance.png'))
    plot_roc_curve(best_pipeline, X_test, y_test, os.path.join(run_dir, 'roc_curve.png'))

    # ── Terminal Visualizations (Decision Tree & Feature Graphs) ──
    try:
        from sklearn.tree import DecisionTreeClassifier
        model = best_pipeline.named_steps['clf']
        vectorizer = best_pipeline.named_steps['vec']
        feature_names = list(vectorizer.get_feature_names_out())
        
        # 1. If Decision Tree, print the visual hierarchical tree structure
        if isinstance(model, DecisionTreeClassifier):
            from sklearn.tree import export_text
            tree_rules = export_text(model, feature_names=feature_names, max_depth=3)
            logger.info("\n" + "="*55 + "\n🌳 DECISION TREE HIERARCHY STRUCTURE 🌳\n" + "="*55 + "\n" + tree_rules)
            
        # 2. Print an ASCII horizontal bar chart graph of top keywords
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            logger.info("\n" + "="*55 + "\n📊 TOP FEATURE IMPORTANCES (TERMINAL GRAPH) 📊\n" + "="*55)
            max_imp = max(importances[indices]) if len(indices) > 0 else 1.0
            for idx in indices:
                bar_scale = int((importances[idx] / max_imp) * 30) if max_imp > 0 else 0
                bar = '█' * bar_scale
                logger.info(f"  {feature_names[idx]:<15} | {importances[idx]:.4f} {bar}")
        elif hasattr(model, 'coef_'):
            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            indices = np.argsort(np.abs(coef))[::-1][:10]
            logger.info("\n" + "="*55 + "\n📊 TOP COEFFICIENT WEIGHTS (TERMINAL GRAPH) 📊\n" + "="*55)
            max_coef = max(np.abs(coef[indices])) if len(indices) > 0 else 1.0
            for idx in indices:
                val = coef[idx]
                bar_scale = int((abs(val) / max_coef) * 30) if max_coef > 0 else 0
                bar = ('█' if val > 0 else '░') * bar_scale
                logger.info(f"  {feature_names[idx]:<15} | {val:+.4f} {bar}")
    except Exception as e:
        logger.warning(f"Could not generate terminal graph visualization: {e}")

    # Summary text
    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write(f"Sentiment Analysis Model: {args.model_type.upper()}\n")
        f.write(f"Vectorizer: {args.vectorizer}\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Training CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n")
        f.write(f"Classes: {target_names}\n")

    logger.info(f"All artifacts saved to {run_dir}")
    return best_pipeline, target_names, label_encoder


def predict_text(text: str, model_dir: str, use_stopwords: bool = False, use_stemming: bool = False) -> Dict[str, Union[str, float]]:
    """Load a trained model and predict sentiment for a single text."""
    pipeline_path = os.path.join(model_dir, 'sentiment_pipeline.pkl')
    encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    if not os.path.exists(pipeline_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Model or encoder not found in {model_dir}")

    pipeline = joblib.load(pipeline_path)
    label_encoder = joblib.load(encoder_path)

    # Preprocess text using the same settings used during training.
    # Note: The preprocessing flags need to be stored with the model for full accuracy.
    # Here we assume they are known from training. For better practice, save them in a config file.
    cleaned = preprocess_text(text, use_stopwords, use_stemming)
    pred_id = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0] if hasattr(pipeline, 'predict_proba') else None
    sentiment = label_encoder.inverse_transform([pred_id])[0]

    result = {'text': text, 'predicted_sentiment': sentiment, 'confidence': None}
    if proba is not None:
        result['confidence'] = float(np.max(proba))
    return result


def main():
    args = parse_args()

    # Inference mode
    if args.predict:
        if not args.model_dir:
            logger.error("--model-dir required for prediction.")
            return
        # For prediction, the preprocessing settings must match training.
        # We'll assume the user knows the settings used. A better way: save preproc config.
        # Here we read from the saved metrics.json if present.
        config_path = os.path.join(args.model_dir, 'metrics.json')
        use_stopwords = args.use_stopwords  # fallback to command line
        use_stemming = args.use_stemming
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                use_stopwords = config['preprocessing'].get('use_stopwords', False)
                use_stemming = config['preprocessing'].get('use_stemming', False)
        result = predict_text(args.predict, args.model_dir, use_stopwords, use_stemming)
        print(json.dumps(result, indent=2))
        return

    # Training mode
    if not args.data:
        logger.error("Training requires --data. Use --predict for inference.")
        return
    train(args)


if __name__ == "__main__":
    main()