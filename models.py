from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model with hyperparameter tuning.
    """
    param_grid = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model with hyperparameter tuning.
    """
    param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def create_ensemble(models_list):
    """
    Create a soft voting ensemble from a list of (name, model) tuples.
    """
    ensemble = VotingClassifier(estimators=models_list, voting='soft')
    return ensemble

def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    return joblib.load(path)
