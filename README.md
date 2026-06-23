# Advanced Sentiment Analysis Project

This project implements a modular, professional sentiment analysis pipeline for customer reviews.

## Project Structure
- `data/`: Raw and processed review data.
- `src/`: Core logic for preprocessing, feature extraction, modeling, and evaluation.
- `notebooks/`: Exploratory Data Analysis and experimentation.
- `app/`: Flask API for real-time sentiment prediction.
- `reports/`: Performance metrics and visualizations.
- `models/`: Saved model artifacts.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run Pipeline**: Train the model and generate reports.
   ```bash
   python run_pipeline.py
   ```
2. **Run API**: Start the prediction server.
   ```bash
   python app/app.py
   ```

## Features
- **Advanced Preprocessing**: Lemmatization, negation handling, and noise removal.
- **Ensemble Modeling**: Voting classifiers combining multiple models.
- **Interpretability**: SHAP explanations for model transparency.
- **Flask Deployment**: RESTful API for integration.
