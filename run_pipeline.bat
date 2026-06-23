@echo off
title Sentiment AI Pipeline & Server Runner
color 0E

echo ===================================================
echo   🧠 STARTING SENTIMENT ANALYSIS PIPELINE 🧠
echo ===================================================
echo.

echo [1/3] Running hyperparameter training pipeline...
python train_model.py --data data/raw/customer_reviews.csv --output reports
if %errorlevel% neq 0 (
    echo.
    echo ❌ ERROR: Model training failed. Make sure python and scikit-learn are installed.
    pause
    exit /b %errorlevel%
)
echo.
echo ✅ Training successful! Evaluation figures and metrics saved in 'reports/'.
echo.

echo [2/3] Deploying trained pipeline and plots to Flask app...
if not exist reports\images mkdir reports\images
set "DEPLOYED="

for /d %%i in (reports\*) do (
    if exist "%%i\sentiment_pipeline.pkl" (
        copy /Y "%%i\sentiment_pipeline.pkl" "app\model.pkl" >nul
        set DEPLOYED=1
    )
    if exist "%%i\confusion_matrix.png" (
        copy /Y "%%i\confusion_matrix.png" "reports\images\confusion_matrix.png" >nul
    )
    if exist "%%i\feature_importance.png" (
        copy /Y "%%i\feature_importance.png" "reports\images\feature_importance.png" >nul
    )
    if exist "%%i\roc_curve.png" (
        copy /Y "%%i\roc_curve.png" "reports\images\roc_curve.png" >nul
    )
)

if not defined DEPLOYED (
    echo.
    echo ❌ ERROR: Could not locate sentiment_pipeline.pkl inside reports/ folder.
    pause
    exit /b 1
)
echo ✅ Model pipeline and evaluation plots deployed successfully!
echo.

echo [3/3] Launching Flask Web Server...
echo ---------------------------------------------------
echo Open your browser and go to: http://127.0.0.1:5000/
echo ---------------------------------------------------
echo.
python app/app.py

pause
