@echo off
echo Checking/Installing TensorFlow...
pip install tensorflow

echo Starting Training Pipeline...
python src/training/train.py
if %ERRORLEVEL% NEQ 0 (
    echo Training failed or stopped.
    exit /b %ERRORLEVEL%
)
echo Training Complete.
