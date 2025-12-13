@echo off
echo Running Fine-Tuning Chunk...
echo This script resumes from the best checkpoint and trains for 10 more epochs.
echo You can run this multiple times to improve accuracy incrementally.

python src/training/train.py --epochs 10 --resume

if %ERRORLEVEL% NEQ 0 (
    echo Training Chunk Failed.
    exit /b %ERRORLEVEL%
)
echo Chunk Complete. Check validation accuracy above.
