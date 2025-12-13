@echo off
echo Running Phase 1 Pipeline...

echo [1/4] Downloading Data...
python src/data/download_data.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [2/4] Cleaning Data...
python src/data/clean.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [3/4] Preprocessing & Aligning...
python src/data/preprocess.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [4/4] Splitting Data...
python src/data/split.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo Phase 1 Complete.
