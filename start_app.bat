@echo off
echo ========================================================
echo Starting Citizen App Image Triage Tester...
echo ========================================================
echo.

echo Installing dependencies (Add "REM" before this line after first use to skip it)
::pip install -r "%~dp0requirements.txt"
echo.

echo Please wait for the AI models to load.
echo Once loaded, the app will be available at:
echo http://127.0.0.1:8080
echo ========================================================
echo.

cd /d "%~dp0image_detection_ai"
uvicorn DetectImage:app --host 127.0.0.1 --port 8080 --reload
pause
