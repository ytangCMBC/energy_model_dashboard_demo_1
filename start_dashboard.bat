@echo off
setlocal

set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

set "PYTHON_EXE=%APP_DIR%env\python.exe"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH=%APP_DIR%src"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

if not exist "%PYTHON_EXE%" (
    echo Could not find the packaged Python environment:
    echo %PYTHON_EXE%
    echo.
    echo Please make sure the env folder is next to start_dashboard.bat.
    pause
    exit /b 1
)

echo Starting Energy Model Dashboard...
echo.
echo If the browser does not open automatically, use:
echo http://localhost:8501
echo.

"%PYTHON_EXE%" -m streamlit run "%APP_DIR%src\beb_all_panels_dashboard.py" --server.port 8501 --server.headless false --browser.gatherUsageStats false

echo.
echo Dashboard stopped.
pause
