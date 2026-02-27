@echo off
setlocal

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Virtualenv Python not found at "%PYTHON_EXE%".
  echo Create it first: py -m venv .venv ^&^& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

"%PYTHON_EXE%" main.py --config config.yaml --start-api --no-display %*
