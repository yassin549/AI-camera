@echo off
setlocal
setlocal EnableDelayedExpansion

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Virtualenv Python not found at "%PYTHON_EXE%".
  echo Create it first: py -m venv .venv ^&^& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

set "ENV_FILE=%~dp0.env.backend.local"
if exist "%ENV_FILE%" (
  for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
    set "k=%%A"
    set "v=%%B"
    if not "!k!"=="" if not "!k:~0,1!"=="#" set "!k!=!v!"
  )
)

"%PYTHON_EXE%" main.py --config config.yaml --start-api --no-display %*
