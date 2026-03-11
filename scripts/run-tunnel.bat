@echo off
setlocal

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8080"

where ngrok >nul 2>&1
if errorlevel 1 (
  echo [ERROR] ngrok is not installed or not on PATH.
  echo Install ngrok and run: ngrok config add-authtoken ^<YOUR_NGROK_TOKEN^>
  exit /b 1
)

echo [INFO] Starting ngrok HTTP tunnel on port %PORT%
ngrok http %PORT%
