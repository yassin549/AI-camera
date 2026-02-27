@echo off
setlocal

set "CF_CONFIG=%~dp0..\cloudflared\config.yml"
if not exist "%CF_CONFIG%" (
  echo [ERROR] Missing cloudflared config file: "%CF_CONFIG%"
  echo Copy cloudflared\config.yml.example to cloudflared\config.yml and fill tunnel values first.
  exit /b 1
)

cloudflared tunnel --config "%CF_CONFIG%" run

