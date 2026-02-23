@echo off
setlocal

set CONFIG=config.yaml
if not "%~1"=="" set CONFIG=%~1

python main.py --config %CONFIG% --start-api %*
