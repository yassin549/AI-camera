@echo off
setlocal

if "%~1"=="" (
  echo Usage: %~nx0 ^<RTSP_URL^> [JANUS_RTP_URL]
  echo Example:
  echo   %~nx0 "rtsp://pyuser:PASS@192.168.1.70:554/..."
  echo   %~nx0 "rtsp://pyuser:PASS@192.168.1.70:554/..." "rtp://127.0.0.1:5004?pkt_size=1200"
  exit /b 1
)

set "RTSP_URL=%~1"
set "JANUS_RTP_URL=%~2"
if "%JANUS_RTP_URL%"=="" set "JANUS_RTP_URL=rtp://127.0.0.1:5004?pkt_size=1200"
set "REENCODE=%AICAM_FFMPEG_REENCODE%"

if /I "%REENCODE%"=="1" goto reencode
if /I "%REENCODE%"=="true" goto reencode
if /I "%REENCODE%"=="yes" goto reencode
goto copy

:copy
ffmpeg ^
  -rtsp_transport tcp ^
  -fflags nobuffer ^
  -flags low_delay ^
  -i "%RTSP_URL%" ^
  -an ^
  -c:v copy ^
  -payload_type 96 ^
  -f rtp ^
  "%JANUS_RTP_URL%"
goto end

:reencode
ffmpeg ^
  -rtsp_transport tcp ^
  -fflags nobuffer ^
  -flags low_delay ^
  -i "%RTSP_URL%" ^
  -an ^
  -c:v libx264 ^
  -preset ultrafast ^
  -tune zerolatency ^
  -g 30 ^
  -keyint_min 30 ^
  -bf 0 ^
  -pix_fmt yuv420p ^
  -payload_type 96 ^
  -f rtp ^
  "%JANUS_RTP_URL%"

:end
