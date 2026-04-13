@echo off
setlocal

set "ROOT=%~dp0"
set "DEST=%ROOT%mediapipe\tasks\testdata\vision\face_landmarker_v2_with_blendshapes.task"
set "URL=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not exist "%ROOT%mediapipe\tasks\testdata\vision" (
  mkdir "%ROOT%mediapipe\tasks\testdata\vision" >nul 2>&1
)

echo Downloading face landmarker task model...
echo   Source: %URL%
echo   Dest:   %DEST%

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { Invoke-WebRequest -Uri '%URL%' -OutFile '%DEST%' -UseBasicParsing; exit 0 } catch { Write-Error $_; exit 1 }"

if errorlevel 1 (
  echo Download failed.
  exit /b 1
)

echo Done.
exit /b 0
