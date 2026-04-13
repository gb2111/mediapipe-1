@echo off
setlocal

set "REPO_ROOT=%~dp0"
set "EXE=%REPO_ROOT%bazel-bin\mediapipe\examples\desktop\face_mesh\face_mouth_landmarks_async_cpu.exe"
set "CAMERA_ARG="

if not exist "%EXE%" (
  echo Nie znaleziono pliku:
  echo   %EXE%
  echo.
  echo Najpierw zbuduj target:
  echo   bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --define tflite_with_xnnpack=false --define tflite_kernel_use_xnnpack=false //mediapipe/examples/desktop/face_mesh:face_mouth_landmarks_async_cpu
  exit /b 1
)

set "RUNFILES_MANIFEST_ONLY=1"
set "RUNFILES_MANIFEST_FILE=%EXE%.runfiles_manifest"

if not "%~1"=="" (
  echo %~1 | findstr /r "^[0-9][0-9]*$" >nul
  if not errorlevel 1 (
    set "CAMERA_ARG=--camera_id=%~1"
    shift
  )
)

"%EXE%" %CAMERA_ARG% %*

endlocal
