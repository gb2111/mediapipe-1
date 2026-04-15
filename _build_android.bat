@echo off
set BAZEL_SH=C:\Progra~1\Git\bin\bash.exe
set ANDROID_HOME=C:\Users\gregb\AppData\Local\Android\Sdk
set ANDROID_SDK_ROOT=C:\Users\gregb\AppData\Local\Android\Sdk
set ANDROID_NDK_HOME=C:\Users\gregb\AppData\Local\Android\Sdk\ndk\android-ndk-r28b
set PATH=C:\Users\gregb\AppData\Roaming\npm;C:\PROGRA~1\Android\ANDROI~1\jbr\bin;C:\Users\gregb\AppData\Local\Android\Sdk\platform-tools;%PATH%
set LOGFILE=D:\Repos\GitHub\mediapipe2\_build_android.log

echo Checking bazel... > %LOGFILE% 2>&1
where bazel >> %LOGFILE% 2>&1
bazel version >> %LOGFILE% 2>&1

echo Starting build... >> %LOGFILE% 2>&1
bazel build --noenable_platform_specific_config -c opt --config=android_arm64 --java_runtime_version=remotejdk_17 --tool_java_runtime_version=remotejdk_17 "--repo_env=PYTHON_BIN_PATH=D:\Repos\GitHub\mediapipe2\.python312\python.exe" --repo_env=HERMETIC_PYTHON_VERSION=3.12 "--shell_executable=C:/Progra~1/Git/bin/bash.exe" "--copt=-Iexternal/com_google_protobuf/src" "--cxxopt=-Iexternal/com_google_protobuf/src" "--host_copt=/Iexternal/com_google_protobuf/src" "--host_cxxopt=/Iexternal/com_google_protobuf/src" "--host_cxxopt=/std:c++20" --verbose_failures //mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceheadroigpu:faceheadroigpu >> %LOGFILE% 2>&1

echo Build exit code: %ERRORLEVEL% >> %LOGFILE%
