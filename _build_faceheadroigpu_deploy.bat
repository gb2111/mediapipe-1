@echo on
set BAZEL_SH=C:\Progra~1\Git\bin\bash.exe
set ANDROID_HOME=C:\Users\gregb\AppData\Local\Android\Sdk
set ANDROID_SDK_ROOT=C:\Users\gregb\AppData\Local\Android\Sdk
set ANDROID_NDK_HOME=C:\Users\gregb\AppData\Local\Android\Sdk\ndk\android-ndk-r28b
set PATH=C:\PROGRA~1\Android\ANDROI~1\jbr\bin;C:\Users\gregb\AppData\Local\Android\Sdk\platform-tools;%PATH%

bazel build --noenable_platform_specific_config -c opt --config=android_arm64 --java_runtime_version=remotejdk_17 --tool_java_runtime_version=remotejdk_17 --repo_env=PYTHON_BIN_PATH=D:\Repos\GitHub\mediapipe2\.python312\python.exe --repo_env=HERMETIC_PYTHON_VERSION=3.12 --worker_extra_flag=Desugar=--jvm_flag=-Xmx1g --shell_executable=C:/Progra~1/Git/bin/bash.exe --verbose_failures //mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceheadroigpu:faceheadroigpu

if %ERRORLEVEL% NEQ 0 (
    echo BUILD FAILED
    exit /b %ERRORLEVEL%
)

echo BUILD OK - deploying...
adb install -r --no-streaming bazel-bin\mediapipe\examples\android\src\java\com\google\mediapipe\apps\faceheadroigpu\faceheadroigpu.apk
