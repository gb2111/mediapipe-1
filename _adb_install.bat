@echo off
set ADB=C:\Users\gregb\AppData\Local\Android\Sdk\platform-tools\adb.exe

echo === Devices === > _adb_out.txt 2>&1
%ADB% devices >> _adb_out.txt 2>&1

echo. >> _adb_out.txt
echo === Installing APK === >> _adb_out.txt
%ADB% install -r bazel-bin\mediapipe\examples\android\src\java\com\google\mediapipe\apps\faceheadroigpu\faceheadroigpu.apk >> _adb_out.txt 2>&1

echo. >> _adb_out.txt
echo === Done === >> _adb_out.txt
