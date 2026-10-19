@echo off
rem Packs the built release into a self-contained zip archive.
rem
rem The archive can be copied to another computer and used without the
rem internet: the exe, CUDA libraries and (if present) the model are inside.
rem
rem Usage:
rem   .\build.bat          (once, to build the release)
rem   .\package.bat        (creates dist\VoiceAI-windows-x64.zip)
rem
rem IMPORTANT: keep this script ASCII-only (see build.bat).
setlocal

set "OUT=%~dp0target\release"
if not exist "%OUT%\VoiceAI.exe" (
    echo Build the project first: .\build.bat
    exit /b 1
)

set "STAGE=%~dp0dist\VoiceAI"
if exist "%~dp0dist" rmdir /s /q "%~dp0dist"
mkdir "%STAGE%"

copy /y "%OUT%\VoiceAI.exe" "%STAGE%\" >nul
rem CUDA libraries: whisper-rs does not start without them.
for %%f in ("%OUT%\cublas64_1*.dll" "%OUT%\cublasLt64_1*.dll" "%OUT%\cudart64_1*.dll") do copy /y "%%~f" "%STAGE%\" >nul
rem The model goes into the archive only if it is already in target\release.
if exist "%OUT%\ggml-large-v3-turbo.bin" copy /y "%OUT%\ggml-large-v3-turbo.bin" "%STAGE%\" >nul
rem License and third-party notices travel with the program.
copy /y "%~dp0LICENSE" "%STAGE%\LICENSE.txt" >nul
copy /y "%~dp0THIRD-PARTY.md" "%STAGE%\" >nul

powershell -NoProfile -Command "Compress-Archive -Path '%STAGE%' -DestinationPath '%~dp0dist\VoiceAI-windows-x64.zip' -Force"
if errorlevel 1 exit /b 1

echo.
echo Done: dist\VoiceAI-windows-x64.zip
echo To install on another computer: unzip it into any folder
echo and run VoiceAI.exe - no internet connection is needed.
exit /b 0
