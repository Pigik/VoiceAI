@echo off
rem Builds VoiceAI with GPU acceleration (CUDA) in one command.
rem
rem Sets up the MSVC + CUDA environment, builds the release and puts the
rem CUDA libraries and the Whisper model next to the exe, so the program
rem runs right away.
rem
rem Requirements:
rem   - Visual Studio 2022 (any edition or Build Tools) with the
rem     "Desktop development with C++" workload
rem   - NVIDIA CUDA Toolkit 12.8+ (tested with 13.x)
rem   - LLVM/Clang (libclang.dll for bindgen)
rem
rem IMPORTANT: keep this script ASCII-only. cmd.exe decodes the file with the
rem console code page, and non-ASCII text breaks echo output and parsing.
setlocal

rem Find vcvars64.bat of any installed Visual Studio edition via vswhere.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VCVARS="
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VCVARS set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" goto :no_vs
call "%VCVARS%" >nul 2>&1
if errorlevel 1 goto :vcvars_failed

rem Find the installed CUDA Toolkit (the last folder in the list is used).
set "CUDA_DIR=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA"
if not exist "%CUDA_DIR%" goto :no_cuda
set "CUDA_VERSION="
for /f "delims=" %%d in ('dir /b /ad /on "%CUDA_DIR%\v*"') do set "CUDA_VERSION=%%d"
if not defined CUDA_VERSION goto :no_cuda
set "CUDA_HOME=%CUDA_DIR%\%CUDA_VERSION%"
set "CUDA_PATH=%CUDA_HOME%"

rem Version "v13.3" -> variable CUDA_PATH_V13_3, read by the MSBuild CUDA
rem integration when CMake builds the CUDA code of whisper.cpp.
for /f "tokens=1,2 delims=." %%a in ("%CUDA_VERSION:v=%") do set "CUDA_PATH_V%%a_%%b=%CUDA_HOME%"

set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\bin\x64;%USERPROFILE%\.cargo\bin;%PATH%"

echo Building with CUDA %CUDA_VERSION% (%CUDA_HOME%)
call cargo build --release
if errorlevel 1 exit /b 1

rem build.rs already copies the CUDA DLLs and the model next to the exe;
rem copy them again in case they were added after the previous build.
set "OUT=%~dp0target\release"
for %%f in ("%CUDA_HOME%\bin\x64\cublas64_1*.dll" "%CUDA_HOME%\bin\x64\cublasLt64_1*.dll" "%CUDA_HOME%\bin\x64\cudart64_1*.dll" "%CUDA_HOME%\bin\cublas64_1*.dll" "%CUDA_HOME%\bin\cublasLt64_1*.dll" "%CUDA_HOME%\bin\cudart64_1*.dll") do copy /y "%%~f" "%OUT%" >nul
if exist "%~dp0ggml-large-v3-turbo.bin" copy /y "%~dp0ggml-large-v3-turbo.bin" "%OUT%" >nul

echo.
echo Done. Run %OUT%\VoiceAI.exe
exit /b 0

:no_vs
echo vcvars64.bat not found. Install Visual Studio 2022 (or Build Tools)
echo with the "Desktop development with C++" workload.
exit /b 1

:vcvars_failed
echo Failed to set up the MSVC environment (vcvars64.bat).
exit /b 1

:no_cuda
echo CUDA Toolkit not found in %CUDA_DIR%
echo Install it, for example: winget install --id Nvidia.CUDA
exit /b 1
