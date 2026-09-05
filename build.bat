@echo off
rem Сборка VoiceAI с GPU-ускорением (CUDA).
rem
rem Одной командой настраивает окружение MSVC + CUDA, собирает релиз
rem и кладёт нужные библиотеки CUDA рядом с исполняемым файлом, чтобы
rem программа сразу запускалась без дополнительных действий.
rem
rem Требования:
rem   - Visual Studio 2022 (C++ workload, MSVC)
rem   - NVIDIA CUDA Toolkit 12.8+ (проверено с 13.x)
rem   - видеокарта NVIDIA с драйвером, поддерживающим CUDA
setlocal

set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo Не найдена vcvars64.bat. Установите Visual Studio 2022 с компонентом "Разработка классических приложений на C++".
    exit /b 1
)
call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
    echo Ошибка загрузки vcvars64.
    exit /b 1
)

rem Ищем установленный CUDA Toolkit (любая версия 12.8+).
set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if not exist "%CUDA_DIR%" (
    echo Не найден CUDA Toolkit: %CUDA_DIR%
    echo Установите его, например: winget install --id Nvidia.CUDA
    exit /b 1
)
for /f "delims=" %%d in ('dir /b /ad "%CUDA_DIR%\v*"') do set "CUDA_VERSION=%%d"
set "CUDA_HOME=%CUDA_DIR%\%CUDA_VERSION%"
set "CUDA_PATH=%CUDA_HOME%"

rem Версия "13.3" -> "13_3" для переменной CUDA_PATH_V13_3, которую
rem читает MSBuild-интеграция CUDA при сборке CUDA-кода.
for /f "tokens=1,2 delims=." %%a in ("%CUDA_VERSION:v=%") do set "CUDA_PATH_V%%a_%%b=%CUDA_HOME%"

set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\bin\x64;%PATH%"
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

echo Сборка с CUDA %CUDA_VERSION% (%CUDA_HOME%)
call cargo build --release
if errorlevel 1 exit /b 1

rem Кладём библиотеки CUDA рядом с exe, чтобы приложение запускалось без PATH.
set "OUT=%~dp0target\release"
for %%f in (cublas64_1*.dll cublasLt64_1*.dll cudart64_1*.dll) do (
    if exist "%CUDA_HOME%\bin\x64\%%f" copy /y "%CUDA_HOME%\bin\x64\%%f" "%OUT%" >nul
)

rem Кладём модель Whisper рядом с exe, чтобы приложение работало из любой папки.
if exist "%~dp0ggml-large-v3-turbo.bin" copy /y "%~dp0ggml-large-v3-turbo.bin" "%OUT%" >nul
if exist "%~dp0ggml-large-v3-turbo-q5_0.bin" copy /y "%~dp0ggml-large-v3-turbo-q5_0.bin" "%OUT%" >nul

echo.
echo Готово. Запускайте %OUT%\VoiceAI.exe
exit /b 0