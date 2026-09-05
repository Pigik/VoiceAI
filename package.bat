@echo off
rem Упаковывает собранный релиз в автономный zip-архив.
rem
rem Получившийся архив можно скопировать на другой компьютер и запускать
rem без интернета: все библиотеки и (если есть) модели уже внутри.
rem
rem Использование:
rem   .\build.bat          (один раз, чтобы собрать релиз)
rem   .\package.bat        (создаёт dist\VoiceAI-windows-x64.zip)
setlocal

set "OUT=%~dp0target\release"
if not exist "%OUT%\VoiceAI.exe" (
    echo Сначала соберите проект: .\build.bat
    exit /b 1
)

set "STAGE=%~dp0dist\VoiceAI"
if exist "%~dp0dist" rmdir /s /q "%~dp0dist"
mkdir "%STAGE%"

copy /y "%OUT%\VoiceAI.exe" "%STAGE%\" >nul
rem Библиотеки CUDA — без них whisper-rs не запустится.
for %%f in (cublas64_1*.dll cublasLt64_1*.dll cudart64_1*.dll) do (
    if exist "%OUT%\%%f" copy /y "%OUT%\%%f" "%STAGE%\" >nul
)
rem Модели кладём в архив, только если они уже есть в target\release.
if exist "%OUT%\ggml-large-v3-turbo.bin" copy /y "%OUT%\ggml-large-v3-turbo.bin" "%STAGE%\" >nul
if exist "%OUT%\ggml-large-v3-turbo-q5_0.bin" copy /y "%OUT%\ggml-large-v3-turbo-q5_0.bin" "%STAGE%\" >nul

powershell -NoProfile -Command "Compress-Archive -Path '%STAGE%' -DestinationPath '%~dp0dist\VoiceAI-windows-x64.zip' -Force"
if errorlevel 1 exit /b 1

echo.
echo Готово: dist\VoiceAI-windows-x64.zip
echo Установка на другом компьютере без интернета: распакуйте zip
echo в любую папку и запустите VoiceAI.exe — сеть не нужна.
exit /b 0