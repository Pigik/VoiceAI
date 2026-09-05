@echo off
rem Скачивает модели Whisper и проверяет их по SHA-256.
rem
rem Если файл уже скачан и его контрольная сумма совпадает — повторное
rem скачивание не выполняется (кеш переиспользуется). При повреждённом файле
rem модель скачивается заново.
rem
rem Использование:  .\download_models.bat
setlocal

set "BASE=https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
set "MAIN=ggml-large-v3-turbo.bin"
set "FALLBACK=ggml-large-v3-turbo-q5_0.bin"
set "MAIN_SHA256=1FC70F774D38EB169993AC391EEA357EF47C88757EF72EE5943879B7E8E2BC69"
set "FALLBACK_SHA256=394221709CD5AD1F40C46E6031CA61BCE88931E6E088C188294C6D5A55FFA7E2"

call :ensure_model "%MAIN%" "%MAIN_SHA256%"
if errorlevel 1 exit /b 1
call :ensure_model "%FALLBACK%" "%FALLBACK_SHA256%"
if errorlevel 1 exit /b 1

echo.
echo Готово: обе модели скачаны и проверены по SHA-256.
exit /b 0

:ensure_model
set "FILE=%~1"
set "EXPECTED=%~2"
call :sha256_of "%FILE%"
if errorlevel 1 exit /b 1
if /i "%HASH%"=="%EXPECTED%" (
    echo Уже есть %FILE% — контрольная сумма совпадает, пропускаем.
    exit /b 0
)
if exist "%FILE%" (
    echo Файл %FILE% повреждён (SHA-256 не совпал) — перекачиваем заново.
    del /q "%FILE%"
)
echo Скачиваем %FILE%...
curl -L -C - -o "%FILE%" "%BASE%/%FILE%"
if errorlevel 1 (
    echo ОШИБКА: не удалось скачать %FILE%.
    exit /b 1
)
call :sha256_of "%FILE%"
if errorlevel 1 exit /b 1
if /i not "%HASH%"=="%EXPECTED%" (
    echo ОШИБКА: SHA-256 файла %FILE% не совпал с ожидаемым.
    echo Ожидалось: %EXPECTED%
    echo Получено:  %HASH%
    del /q "%FILE%"
    exit /b 1
)
echo OK: %FILE% скачан и проверен.
exit /b 0

:sha256_of
set "HASH="
for /f "skip=1 tokens=*" %%h in ('certutil -hashfile "%~1" SHA256 2^>nul') do (
    if not defined HASH set "HASH=%%h"
)
if not defined HASH (
    echo ОШИБКА: не удалось вычислить SHA-256 для "%~1".
    exit /b 1
)
rem certutil печатает хэш без пробелов, но на всякий случай убираем их.
set "HASH=%HASH: =%"
exit /b 0