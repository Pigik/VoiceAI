@echo off
rem Downloads Whisper models and checks their SHA-256.
rem
rem If the file is already downloaded and the checksum matches, the download
rem is skipped (cache is reused). A corrupted file is re-downloaded.
rem
rem Primary source is the project's own GitHub Release: such a link never
rem expires while the repo lives. HuggingFace remains the fallback source
rem only for the "zeroth" release, before the model exists in the Release.
rem
rem IMPORTANT: keep this script ASCII-only. cmd.exe on GitHub runners and on
rem Russian Windows decodes UTF-8 bytes according to the console codepage,
rem and non-ASCII characters (e.g. the em dash) break batch parsing with
rem '... was unexpected at this time', failing the whole CI job.
rem
rem Usage:
rem   .\download_models.bat
rem   .\download_models.bat "https://github.com/OWNER/REPO/releases/download/v1.0.0"
setlocal

set "BASE=%~1"
if "%BASE%"=="" set "BASE=https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
set "MAIN=ggml-large-v3-turbo.bin"
set "MAIN_SHA256=1FC70F774D38EB169993AC391EEA357EF47C88757EF72EE5943879B7E8E2BC69"

call :ensure_model "%MAIN%" "%MAIN_SHA256%"
if errorlevel 1 exit /b 1

echo.
echo Done: model downloaded and verified by SHA-256.
exit /b 0

:ensure_model
set "FILE=%~1"
set "EXPECTED=%~2"
if not exist "%FILE%" goto :download
call :sha256_of "%FILE%"
if errorlevel 1 exit /b 1
if /i "%HASH%"=="%EXPECTED%" (
    echo Already present: %FILE% - checksum matches, skipping.
    exit /b 0
)
rem NOTE: keep echo lines outside-if-parens structures label-free: a "(" or
rem ")" inside an echo that lives inside an if (...) block makes cmd treat
rem the text as a nested group and fail parsing with 'unexpected at this time'.
echo File %FILE% is corrupted, SHA-256 mismatch - downloading again.
del /q "%FILE%"

:download
echo Downloading %FILE%...
curl -f -L -C - -o "%FILE%" "%BASE%/%FILE%"
if errorlevel 1 (
    echo ERROR: failed to download %FILE%.
    exit /b 1
)
call :sha256_of "%FILE%"
if errorlevel 1 exit /b 1
if /i not "%HASH%"=="%EXPECTED%" (
    echo ERROR: SHA-256 of %FILE% does not match the expected value.
    echo Expected: %EXPECTED%
    echo Got:      %HASH%
    del /q "%FILE%"
    exit /b 1
)
echo OK: %FILE% downloaded and verified.
exit /b 0

:sha256_of
set "HASH="
for /f "skip=1 tokens=*" %%h in ('certutil -hashfile "%~1" SHA256 2^>nul') do (
    if not defined HASH set "HASH=%%h"
)
if not defined HASH (
    echo ERROR: failed to compute SHA-256 for "%~1".
    exit /b 1
)
rem certutil prints the hash without spaces, but strip them just in case.
set "HASH=%HASH: =%"
exit /b 0