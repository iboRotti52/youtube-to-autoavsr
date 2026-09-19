@echo off
setlocal

cd /d "%~dp0\.."
set PYTHONPATH=%CD%\src;%PYTHONPATH%

if not exist ".venv\Scripts\python.exe" (
    echo [HATA] Sanal ortam (.venv) bulunamadi!
    echo Once scripts\setup_once.bat calistirarak kurulumu tamamlayin.
    exit /b 1
)

call .venv\Scripts\activate.bat

if "%~1"=="" (
    python -m yt2avsr process-both-sources --config configs/default.yaml
) else (
    set "CMD=%~1"
    if "%CMD%"=="add" goto run_direct
    if "%CMD%"=="modal" goto run_direct
    if "%CMD%"=="push-data" goto run_direct
    if "%CMD%"=="pull-data" goto run_direct
    if "%CMD%"=="sync-processed" goto run_direct
    if "%CMD%"=="dedup-sources" goto run_direct
    if "%CMD%"=="inspect" goto run_direct
    if "%CMD%"=="manifest" goto run_direct
    if "%CMD%"=="process-local" goto run_direct
    if "%CMD%"=="process-sources" goto run_direct
    if "%CMD%"=="process-both-sources" goto run_direct
    
    :: Diger tum bayraklar (ornegin --shard 0 veya --url ...)
    python -m yt2avsr process-both-sources --config configs/default.yaml %*
    goto end

:run_direct
    python -m yt2avsr %*

:end
)
