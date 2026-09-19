@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   YouTube to Auto-AVSR - Windows Kurulum Sihirbazi
echo ========================================================
echo.

cd /d "%~dp0\.."

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [HATA] Python bulunamadi!
    echo Lutfen Python 3.11 yukleyin ve kurulum ekraninda "Add python.exe to PATH" secenegini isaretleyin.
    pause
    exit /b 1
)

where ffmpeg >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [UYARI] ffmpeg bulunamadi!
    echo Videolari kesmek ve islemek icin ffmpeg gereklidir.
    echo "winget install Gyan.FFmpeg" komutuyla yukleyebilirsiniz.
)

where git >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [HATA] Git bulunamadi! Lutfen Git yukleyin.
    pause
    exit /b 1
)

echo [1/4] Git LFS kontrol ediliyor...
git lfs install >nul 2>nul

echo [2/4] Sanal ortam (.venv) hazirlaniyor...
if not exist ".venv" (
    python -m venv .venv
)

echo [3/4] Bagimliliklar yukleniyor (pip install -e .)...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e .

echo [4/4] Model agirliklari ve araclar indiriliyor...
python -m yt2avsr setup-external --config configs/default.yaml
python -m yt2avsr setup-retinaface --config configs/default.yaml
python -m yt2avsr setup-whisper --config configs/default.yaml

echo.
echo ========================================================
echo   Kurulum basariyla tamamlandi!
echo.
echo   Komutlari su sekilde calistirabilirsiniz:
echo     - Video eklemek:  scripts\run.bat add "URL"
echo     - Shard calistir: scripts\run.bat modal --shard 0
echo     - Yerelde calis:  scripts\run.bat --shard 0
echo ========================================================
pause
