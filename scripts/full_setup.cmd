@echo off
setlocal

python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

python scripts\install_missing.py --requirements requirements.txt
if errorlevel 1 exit /b 1

if not exist .env (
  copy .env.example .env >nul
)

python scripts\download_datasets.py --profile coverage --max-total-gb 3.0 --max-dataset-gb 1.2 --max-files-per-dataset 10000
if errorlevel 1 exit /b 1

python scripts\build_yolo_dataset.py --config configs\dataset_sources.yaml --out datasets\disaster
if errorlevel 1 exit /b 1

if "%TRAIN_EPOCHS%"=="" set TRAIN_EPOCHS=35
if "%TRAIN_DEVICE%"=="" set TRAIN_DEVICE=auto
if "%TRAIN_WORKERS%"=="" set TRAIN_WORKERS=4
if "%TRAIN_CACHE%"=="" set TRAIN_CACHE=disk
if "%TRAIN_AMP%"=="" set TRAIN_AMP=false

python scripts\train_yolo.py --data datasets\disaster\data.yaml --device %TRAIN_DEVICE% --epochs %TRAIN_EPOCHS% --imgsz 640 --batch 4 --workers %TRAIN_WORKERS% --cache %TRAIN_CACHE% --amp %TRAIN_AMP%
if errorlevel 1 exit /b 1

if "%RUN_APP%"=="" set RUN_APP=1
if "%RUN_APP%"=="0" goto :done

python main.py

:done

endlocal
