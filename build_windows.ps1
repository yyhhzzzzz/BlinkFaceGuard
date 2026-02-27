Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

python -m pip install -r requirements_app.txt

if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist) { Remove-Item -Recurse -Force dist }

pyinstaller `
  --noconfirm `
  --windowed `
  --name BlinkFilterGUI `
  --collect-all ultralytics `
  --collect-all mediapipe `
  --collect-all cv2 `
  --hidden-import pandas `
  --hidden-import tqdm `
  --add-data "models;models" `
  gui_blink_filter.py

Write-Host "Build finished: dist/BlinkFilterGUI/BlinkFilterGUI.exe"
