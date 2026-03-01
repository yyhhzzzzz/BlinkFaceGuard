Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
  throw "Missing virtualenv python: $Py"
}

& $Py -m pip install -r requirements_app.txt

if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist) { Remove-Item -Recurse -Force dist }

$Spec = Join-Path $PSScriptRoot "BlinkFilterGUI.spec"
if (Test-Path $Spec) {
  & $Py -m PyInstaller `
    --clean `
    --noconfirm `
    $Spec
} else {
  & $Py -m PyInstaller `
    --clean `
    --noconfirm `
    --windowed `
    --name BlinkFilterGUI `
    --collect-all ultralytics `
    --collect-all mediapipe `
    --collect-all cv2 `
    --collect-binaries torch `
    --hidden-import pandas `
    --hidden-import tqdm `
    --runtime-hook "runtime_hooks\pyi_rth_torch_path.py" `
    --add-data "models;models" `
    gui_blink_filter.py
}

if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Write-Host "Build finished: dist/BlinkFilterGUI/BlinkFilterGUI.exe"
