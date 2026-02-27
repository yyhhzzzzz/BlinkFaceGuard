# 打包说明（macOS + Windows）

## 关键说明
- Python 桌面程序不能在 macOS 直接产出 Windows `.exe`，反之亦然。
- 需要在对应系统各打包一次，或使用仓库内的 GitHub Actions 自动构建。

## 本地打包

### macOS
```bash
cd /Users/yaoyihan/yyh_project/YoloFace
bash build_mac.sh
```

产物：
- `dist/BlinkFilterGUI.app`

### Windows（PowerShell）
```powershell
cd <你的项目目录>\YoloFace
.\build_windows.ps1
```

产物：
- `dist\BlinkFilterGUI\BlinkFilterGUI.exe`

## 自动跨平台构建（推荐）
- 工作流文件：`.github/workflows/build-app.yml`
- 在 GitHub 仓库中触发 `Build Desktop App` 工作流后，会分别生成：
  - `BlinkFilterGUI-macOS`
  - `BlinkFilterGUI-Windows`

## 运行要求
- 程序默认会读取 `models` 目录下的模型文件。
- 当前打包脚本已把项目内 `models` 文件夹打入产物中。
