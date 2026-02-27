# YoloFace Blink Filter

基于 YOLO 人脸检测 + MediaPipe Face Landmarker 的批量眨眼筛图工具，支持命令行与桌面 GUI（PyQt/PySide）。

## 功能特性
- 批量递归读取图片并检测多人合照
- 通过 EAR（Eye Aspect Ratio）判断闭眼
- 输出带标注的可视化图片
- 生成 `report.csv`、`list_PASS.csv`、`list_FAIL.csv` 等结果文件
- 提供中文 GUI 界面（参数调节、日志、结果表、预览）
- 支持打包为 macOS `.app` 与 Windows `.exe`

## 项目结构
```text
YoloFace/
  blink_filter_yolo_tasks.py      # 核心处理逻辑（CLI + 可复用函数）
  gui_blink_filter.py             # 中文 GUI 入口
  build_mac.sh                    # macOS 打包脚本
  build_windows.ps1               # Windows 打包脚本
  requirements_app.txt            # GUI/打包依赖
  PACKAGING.md                    # 打包说明
  models/                         # 模型文件（本地准备）
  input_photos/                   # 输入示例
  out_blink_filter/               # 输出结果
```

## 环境要求
- Python 3.10~3.12（建议 3.11）
- macOS 或 Windows
- 建议使用虚拟环境

## 安装依赖
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_app.txt
```

Windows（PowerShell）：
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_app.txt
```

## 准备模型
请将以下模型放到 `models/` 目录：
- `face_landmarker.task`（MediaPipe Face Landmarker）
- `yolo_face.pt`（YOLO 人脸检测权重）

默认路径：
- `models/face_landmarker.task`
- `models/yolo_face.pt`

## 命令行使用
macOS/Linux（zsh/bash）：
```bash
python blink_filter_yolo_tasks.py \
  --input_dir input_photos \
  --out_dir out_blink_filter \
  --mp_model_path models/face_landmarker.task \
  --yolo_face_weights models/yolo_face.pt \
  --yolo_conf 0.35 \
  --ear_thresh 0.20 \
  --min_face_px 60 \
  --pad 0.25 \
  --max_faces 40 \
  --resize_long 2400
```

## GUI 使用
```bash
python gui_blink_filter.py
```

界面支持：
- 选择输入/输出目录与模型路径
- 配置阈值参数
- 开始运行与停止
- 查看日志、结果表、标注图预览

## 输出说明
默认输出到 `out_blink_filter/`：
- `annotated/`：标注图
- `report.csv`：总表
- `list_PASS.csv` / `list_FAIL.csv` / `list_UNKNOWN.csv` / `list_ERROR.csv`

## 打包
详见 [PACKAGING.md](./PACKAGING.md)。

快速命令：
- macOS: `bash build_mac.sh`
- Windows: `.\build_windows.ps1`

## 开源与发布建议
- 模型权重通常体积大且可能有授权限制，建议**不要直接提交到仓库**
- 推荐仅保留 `models/.gitkeep`，并在 README 中说明模型下载方式
- 首次发布前请确认你使用的模型与依赖许可证条款

## License
建议按你的需求添加，例如 MIT：
```text
MIT License
```
# YoloFaceEyesDetection
# YoloFaceEyesDetection
