import csv
import os
import sys
from types import SimpleNamespace

from blink_filter_yolo_tasks import run_pipeline

try:
    from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal as Signal
    from PyQt6.QtGui import QColor, QDesktopServices, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QDoubleSpinBox,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QHeaderView,
    )
    QT_BINDING = "PyQt6"
except ImportError:
    from PySide6.QtCore import Qt, QThread, QUrl, Signal
    from PySide6.QtGui import QColor, QDesktopServices, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QDoubleSpinBox,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QHeaderView,
    )
    QT_BINDING = "PySide6"


class BlinkFilterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_preview_path = ""
        self.report_headers = []
        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        self.setWindowTitle(f"眨眼筛图工具 ({QT_BINDING})")
        self.resize(1250, 820)

        self._build_ui()
        self._set_defaults()
        self._set_running(False)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)

        path_box = QGroupBox("输入 / 输出")
        path_layout = QGridLayout(path_box)
        self.input_dir_edit = QLineEdit()
        self.out_dir_edit = QLineEdit()
        self.mp_model_edit = QLineEdit()
        self.yolo_weight_edit = QLineEdit()
        path_layout.addWidget(QLabel("输入目录"), 0, 0)
        path_layout.addLayout(self._path_row(self.input_dir_edit, "dir"), 0, 1)
        path_layout.addWidget(QLabel("输出目录"), 1, 0)
        path_layout.addLayout(self._path_row(self.out_dir_edit, "dir"), 1, 1)
        path_layout.addWidget(QLabel("MediaPipe 模型"), 2, 0)
        path_layout.addLayout(self._path_row(self.mp_model_edit, "file"), 2, 1)
        path_layout.addWidget(QLabel("YOLO 人脸权重"), 3, 0)
        path_layout.addLayout(self._path_row(self.yolo_weight_edit, "file"), 3, 1)
        main_layout.addWidget(path_box)

        param_box = QGroupBox("参数设置")
        param_layout = QFormLayout(param_box)
        self.yolo_conf_spin = self._double_spin(0.01, 1.0, 0.35, 0.01)
        self.yolo_iou_spin = self._double_spin(0.01, 1.0, 0.50, 0.01)
        self.ear_thresh_spin = self._double_spin(0.01, 1.0, 0.20, 0.01)
        self.pad_spin = self._double_spin(0.0, 2.0, 0.25, 0.01)
        self.min_face_spin = self._int_spin(1, 5000, 60)
        self.max_faces_spin = self._int_spin(1, 9999, 40)
        self.resize_long_spin = self._int_spin(0, 10000, 2400)
        param_layout.addRow("YOLO 置信度", self.yolo_conf_spin)
        param_layout.addRow("YOLO IoU", self.yolo_iou_spin)
        param_layout.addRow("EAR 阈值", self.ear_thresh_spin)
        param_layout.addRow("最小人脸像素", self.min_face_spin)
        param_layout.addRow("扩边比例 Pad", self.pad_spin)
        param_layout.addRow("最大人脸数", self.max_faces_spin)
        param_layout.addRow("长边缩放", self.resize_long_spin)
        main_layout.addWidget(param_box)

        action_layout = QHBoxLayout()
        self.run_btn = QPushButton("开始运行")
        self.stop_btn = QPushButton("停止")
        self.open_out_btn = QPushButton("打开输出目录")
        self.reload_btn = QPushButton("重载报告")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMinimumWidth(260)
        action_layout.addWidget(self.run_btn)
        action_layout.addWidget(self.stop_btn)
        action_layout.addWidget(self.open_out_btn)
        action_layout.addWidget(self.reload_btn)
        action_layout.addWidget(self.progress, 1)
        main_layout.addLayout(action_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        left_layout.addWidget(QLabel("结果表 (report.csv)"))
        left_layout.addWidget(self.table, 3)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(QLabel("运行日志"))
        left_layout.addWidget(self.log_text, 2)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("标注图预览"))
        self.preview_label = QLabel("暂无预览")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #999;")
        self.preview_label.setMinimumSize(360, 360)
        right_layout.addWidget(self.preview_label, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter, 1)

        self.run_btn.clicked.connect(self.start_run)
        self.stop_btn.clicked.connect(self.stop_run)
        self.reload_btn.clicked.connect(self.load_report)
        self.open_out_btn.clicked.connect(self.open_output_dir)
        self.table.cellClicked.connect(self.on_row_selected)

    def _set_defaults(self):
        self.input_dir_edit.setText(os.path.join(self.project_dir, "input_photos"))
        self.out_dir_edit.setText(os.path.join(self.project_dir, "out_blink_filter"))
        self.mp_model_edit.setText(os.path.join(self.project_dir, "models", "face_landmarker.task"))
        self.yolo_weight_edit.setText(os.path.join(self.project_dir, "models", "yolo_face.pt"))

    def _path_row(self, line_edit, mode):
        layout = QHBoxLayout()
        btn = QPushButton("浏览")
        btn.setMaximumWidth(90)
        layout.addWidget(line_edit, 1)
        layout.addWidget(btn)
        if mode == "dir":
            btn.clicked.connect(lambda: self.pick_dir(line_edit))
        else:
            btn.clicked.connect(lambda: self.pick_file(line_edit))
        return layout

    def _double_spin(self, lo, hi, val, step):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setSingleStep(step)
        return spin

    def _int_spin(self, lo, hi, val):
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        return spin

    def pick_dir(self, target):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹", target.text() or self.project_dir)
        if path:
            target.setText(path)

    def pick_file(self, target):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", target.text() or self.project_dir)
        if path:
            target.setText(path)

    def start_run(self):
        input_dir = self.input_dir_edit.text().strip()
        out_dir = self.out_dir_edit.text().strip()
        mp_model = self.mp_model_edit.text().strip()
        yolo_weight = self.yolo_weight_edit.text().strip()

        if not input_dir or not os.path.isdir(input_dir):
            self._error("输入目录不存在。")
            return
        if not mp_model or not os.path.isfile(mp_model):
            self._error("MediaPipe 模型文件不存在。")
            return
        if not yolo_weight or not os.path.isfile(yolo_weight):
            self._error("YOLO 人脸权重文件不存在。")
            return

        os.makedirs(out_dir, exist_ok=True)

        self.log_text.clear()
        self.table.setRowCount(0)
        self.preview_label.setText("暂无预览")
        self.preview_label.setPixmap(QPixmap())
        self.current_preview_path = ""
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        args = SimpleNamespace(
            input_dir=input_dir,
            out_dir=out_dir,
            mp_model_path=mp_model,
            yolo_face_weights=yolo_weight,
            yolo_conf=float(self.yolo_conf_spin.value()),
            yolo_iou=float(self.yolo_iou_spin.value()),
            max_faces=int(self.max_faces_spin.value()),
            ear_thresh=float(self.ear_thresh_spin.value()),
            min_face_px=int(self.min_face_spin.value()),
            pad=float(self.pad_spin.value()),
            resize_long=int(self.resize_long_spin.value()),
        )

        self.worker = PipelineWorker(args)
        self.worker.log_signal.connect(self._log)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_worker_finished)
        self.worker.error_signal.connect(self._on_worker_error)

        self._set_running(True)
        self._log("[运行] 开始处理图片...")
        self.worker.start()

    def stop_run(self):
        if self.worker is None:
            return
        if not self.worker.isRunning():
            return
        self._log("[信息] 正在停止...（当前版本会在本轮任务结束后退出）")
        self.worker.requestInterruption()

    def open_output_dir(self):
        out_dir = self.out_dir_edit.text().strip()
        if out_dir and os.path.isdir(out_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(out_dir))
        else:
            self._error("输出目录不存在。")

    def _on_progress(self, done, total):
        total = max(total, 1)
        pct = int(done * 100 / total)
        self.progress.setRange(0, 100)
        self.progress.setValue(pct)

    def _on_worker_finished(self):
        self._log("[完成] 任务执行成功。")
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self._set_running(False)
        self.load_report()

    def _on_worker_error(self, text):
        self._log(f"[错误] {text}")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._set_running(False)
        self.load_report()

    def load_report(self):
        out_dir = self.out_dir_edit.text().strip()
        csv_path = os.path.join(out_dir, "report.csv")
        if not os.path.isfile(csv_path):
            self._log(f"[警告] 未找到报告: {csv_path}")
            return

        rows = []
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for row in reader:
                rows.append(row)

        self.table.clear()
        self.report_headers = headers
        display_headers = {
            "path": "图片路径",
            "status": "状态",
            "reason": "原因",
            "faces": "人脸数",
            "closed_faces": "闭眼人数",
            "min_ear": "最小EAR",
        }
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels([display_headers.get(h, h) for h in headers])
        self.table.setRowCount(len(rows))

        status_colors = {
            "PASS": QColor(220, 248, 220),
            "FAIL": QColor(255, 224, 224),
            "UNKNOWN": QColor(255, 245, 218),
            "ERROR": QColor(235, 235, 235),
        }
        status_text = {
            "PASS": "通过",
            "FAIL": "不通过",
            "UNKNOWN": "未知",
            "ERROR": "错误",
        }
        reason_text = {
            "no_face_detected": "未检测到人脸",
            "imread_failed": "图片读取失败",
        }

        for r, row in enumerate(rows):
            status = row.get("status", "")
            color = status_colors.get(status)
            for c, key in enumerate(headers):
                value = row.get(key, "")
                if key == "status":
                    value = status_text.get(value, value)
                elif key == "reason":
                    value = reason_text.get(value, value)
                item = QTableWidgetItem(value)
                if color is not None:
                    item.setBackground(color)
                self.table.setItem(r, c, item)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._log(f"[信息] 已加载报告: {csv_path}（{len(rows)} 行）")

    def on_row_selected(self, row, _col):
        path_col = self._find_column_by_key("path")
        if path_col < 0:
            return
        item = self.table.item(row, path_col)
        if item is None:
            return
        src_path = item.text().strip()
        if not src_path:
            return

        out_dir = self.out_dir_edit.text().strip()
        ann_path = os.path.join(out_dir, "annotated", os.path.basename(src_path))
        if not os.path.isfile(ann_path):
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(f"未找到标注图：\n{ann_path}")
            self.current_preview_path = ""
            return

        self.current_preview_path = ann_path
        self._refresh_preview()

    def _find_column_by_key(self, key):
        if not self.report_headers:
            return -1
        try:
            return self.report_headers.index(key)
        except ValueError:
            return -1

    def _refresh_preview(self):
        if not self.current_preview_path:
            return
        pix = QPixmap(self.current_preview_path)
        if pix.isNull():
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("预览加载失败。")
            return
        scaled = pix.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview()

    def _set_running(self, running):
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.reload_btn.setEnabled(not running)
        if running:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
        else:
            self.progress.setRange(0, 100)

    def _log(self, msg):
        msg = msg.rstrip("\n")
        if not msg:
            return
        self.log_text.append(msg)

    def _error(self, text):
        QMessageBox.critical(self, "错误", text)
        self._log(f"[错误] {text}")


class PipelineWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, int)
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        try:
            run_pipeline(
                self.args,
                progress_cb=lambda d, t: self.progress_signal.emit(d, t),
                log_cb=lambda m: self.log_signal.emit(str(m)),
            )
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))


def main():
    app = QApplication(sys.argv)
    win = BlinkFilterWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
