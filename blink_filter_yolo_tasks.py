import os
import math
import argparse
import sys
import pandas as pd
from tqdm import tqdm

_DLL_DIR_HANDLES = []


def _prepare_windows_native_runtime():
    if os.name != "nt":
        return

    # Avoid OpenMP duplicate runtime init failure in mixed native stacks.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    dll_dirs = []
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        dll_dirs.append(os.path.join(meipass, "torch", "lib"))

    exe_dir = os.path.dirname(getattr(sys, "executable", ""))
    if exe_dir:
        dll_dirs.append(os.path.join(exe_dir, "_internal", "torch", "lib"))
        dll_dirs.append(os.path.join(exe_dir, "Lib", "site-packages", "torch", "lib"))

    project_dir = os.path.dirname(os.path.abspath(__file__))
    dll_dirs.append(os.path.join(project_dir, ".venv", "Lib", "site-packages", "torch", "lib"))

    for dll_dir in dll_dirs:
        if not dll_dir or not os.path.isdir(dll_dir):
            continue
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            _DLL_DIR_HANDLES.append(os.add_dll_directory(dll_dir))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def list_images(input_dir: str):
    paths = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(root, fn))
    paths.sort()
    return paths


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def ear_from_landmarks(pts):
    """
    pts: list of 6 (x,y) points for one eye:
        [p1(left corner), p2(upper), p3(upper), p4(right corner), p5(lower), p6(lower)]
    EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|)
    """
    p1, p2, p3, p4, p5, p6 = pts
    denom = 2.0 * dist(p1, p4)
    if denom < 1e-6:
        return 0.0
    return (dist(p2, p6) + dist(p3, p5)) / denom


def extract_eye_points(face_landmarks, w, h, idxs):
    # face_landmarks: list of NormalizedLandmark (x,y in [0,1])
    pts = []
    for i in idxs:
        lm = face_landmarks[i]
        pts.append((lm.x * w, lm.y * h))
    return pts


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="包含照片的文件夹（可递归）")
    ap.add_argument("--out_dir", default="out_blink_filter", help="输出目录")
    ap.add_argument("--mp_model_path", default="models/face_landmarker.task", help="MediaPipe FaceLandmarker 模型路径")
    ap.add_argument("--yolo_face_weights", default="models/yolo_face.pt", help="YOLO 人脸检测权重路径")
    ap.add_argument("--yolo_conf", type=float, default=0.35, help="YOLO 置信度阈值")
    ap.add_argument("--yolo_iou", type=float, default=0.5, help="YOLO NMS IoU 阈值")
    ap.add_argument("--max_faces", type=int, default=40, help="每张图最多处理的人脸数（合照可调）")
    ap.add_argument("--ear_thresh", type=float, default=0.20, help="EAR 阈值：低于此值判定闭眼（0.18~0.22 常用）")
    ap.add_argument("--min_face_px", type=int, default=60, help="过滤太小的人脸（框短边像素阈值）")
    ap.add_argument("--pad", type=float, default=0.25, help="裁剪脸框时的扩边比例（合照建议 0.2~0.35）")
    ap.add_argument("--resize_long", type=int, default=2400, help="推理前把图片长边缩放到该值（0 表示不缩放）")
    return ap


def run_pipeline(args, progress_cb=None, log_cb=print):
    _prepare_windows_native_runtime()
    # Delay heavy native imports to reduce startup-time DLL init conflicts.
    from ultralytics import YOLO
    import cv2
    import mediapipe as mp

    if not os.path.exists(args.mp_model_path):
        raise FileNotFoundError(f"找不到 mediapipe 模型文件: {args.mp_model_path}")
    if not os.path.exists(args.yolo_face_weights):
        raise FileNotFoundError(f"找不到 YOLO 人脸权重: {args.yolo_face_weights}")

    os.makedirs(args.out_dir, exist_ok=True)
    ann_dir = os.path.join(args.out_dir, "annotated")
    os.makedirs(ann_dir, exist_ok=True)

    # 1) YOLO（GPU）做人脸检测
    yolo = YOLO(args.yolo_face_weights)

    # 2) MediaPipe Tasks FaceLandmarker（0.10.x）
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    mp_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.mp_model_path),
        running_mode=RunningMode.IMAGE,
        num_faces=1,  # 每个裁剪脸图只需要 1 张脸
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    # 眼睛 6 点 EAR（稳定组合）
    LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

    img_paths = list_images(args.input_dir)
    if not img_paths:
        log_cb("未找到图片。请检查 input_dir。")
        return

    rows = []

    with FaceLandmarker.create_from_options(mp_options) as landmarker:
        total = len(img_paths)
        use_tqdm = (progress_cb is None) and (getattr(sys, "stderr", None) is not None)
        iterable = tqdm(img_paths, desc="Processing") if use_tqdm else img_paths
        for idx, p in enumerate(iterable, start=1):
            if progress_cb:
                progress_cb(idx - 1, total)
            bgr0 = cv2.imread(p)
            if bgr0 is None:
                rows.append({"path": p, "status": "ERROR", "reason": "imread_failed",
                             "faces": 0, "closed_faces": 0, "min_ear": ""})
                continue

            h0, w0 = bgr0.shape[:2]

            # 可选缩放：加速 YOLO + 更稳（合照大图很有用）
            bgr = bgr0
            if args.resize_long and max(h0, w0) > args.resize_long:
                scale = args.resize_long / float(max(h0, w0))
                new_w = int(round(w0 * scale))
                new_h = int(round(h0 * scale))
                bgr = cv2.resize(bgr0, (new_w, new_h), interpolation=cv2.INTER_AREA)

            h, w = bgr.shape[:2]

            # YOLO 检测脸框
            res = yolo.predict(source=bgr, conf=args.yolo_conf, iou=args.yolo_iou, verbose=False)
            boxes = []
            for r in res:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    x1 = clamp(x1, 0, w - 1)
                    y1 = clamp(y1, 0, h - 1)
                    x2 = clamp(x2, 0, w - 1)
                    y2 = clamp(y2, 0, h - 1)
                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue
                    if min(bw, bh) < args.min_face_px:
                        continue
                    boxes.append((x1, y1, x2, y2))

            if not boxes:
                vis = bgr.copy()
                cv2.putText(vis, "NO FACE DETECTED (YOLO)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(ann_dir, os.path.basename(p)), vis)
                rows.append({"path": p, "status": "UNKNOWN", "reason": "no_face_detected",
                             "faces": 0, "closed_faces": 0, "min_ear": ""})
                continue

            # 合照：限制最多处理的人脸数（按框面积从大到小）
            boxes.sort(key=lambda bb: (bb[2]-bb[0])*(bb[3]-bb[1]), reverse=True)
            boxes = boxes[:args.max_faces]

            closed_count = 0
            min_ear = 999.0
            per_face = []  # (box, ear, closed)  ear=None表示关键点失败

            for (x1, y1, x2, y2) in boxes:
                bw, bh = x2 - x1, y2 - y1

                # 裁剪扩边，避免眼睛贴边导致关键点不稳
                pad_x = int(bw * args.pad)
                pad_y = int(bh * args.pad)
                cx1 = clamp(x1 - pad_x, 0, w - 1)
                cy1 = clamp(y1 - pad_y, 0, h - 1)
                cx2 = clamp(x2 + pad_x, 0, w - 1)
                cy2 = clamp(y2 + pad_y, 0, h - 1)

                crop = bgr[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    per_face.append(((x1, y1, x2, y2), None, None))
                    continue

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                lm_res = landmarker.detect(mp_image)
                if not lm_res.face_landmarks:
                    per_face.append(((x1, y1, x2, y2), None, None))
                    continue

                face_lms = lm_res.face_landmarks[0]
                ch, cw = crop.shape[:2]

                left_pts = extract_eye_points(face_lms, cw, ch, LEFT_EYE_IDXS)
                right_pts = extract_eye_points(face_lms, cw, ch, RIGHT_EYE_IDXS)

                ear = min(ear_from_landmarks(left_pts), ear_from_landmarks(right_pts))
                min_ear = min(min_ear, ear)

                is_closed = ear < args.ear_thresh
                if is_closed:
                    closed_count += 1

                per_face.append(((x1, y1, x2, y2), ear, is_closed))

            # 策略 A：任何人闭眼 -> FAIL
            status = "FAIL" if closed_count > 0 else "PASS"
            min_ear_val = "" if min_ear >= 900 else float(min_ear)

            # 可视化：每张脸标注 OPEN/CLOSED/LM_FAIL
            vis = bgr.copy()
            for (x1, y1, x2, y2), ear, is_closed in per_face:
                if ear is None:
                    color = (0, 165, 255)  # orange: landmark failed
                    label = "LM_FAIL"
                else:
                    if is_closed:
                        color = (0, 0, 255)
                        label = f"CLOSED {ear:.3f}"
                    else:
                        color = (0, 200, 0)
                        label = f"OPEN {ear:.3f}"

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(vis, f"{status}  closed_faces={closed_count}  minEAR={min_ear_val if min_ear_val!='' else 'NA'}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 200, 0) if status == "PASS" else (0, 0, 255), 2)

            cv2.imwrite(os.path.join(ann_dir, os.path.basename(p)), vis)

            rows.append({
                "path": p,
                "status": status,
                "reason": "",
                "faces": len(boxes),
                "closed_faces": closed_count,
                "min_ear": min_ear_val
            })
            if progress_cb:
                progress_cb(idx, total)

    df = pd.DataFrame(rows)

    # 排序：PASS优先 + 眼睛越开(minEAR越大)越靠前
    df_sorted = df.copy()
    df_sorted["min_ear_num"] = pd.to_numeric(df_sorted["min_ear"], errors="coerce")
    order = {"PASS": 0, "FAIL": 1, "UNKNOWN": 2, "ERROR": 3}
    df_sorted["status_ord"] = df_sorted["status"].map(order).fillna(9).astype(int)
    df_sorted = df_sorted.sort_values(by=["status_ord", "min_ear_num"], ascending=[True, False])
    df_sorted.drop(columns=["min_ear_num", "status_ord"], inplace=True)

    csv_path = os.path.join(args.out_dir, "report.csv")
    df_sorted.to_csv(csv_path, index=False, encoding="utf-8-sig")

    for k in ["PASS", "FAIL", "UNKNOWN", "ERROR"]:
        sub = df_sorted[df_sorted["status"] == k]
        sub.to_csv(os.path.join(args.out_dir, f"list_{k}.csv"), index=False, encoding="utf-8-sig")

    log_cb(f"Done. Output: {args.out_dir}")
    log_cb(f"- Annotated images: {ann_dir}")
    log_cb(f"- CSV report: {csv_path}")


def main():
    args = build_arg_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
