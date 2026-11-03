import os
import glob
import json
import math
import argparse
import cv2
import numpy as np
from anatomical_branch.clinical_algorithm import imread_unicode, imwrite_unicode

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(ROOT_DIR, "samples")
SAMPLE_KP_JSON = os.path.join(SAMPLES_DIR, "sample_keypoints.json")
MANUAL_KP_JSON = os.path.join(ROOT_DIR, "outputs", "xai_results", "manual_annotations.json")
OUT_STD_DIR = os.path.join(ROOT_DIR, "outputs", "standardized_512")

FINGER_CHAINS = [
    (0, 1, 2),    # Fifth finger:  0 -> 1 -> 2
    (3, 4, 5),    # Fourth finger: 3 -> 4 -> 5
    (6, 7, 8),    # Middle finger: 6 -> 7 -> 8
    (9, 10, 11),  # Index finger:  9 -> 10 -> 11
    (12, 13),     # Thumb:         12 -> 13
]

KP_GUIDE_NAMES = {
    0: "0: Fifth Finger - Distal (DIP5)",
    1: "1: Fifth Finger - Middle (PIP5)",
    2: "2: Fifth Finger - Proximal (MCP5)",
    3: "3: Fourth Finger - Distal (DIP4)",
    4: "4: Fourth Finger - Middle (PIP4)",
    5: "5: Fourth Finger - Proximal (MCP4)",
    6: "6: Middle Finger - Distal (DIP3)",
    7: "7: Middle Finger - Middle (PIP3)",
    8: "8: Middle Finger - Proximal (MCP3)",
    9: "9: Index Finger - Distal (DIP2)",
    10: "10: Index Finger - Middle (PIP2)",
    11: "11: Index Finger - Proximal (MCP2)",
    12: "12: Thumb - Interphalangeal (IP1)",
    13: "13: Thumb - Metacarpophalangeal (MCP1)",
    14: "14: First Metacarpal Base (CMC1 - anchor)",
    15: "15: Carpal Bone Cavity Center (Carpal)",
    16: "16: Distal Radius-Ulna Center (Wrist)",
}


def load_keypoint_store():
    store = {}
    for p in (SAMPLE_KP_JSON, MANUAL_KP_JSON):
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    store.update(data)
            except Exception:
                pass
    return store


def save_manual_keypoints(case_label, raw_kpts, kpts_512=None):
    os.makedirs(os.path.dirname(MANUAL_KP_JSON), exist_ok=True)
    data = {}
    if os.path.exists(MANUAL_KP_JSON):
        try:
            with open(MANUAL_KP_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    entry = data.get(case_label, {})
    entry["raw_keypoints"] = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in raw_kpts[:17]]
    if kpts_512 is not None:
        entry["keypoints_512"] = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in kpts_512[:17]]
    data[case_label] = entry
    with open(MANUAL_KP_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def annotate_17_keypoints_interactive(orig_img, case_label="scan", gender="", real_age=None, prev_kpts=None):
    """Interactive OpenCV GUI for manually annotating 17 anatomical hand keypoints (0 -> 16)."""
    h, w = orig_img.shape[:2]
    disp_h = 840
    scale = disp_h / float(h)
    disp_w = int(round(w * scale))
    disp_img = cv2.resize(orig_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    clicked_points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_points
        sc = param["scale"]
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) < 17:
                clicked_points.append([x / sc, y / sc])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(clicked_points) > 0:
                clicked_points.pop()

    win_name = "Manual 17-Keypoint Annotation (0 -> 16)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, mouse_callback, param={"scale": scale})

    while True:
        temp_img = disp_img.copy()
        for chain in FINGER_CHAINS:
            for i in range(len(chain) - 1):
                ia, ib = chain[i], chain[i + 1]
                if ia < len(clicked_points) and ib < len(clicked_points):
                    xa, ya = int(round(clicked_points[ia][0] * scale)), int(round(clicked_points[ia][1] * scale))
                    xb, yb = int(round(clicked_points[ib][0] * scale)), int(round(clicked_points[ib][1] * scale))
                    cv2.line(temp_img, (xa, ya), (xb, yb), (60, 235, 100), 2, cv2.LINE_AA)

        for i, pt in enumerate(clicked_points):
            cx, cy = int(round(pt[0] * scale)), int(round(pt[1] * scale))
            is_indep = i in (14, 15, 16)
            fill_c = (0, 165, 255) if is_indep else (40, 40, 245)
            txt_c = (80, 225, 255) if is_indep else (120, 255, 140)
            cv2.circle(temp_img, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(temp_img, (cx, cy), 4, fill_c, -1, cv2.LINE_AA)
            cv2.putText(temp_img, str(i), (cx + 8, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(temp_img, str(i), (cx + 8, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_c, 2, cv2.LINE_AA)

        cv2.rectangle(temp_img, (0, 0), (disp_w, 88), (20, 25, 35), -1)
        cur_idx = len(clicked_points)
        next_guide = KP_GUIDE_NAMES.get(cur_idx, "Completed 17/17 keypoints!")
        age_str = f", Real: {real_age:.0f}m" if real_age is not None else ""
        info_line1 = f"Case: {case_label} ({gender.upper()}{age_str})"
        info_line2 = f"Annotating [{cur_idx}/17] -> {next_guide}"
        info_line3 = "Left=Click | Right=Undo | R=Reset | L=Load Prev | S/ESC=Skip | Q=Quit"
        cv2.putText(temp_img, info_line1, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(temp_img, info_line2, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 235, 255), 2, cv2.LINE_AA)
        cv2.putText(temp_img, info_line3, (12, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 210, 225), 1, cv2.LINE_AA)

        cv2.imshow(win_name, temp_img)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ord("Q"), 27, ord("s"), ord("S")):
            cv2.destroyWindow(win_name)
            return None
        elif key in (ord("r"), ord("R")):
            clicked_points = []
        elif key in (ord("l"), ord("L")) and prev_kpts is not None:
            clicked_points = [[float(p[0]), float(p[1])] for p in prev_kpts[:17]]

        if len(clicked_points) == 17:
            cv2.imshow(win_name, temp_img)
            cv2.waitKey(250)
            cv2.destroyWindow(win_name)
            return clicked_points


def standardize_to_valid_512(orig_img, manual_kpts):
    """
    Standardize a hand radiograph to 512x512 RGB and extract 16 anatomical ROI bounding boxes
    directly from 17 manually annotated keypoints (0 -> 16):
      1. Crop excessive empty detector margins around smaller hands while preserving manual keypoint coordinates.
      2. Rotate image so KP6 -> KP8 (middle finger axis) is vertical on the centerline.
      3. Apply CLAHE contrast enhancement in LAB space + aspect-ratio preserving Letterbox to 512x512.
      4. Compute 16 anatomical ROI bounding boxes (squares 0..13 + wrists 15, 16).
    """
    if manual_kpts is None or len(manual_kpts) < 17:
        return None, None, None

    kpts = np.array([[float(pt[0]), float(pt[1])] for pt in manual_kpts[:17]], dtype=np.float64)
    h, w = orig_img.shape[:2]

    L_raw = math.hypot(kpts[6][0] - kpts[16][0], kpts[6][1] - kpts[16][1])
    est_L_512 = L_raw * (512.0 / max(w, h))
    work_img = orig_img
    if L_raw > 50 and est_L_512 < 265.0:
        target_span = L_raw * (512.0 / 292.0)
        cx_m = float(np.mean(kpts[:, 0]))
        cy_m = float(np.mean(kpts[:, 1]))
        half_span = target_span / 2.0
        x1 = max(0, int(round(cx_m - half_span)))
        x2 = min(w, int(round(cx_m + half_span)))
        y1 = max(0, int(round(cy_m - half_span)))
        y2 = min(h, int(round(cy_m + half_span)))
        if (x2 - x1) > L_raw * 1.15 and (y2 - y1) > L_raw * 1.15:
            work_img = orig_img[y1:y2, x1:x2].copy()
            kpts[:, 0] -= x1
            kpts[:, 1] -= y1

    x6, y6 = float(kpts[6][0]), float(kpts[6][1])
    x8, y8 = float(kpts[8][0]), float(kpts[8][1])
    dx, dy = x6 - x8, y6 - y8
    rot_angle_deg = math.degrees(math.atan2(dy, dx) + math.pi / 2.0)

    wh, ww = work_img.shape[:2]
    cx, cy = (x6 + x8) / 2.0, (y6 + y8) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), rot_angle_deg, 1.0)

    corners = np.array([[0, 0], [ww, 0], [ww, wh], [0, wh]], dtype=np.float32)
    corners_ones = np.hstack([corners, np.ones((4, 1))])
    rotated_corners = M.dot(corners_ones.T).T
    min_x, min_y = np.min(rotated_corners, axis=0)
    max_x, max_y = np.max(rotated_corners, axis=0)

    half_w = max(cx - min_x, max_x - cx)
    new_w = int(math.ceil(half_w * 2))
    new_h = int(math.ceil(max_y - min_y))

    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += -min_y

    rotated_img = cv2.warpAffine(work_img, M, (new_w, new_h), borderValue=(0, 0, 0))

    lab = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2LAB)
    l_channel, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    enhanced_img = cv2.cvtColor(cv2.merge((cl, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

    target_size = 512
    scale = target_size / max(new_w, new_h)
    new_w_scaled = int(new_w * scale)
    new_h_scaled = int(new_h * scale)
    resized = cv2.resize(enhanced_img, (new_w_scaled, new_h_scaled), interpolation=cv2.INTER_AREA)

    pad_top = (target_size - new_h_scaled) // 2
    pad_bottom = target_size - new_h_scaled - pad_top
    pad_left = (target_size - new_w_scaled) // 2
    pad_right = target_size - new_w_scaled - pad_left

    final_img = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    kpts_512 = []
    for i in range(17):
        pt = M.dot(np.array([kpts[i][0], kpts[i][1], 1.0]))
        fx = float(pt[0] * scale + pad_left)
        fy = float(pt[1] * scale + pad_top)
        kpts_512.append([fx, fy, 2])

    pt6, pt7, pt14, pt15, pt16 = kpts_512[6], kpts_512[7], kpts_512[14], kpts_512[15], kpts_512[16]
    dist_6_7 = math.hypot(pt6[0] - pt7[0], pt6[1] - pt7[1])
    L = math.hypot(pt6[0] - pt16[0], pt6[1] - pt16[1])
    if dist_6_7 < 5.0 or L < 20.0:
        return None, None, None

    squares = []
    half_s = (0.8 * dist_6_7) / 2.0
    for i in range(14):
        px, py = kpts_512[i][0], kpts_512[i][1]
        squares.append({
            "point_index": i,
            "bbox": [px - half_s, py - half_s, px + half_s, py + half_s],
        })

    w1, h1 = 0.45 * L, 0.27 * L
    right_edge1 = pt14[0] + 0.05 * L
    squares.append({
        "point_index": 15,
        "bbox": [right_edge1 - w1, pt15[1] - h1 / 2.0, right_edge1, pt15[1] + h1 / 2.0],
    })

    w2, h2 = 0.40 * L, 0.20 * L
    squares.append({
        "point_index": 16,
        "bbox": [pt16[0] - w2 / 2.0, pt16[1] - h2 / 2.0, pt16[0] + w2 / 2.0, pt16[1] + h2 / 2.0],
    })

    return final_img, kpts_512, squares


def preprocess_single_raw_image(orig_img, img_path=None, gender="female", real_age=None, force_interactive=False):
    """
    Standardize a single raw radiograph using manual 17-keypoint annotations:
      - Uses stored manual keypoints if available (unless force_interactive=True).
      - Otherwise launches the interactive 17-keypoint GUI so the user clicks keypoints 0 -> 16.
    """
    store = load_keypoint_store()
    case_label = os.path.splitext(os.path.basename(img_path))[0] if img_path else "scan"
    prev_kpts = store.get(case_label, {}).get("raw_keypoints")

    if prev_kpts is None and img_path is None:
        h, w = orig_img.shape[:2]
        for k, v in store.items():
            cand_p = os.path.join(SAMPLES_DIR, f"{k}.png")
            if os.path.exists(cand_p):
                cim = imread_unicode(cand_p)
                if cim is not None and cim.shape[:2] == (h, w) and np.array_equal(cim, orig_img):
                    case_label = k
                    prev_kpts = v.get("raw_keypoints")
                    break

    if prev_kpts is not None and not force_interactive:
        manual_kpts = prev_kpts
    else:
        manual_kpts = annotate_17_keypoints_interactive(
            orig_img, case_label=case_label, gender=gender, real_age=real_age, prev_kpts=prev_kpts
        )
        if manual_kpts is None:
            return None, None, None

    final_img, kpts_512, squares = standardize_to_valid_512(orig_img, manual_kpts)
    if final_img is not None and (force_interactive or prev_kpts is None):
        save_manual_keypoints(case_label, manual_kpts, kpts_512)
    return final_img, kpts_512, squares


def main():
    parser = argparse.ArgumentParser(description="Manual 17-Keypoint Standardization (512x512 RGB + 16 ROIs) for Hand Radiographs")
    parser.add_argument("--input_dir", type=str, default=SAMPLES_DIR, help="Directory of input hand X-ray images (default: samples/)")
    parser.add_argument("--output_dir", type=str, default=OUT_STD_DIR, help="Directory to save standardized 512x512 images and JSON metadata")
    parser.add_argument("--redo", action="store_true", help="Re-annotate 17 keypoints interactively even if saved keypoints exist")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True))
    image_paths = sorted(image_paths)

    annotations = {}
    for img_path in image_paths:
        orig_img = imread_unicode(img_path)
        if orig_img is None:
            continue
        final_img, kpts_512, squares = preprocess_single_raw_image(orig_img, img_path=img_path, force_interactive=args.redo)
        if final_img is None:
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_img_path = os.path.join(args.output_dir, f"{stem}.png")
        imwrite_unicode(out_img_path, final_img)
        annotations[stem] = {
            "keypoints": kpts_512,
            "squares": squares,
            "std_path": out_img_path,
        }

    json_out = os.path.join(args.output_dir, "standardized_annotations.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)
    print(f"Standardized {len(annotations)} images -> {args.output_dir}/")


if __name__ == "__main__":
    main()
