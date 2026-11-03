import math
import os
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
try:
    import torch
except ImportError:
    torch = None


def imread_unicode(path):
    """Cross-platform (Windows/macOS/Linux) Unicode-safe image reader for paths with Vietnamese characters."""
    if not os.path.exists(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(path)


def imwrite_unicode(path, img, params=None):
    """Cross-platform (Windows/macOS/Linux) Unicode-safe image writer for paths with Vietnamese characters."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, img, params or [])
        if ok:
            buf.tofile(path)
            return True
    except Exception:
        pass
    return cv2.imwrite(path, img, params or [])


# ==============================================================================
# STEP 0: ANATOMICAL GROUPING OF THE 16 ROIs (GASKIN / GREULICH & PYLE)
# ==============================================================================
ROI_INDICES = list(range(14)) + [15, 16]

ANATOMICAL_GROUPS = {
    "DIP":    {"rois": [0, 3, 6, 9, 12], "name": "Group 1: Distal (DIP)",       "color_rgb": (239, 68, 68),   "hex": "#EF4444"},
    "PIP":    {"rois": [1, 4, 7, 10],    "name": "Group 2: Middle (PIP)",       "color_rgb": (245, 158, 11),  "hex": "#F59E0B"},
    "MCP":    {"rois": [2, 5, 8, 11],    "name": "Group 3: Proximal (MCP)",     "color_rgb": (16, 185, 129),  "hex": "#10B981"},
    "Thumb":  {"rois": [13],             "name": "Group 4: Thumb (MCP1)",       "color_rgb": (168, 85, 247),  "hex": "#A855F7"},
    "Carpal": {"rois": [15],             "name": "Group 5: Carpal Bones",       "color_rgb": (56, 189, 248),  "hex": "#38BDF8"},
    "Wrist":  {"rois": [16],             "name": "Group 6: Radius-Ulna (Wrist)","color_rgb": (244, 114, 182), "hex": "#F472B6"},
}

ROI_TO_GROUP = {}
for g_key, g_info in ANATOMICAL_GROUPS.items():
    for r_idx in g_info["rois"]:
        ROI_TO_GROUP[r_idx] = g_key

GASKIN_STAGE_PRIORS = {
    1: {"DIP": 0.15, "PIP": 0.25, "MCP": 0.25, "Thumb": 0.05, "Carpal": 0.20, "Wrist": 0.10},
    2: {"DIP": 0.25, "PIP": 0.30, "MCP": 0.20, "Thumb": 0.05, "Carpal": 0.10, "Wrist": 0.10},
    3: {"DIP": 0.25, "PIP": 0.25, "MCP": 0.15, "Thumb": 0.20, "Carpal": 0.05, "Wrist": 0.10},
    4: {"DIP": 0.30, "PIP": 0.30, "MCP": 0.15, "Thumb": 0.05, "Carpal": 0.00, "Wrist": 0.20},
    5: {"DIP": 0.00, "PIP": 0.15, "MCP": 0.10, "Thumb": 0.00, "Carpal": 0.00, "Wrist": 0.75},
}

STAGE_DESCRIPTIONS = {
    1: "STAGE 1: Infancy & Toddlerhood (Carpal & Epiphyseal Ossification Onset)",
    2: "STAGE 2: Pre-Puberty (Epiphyseal Width Expansion R_w)",
    3: "STAGE 3: Puberty (Adductor Sesamoid Ossification & Epiphyseal Capping)",
    4: "STAGE 4: Early Adolescence (Phalangeal Physeal Fusion DIP -> MCP -> PIP)",
    5: "STAGE 5: Late Adolescence (Phalanges Saturated -> Distal Radius-Ulna Fusion)",
}


# ==============================================================================
# STEP 1 (MODULE 1): AUTOMATED MORPHOLOGICAL FEATURE EXTRACTION (1A -> 1E)
# ==============================================================================
def _measure_carpal_1a(crop_rgb):
    """[1A] Carpal cavity segmentation, border-touching rejection & moment counting on ROI 15."""
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)

    x_l, x_r, y_t, y_b = 30, 195, 30, 195
    cavity = blur[y_t:y_b, x_l:x_r]
    otsu_t, _ = cv2.threshold(cavity, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    p_t = np.percentile(cavity, 58)
    thr_val = max(float(otsu_t), float(p_t))

    raw_mask = np.zeros_like(gray)
    raw_mask[y_t:y_b, x_l:x_r] = (cavity > thr_val).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    carpal_cnts = []
    total_bone_area = float((cavity > otsu_t).sum())
    for c in cnts:
        area_c = cv2.contourArea(c)
        if area_c < 60:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if by <= y_t + 2 or (by + bh) >= y_b - 2 or bx <= x_l + 1 or (bx + bw) >= x_r - 1:
            continue
        carpal_cnts.append(c)

    occupancy_ratio = total_bone_area / float((x_r - x_l) * (y_b - y_t))
    n_carpal = min(8, max(len(carpal_cnts), int(round(occupancy_ratio * 10.5))))
    return {
        "n_carpal": int(n_carpal),
        "occupancy_ratio": float(occupancy_ratio),
    }


# Quy tắc chọn đoạn thẳng trục xương tham chiếu cho từng vùng ROI (0 -> 13):
# - 0 và 1 vuông góc với 0->1; 2 vuông góc với 1->2
# - 3 và 4 vuông góc với 3->4; 5 vuông góc với 4->5
# - 6 và 7 vuông góc với 6->7; 8 vuông góc với 7->8
# - 9 và 10 vuông góc với 9->10; 11 vuông góc với 10->11
# - 12 và 13 vuông góc với 12->13
ROI_AXIS_SEGMENT = {
    0: (0, 1),   1: (0, 1),   2: (1, 2),
    3: (3, 4),   4: (3, 4),   5: (4, 5),
    6: (6, 7),   7: (6, 7),   8: (7, 8),
    9: (9, 10), 10: (9, 10), 11: (10, 11),
    12: (12, 13), 13: (12, 13),
}


def compute_roi_axis_vectors(valid_bboxes=None, keypoints=None):
    """
    Tính vector đơn vị dọc trục xương u=(ux, uy) và vector vuông góc n=(-uy, ux)
    cho các vùng ROI khớp ngón tay (0 -> 13) dựa trên các đoạn nối keypoint.
    """
    pts = {}
    if keypoints is not None:
        for idx, pt in enumerate(keypoints):
            if len(pt) >= 2 and (len(pt) < 3 or pt[2] > 0):
                pts[int(idx)] = (float(pt[0]), float(pt[1]))
    if valid_bboxes is not None:
        for r_idx, box in valid_bboxes.items():
            idx = int(r_idx)
            if idx not in pts:
                pts[idx] = (0.5 * float(box[0] + box[2]), 0.5 * float(box[1] + box[3]))

    axis_info = {}
    for r_idx in range(14):
        seg_a, seg_b = ROI_AXIS_SEGMENT.get(r_idx, (r_idx, r_idx))
        if seg_a in pts and seg_b in pts and seg_a != seg_b:
            dx = pts[seg_b][0] - pts[seg_a][0]
            dy = pts[seg_b][1] - pts[seg_a][1]
            seg_len = max(1e-6, float(math.hypot(dx, dy)))
            ux, uy = dx / seg_len, dy / seg_len
        else:
            ux, uy = 0.0, 1.0

        nx, ny = -uy, ux
        angle_deg = float(math.degrees(math.atan2(ux, uy)))

        cx_224, cy_224 = 112.0, 112.0
        if valid_bboxes is not None and r_idx in valid_bboxes and r_idx in pts:
            x1, y1, x2, y2 = valid_bboxes[r_idx]
            if (x2 - x1) > 1e-3 and (y2 - y1) > 1e-3:
                cx_224 = (pts[r_idx][0] - float(x1)) / float(x2 - x1) * 224.0
                cy_224 = (pts[r_idx][1] - float(y1)) / float(y2 - y1) * 224.0
                cx_224 = float(np.clip(cx_224, 64.0, 160.0))
                cy_224 = float(np.clip(cy_224, 64.0, 160.0))

        # Xác định hướng phía trong lòng bàn tay (về phía ngón trỏ ROI 11) cho ROI 13 (Xương vừng - 1D)
        palm_side = 1.0
        if r_idx == 13 and 13 in pts and 11 in pts:
            to_palm_x = pts[11][0] - pts[13][0]
            to_palm_y = pts[11][1] - pts[13][1]
            dot_n = to_palm_x * nx + to_palm_y * ny
            palm_side = 1.0 if dot_n >= 0.0 else -1.0
        elif r_idx == 13:
            palm_side = 1.0 if nx <= 0.0 else -1.0

        axis_info[r_idx] = {
            "seg": (seg_a, seg_b),
            "ux": float(ux),
            "uy": float(uy),
            "nx": float(nx),
            "ny": float(ny),
            "angle_deg": angle_deg,
            "cx": cx_224,
            "cy": cy_224,
            "palm_side": palm_side,
        }
    return axis_info


def _measure_joint_1b_1c(crop_rgb, ux=0.0, uy=1.0, cx=112.0, cy=112.0):
    """
    [1B & 1C] Đo tỷ lệ bề rộng đầu xương / hành xương (R_w = W_epi / W_meta) và độ hàn sụn (F_r)
    theo đúng trục nghiêng của từng ngón tay:
      - W_meta: luôn đi qua điểm keypoint (cx, cy) (tức s_meta = 0) và vuông góc với đoạn trục xương u=(ux, uy).
      - W_epi: song song với W_meta (dọc theo phương vuông góc n=(-uy, ux)), nằm tại đỉnh sáng đối diện qua khe sụn.
      - Đường khe sụn (Physis - 1C): song song với W_meta và W_epi tại cực tiểu cường độ sáng s_physis giữa 2 vạch.
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    H, W = blur.shape[:2]

    nx, ny = -uy, ux  # vector đơn vị vuông góc với trục xương

    def sample_pt(s, t):
        px = int(round(cx + s * ux + t * nx))
        py = int(round(cy + s * uy + t * ny))
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))
        return float(blur[py, px])

    # 1. Lấy profile cường độ sáng dọc theo trục xương u (s từ -36 đến +24)
    axial_profile = {}
    for s in range(-36, 25):
        axial_profile[s] = float(np.mean([sample_pt(s, t) for t in range(-12, 13, 2)]))

    # W_meta luôn đi qua điểm keypoint -> s_meta = 0
    s_meta = 0.0

    # Khe sụn tiếp hợp (1C): cực tiểu cường độ sáng phía trên keypoint (s trong [-18, -5])
    s_physis = float(min(range(-18, -5), key=lambda s: axial_profile[s]))

    # W_epi song song với W_meta tại đỉnh sáng tiếp theo phía trên khe sụn
    s_epi_start = max(-34, int(s_physis) - 14)
    s_epi_end = max(-30, int(s_physis) - 3)
    s_epi = float(max(range(s_epi_start, s_epi_end), key=lambda s: axial_profile[s]))

    # Khe khớp dưới keypoint
    s_joint = float(min(range(10, 25), key=lambda s: axial_profile[s]))

    # 2. Quét biên xương dọc theo phương vuông góc n tại s_meta = 0 và s_epi
    def scan_perpendicular_edges(s_center, ds_range=range(-6, 10)):
        prof_t = {}
        for t in range(-72, 73):
            prof_t[t] = float(np.percentile([sample_pt(s_center + ds, t) for ds in ds_range], 80))

        i_bg_left = float(np.percentile([prof_t[t] for t in range(-70, -48)], 25))
        i_peak_left = float(np.percentile([prof_t[t] for t in range(-38, -4)], 85))
        t_thr_left = i_bg_left + 0.45 * max(10.0, i_peak_left - i_bg_left)

        i_bg_right = float(np.percentile([prof_t[t] for t in range(48, 71)], 25))
        i_peak_right = float(np.percentile([prof_t[t] for t in range(4, 39)], 85))
        t_thr_right = i_bg_right + 0.45 * max(10.0, i_peak_right - i_bg_right)

        t_left = -38
        for t in range(-62, -16):
            if prof_t[t] >= t_thr_left:
                t_left = t
                break

        t_right = 38
        for t in range(62, 16, -1):
            if prof_t[t] >= t_thr_right:
                t_right = t
                break

        # Cân bằng đối xứng 2 bên trục xương (t = 0) để tâm đường đo luôn nằm đúng trên trục xanh lá & keypoint
        half_w = 0.5 * (abs(float(t_left)) + abs(float(t_right)))
        t_left = -half_w
        t_right = half_w

        p1 = (cx + s_center * ux + t_left * nx, cy + s_center * uy + t_left * ny)
        p2 = (cx + s_center * ux + t_right * nx, cy + s_center * uy + t_right * ny)
        width = max(12.0, float(t_right - t_left))
        return p1, p2, width, (t_left, t_right)

    meta_p1, meta_p2, w_meta, meta_t = scan_perpendicular_edges(s_meta, ds_range=range(-5, 11))
    epi_p1, epi_p2, w_epi, epi_t = scan_perpendicular_edges(s_epi, ds_range=range(-6, 7))
    rw = float(np.clip(w_epi / max(14.0, w_meta), 0.40, 1.30))

    # Đường khe sụn 1C song song với W_meta & W_epi tại s_physis
    t_phy_l = 0.5 * (meta_t[0] + epi_t[0])
    t_phy_r = 0.5 * (meta_t[1] + epi_t[1])
    physis_p1 = (cx + s_physis * ux + t_phy_l * nx, cy + s_physis * uy + t_phy_l * ny)
    physis_p2 = (cx + s_physis * ux + t_phy_r * nx, cy + s_physis * uy + t_phy_r * ny)

    # Điểm số hàn sụn (Fusion Score F_r - Module 1C) tính dọc theo trục nghiêng của xương
    i_floor = float(np.percentile(blur[:, 40:184], 20)) * 0.65
    i_gap = max(0.0, float(axial_profile[int(round(s_physis))]) - i_floor)
    i_bone = max(
        10.0,
        0.5 * (float(axial_profile[int(round(s_epi))]) + float(axial_profile[int(round(s_meta))])) - i_floor,
    )
    fusion_score = float(np.clip(i_gap / i_bone, 0.0, 1.0))

    y_meta_compat = int(round(cy + s_meta * uy))
    y_physis_compat = int(round(cy + s_physis * uy))
    y_epi_compat = int(round(cy + s_epi * uy))
    y_joint_compat = int(round(cy + s_joint * uy))

    return {
        "rw": rw,
        "w_epi": w_epi,
        "w_meta": w_meta,
        "fusion_score": fusion_score,
        "s_meta": s_meta,
        "s_physis": s_physis,
        "s_epi": s_epi,
        "meta_pts": (meta_p1, meta_p2),
        "epi_pts": (epi_p1, epi_p2),
        "physis_pts": (physis_p1, physis_p2),
        "keypoint_pt": (cx, cy),
        "ux": float(ux),
        "uy": float(uy),
        "nx": float(nx),
        "ny": float(ny),
        "y_meta": y_meta_compat,
        "y_physis": y_physis_compat,
        "y_epi": y_epi_compat,
        "y_joint": y_joint_compat,
        "epi_x": (int(round(min(epi_p1[0], epi_p2[0]))), int(round(max(epi_p1[0], epi_p2[0])))),
        "meta_x": (int(round(min(meta_p1[0], meta_p2[0]))), int(round(max(meta_p1[0], meta_p2[0])))),
    }


def _measure_sesamoid_1d(crop_rgb, ux=0.0, uy=1.0, palm_side=1.0, cx=112.0, cy=112.0, bone_half_w=44.0):
    """
    [1D] Phát hiện xương vừng (Adductor Sesamoid) tại ROI 13 (MCP1 ngón cái)
    theo hệ trục xoay nghiêng song song và vuông góc với đoạn 12 -> 13, nằm ngoài biên xương MCP1.
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    p10 = float(np.percentile(gray, 10))
    p98 = float(np.percentile(gray, 98))
    gray_norm = np.clip((gray - p10) / max(15.0, p98 - p10) * 255.0, 0.0, 255.0)

    g1 = cv2.GaussianBlur(gray_norm, (5, 5), 1.2)
    g2 = cv2.GaussianBlur(gray_norm, (25, 25), 6.0)
    dog = np.maximum(0.0, g1 - g2)
    H, W = gray_norm.shape[:2]

    nx, ny = (-uy * palm_side, ux * palm_side)  # hướng vuông góc về phía bờ trong (phía ngón trỏ)

    # Khung tìm kiếm xương vừng kéo dài dọc xuống bờ bên chỏm xương bàn 1 (tránh chồng lên thân xương)
    s_min, s_max = -12.0, 84.0
    t_min = float(np.clip(bone_half_w + 10.0, 52.0, 66.0))
    t_max = float(min(102.0, t_min + 38.0))

    ses_vals = []
    dog_vals = []
    for s in range(int(s_min), int(s_max) + 1, 2):
        # Dịch nhẹ t_min ra ngoài ở đoạn dưới (s > 35) để tránh chạm rìa chỏm xương bàn 1
        t_min_s = t_min + max(0.0, (s - 35.0) * 0.15)
        for t in range(int(t_min_s), int(t_max) + 1, 2):
            px = int(round(cx + s * ux + t * nx))
            py = int(round(cy + s * uy + t * ny))
            if 2 <= px < W - 2 and 2 <= py < H - 2:
                ses_vals.append(float(gray_norm[py, px]))
                dog_vals.append(float(dog[py, px]))

    if len(ses_vals) >= 10:
        med_ses = float(np.median(ses_vals))
        p92_val = float(np.percentile(ses_vals, 92))
        prominence = float(np.clip((p92_val - med_ses) / 95.0, 0.0, 1.0))
        dog_energy = float(np.percentile(dog_vals, 95) / 38.0)
        ses_score = float(np.clip(0.55 * prominence + 0.45 * min(1.0, dog_energy), 0.0, 1.0))
    else:
        ses_score = 0.0

    # Đỉnh đa giác tìm kiếm xương vừng xoay nghiêng theo trục 12 -> 13 và cắt vát dọc theo mép trái ảnh (x >= 2)
    raw_poly = [
        (cx + s_min * ux + t_min * nx, cy + s_min * uy + t_min * ny),
        (cx + s_min * ux + t_max * nx, cy + s_min * uy + t_max * ny),
        (cx + s_max * ux + t_max * nx, cy + s_max * uy + t_max * ny),
        (cx + s_max * ux + (t_min + 6.0) * nx, cy + s_max * uy + (t_min + 6.0) * ny),
    ]
    clipped_poly = []
    x_bound = 2.0
    for i in range(len(raw_poly)):
        x1, y1 = raw_poly[i]
        x2, y2 = raw_poly[(i + 1) % len(raw_poly)]
        in1, in2 = (x1 >= x_bound), (x2 >= x_bound)
        if in1 and in2:
            clipped_poly.append((x2, y2))
        elif in1 and not in2:
            t_c = (x_bound - x1) / (x2 - x1 - 1e-8)
            clipped_poly.append((x_bound, y1 + t_c * (y2 - y1)))
        elif not in1 and in2:
            t_c = (x_bound - x1) / (x2 - x1 + 1e-8)
            clipped_poly.append((x_bound, y1 + t_c * (y2 - y1)))
            clipped_poly.append((x2, y2))
    ses_pts = [(float(np.clip(px, 2.0, W - 3.0)), float(np.clip(py, 2.0, H - 3.0))) for px, py in (clipped_poly or raw_poly)]

    return {
        "sesamoid_score": ses_score,
        "sesamoid_present": int(ses_score >= 0.52),
        "ses_pts": ses_pts,
    }


def _measure_wrist_1e(crop_rgb):
    """[1E] BoneMask-gated Sobel internal edge magnitude & fusion score on ROI 16."""
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    eq = clahe.apply(gray).astype(np.float32)

    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    bmask = eq > np.percentile(eq, 45)
    sobel_score = float(mag[bmask].mean()) if bmask.sum() > 0 else float(mag.mean())

    blur = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    vp = blur[:, 80:145].mean(axis=1)
    i_floor = float(np.percentile(blur, 20)) * 0.65
    i_gap = max(0.0, float(np.min(vp[75:135])) - i_floor)
    i_bone = max(10.0, float(np.max(vp[65:155])) - i_floor)
    wrist_fusion = float(np.clip(i_gap / i_bone, 0.0, 1.0))

    return {
        "sobel_score": sobel_score,
        "wrist_fusion": wrist_fusion,
    }


def extract_roi_morphology(patient_crops, valid_bboxes=None, keypoints=None):
    """Run Step 1 (1A -> 1E) across all 16 ROI crops (224x224 RGB) with angled finger-axis orientation."""
    morph = {"per_roi": {}}
    rw_list, f_dip_list, f_pip_list, f_mcp_list = [], [], [], []
    axis_map = compute_roi_axis_vectors(valid_bboxes=valid_bboxes, keypoints=keypoints)

    for r_idx in range(14):
        if r_idx in patient_crops:
            ax = axis_map.get(r_idx, {"ux": 0.0, "uy": 1.0, "cx": 112.0, "cy": 112.0, "seg": (r_idx, r_idx), "angle_deg": 0.0})
            res = _measure_joint_1b_1c(
                patient_crops[r_idx],
                ux=ax["ux"],
                uy=ax["uy"],
                cx=ax["cx"],
                cy=ax["cy"],
            )
            res["seg"] = ax["seg"]
            res["angle_deg"] = ax["angle_deg"]
            morph["per_roi"][r_idx] = res
            rw_list.append(res["rw"])
            if r_idx in ANATOMICAL_GROUPS["DIP"]["rois"]:
                f_dip_list.append(res["fusion_score"])
            elif r_idx in ANATOMICAL_GROUPS["PIP"]["rois"]:
                f_pip_list.append(res["fusion_score"])
            elif r_idx in ANATOMICAL_GROUPS["MCP"]["rois"]:
                f_mcp_list.append(res["fusion_score"])

    if 13 in patient_crops:
        ax13 = axis_map.get(13, {"ux": 0.0, "uy": 1.0, "cx": 112.0, "cy": 112.0, "palm_side": 1.0})
        r13_info = morph["per_roi"].get(13, {})
        half_w13 = 0.5 * max(float(r13_info.get("w_meta", 88.0)), float(r13_info.get("w_epi", 88.0)))
        ses_res = _measure_sesamoid_1d(
            patient_crops[13],
            ux=ax13["ux"],
            uy=ax13["uy"],
            palm_side=ax13["palm_side"],
            cx=ax13["cx"],
            cy=ax13["cy"],
            bone_half_w=half_w13,
        )
        morph["per_roi"].setdefault(13, {}).update(ses_res)
    else:
        ses_res = {"sesamoid_score": 0.0, "sesamoid_present": 0, "ses_pts": []}

    if 15 in patient_crops:
        c_res = _measure_carpal_1a(patient_crops[15])
        morph["per_roi"][15] = c_res
    else:
        c_res = {"n_carpal": 5, "occupancy_ratio": 0.45}

    if 16 in patient_crops:
        w_res = _measure_wrist_1e(patient_crops[16])
        morph["per_roi"][16] = w_res
    else:
        w_res = {"sobel_score": 50.0, "wrist_fusion": 0.70}

    morph["summary"] = {
        "n_carpal": c_res["n_carpal"],
        "carpal_occupancy": c_res["occupancy_ratio"],
        "mean_rw": float(np.mean(rw_list)) if rw_list else 0.85,
        "mean_f_dip": float(np.mean(f_dip_list)) if f_dip_list else 0.70,
        "mean_f_pip": float(np.mean(f_pip_list)) if f_pip_list else 0.70,
        "mean_f_mcp": float(np.mean(f_mcp_list)) if f_mcp_list else 0.70,
        "sesamoid_score": ses_res["sesamoid_score"],
        "sesamoid_present": ses_res["sesamoid_present"],
        "wrist_sobel": w_res["sobel_score"],
        "wrist_fusion": w_res["wrist_fusion"],
    }
    return morph


# ==============================================================================
# STEP 2 (MODULE 2): SOFT TOP-K ATLAS RETRIEVAL
# ==============================================================================
def soft_topk_retrieval(vec, db_tensor, db_ages, db_ids=None, k=10, tau=180.0):
    """
    Compute Cosine Similarity against Atlas DB, extract Top-K matches, filter internal outliers,
    and return interpolated soft_age, sigma (uncertainty), and best_id / candidate_ids.
    """
    sim = torch.nn.functional.cosine_similarity(vec.unsqueeze(0), db_tensor)
    k_eff = min(k, sim.size(0))
    top_sims, top_idxs = torch.topk(sim, k=k_eff, largest=True, sorted=True)

    top_ages = db_ages[top_idxs].float()
    top1_age = float(top_ages[0].item())
    tau_sharp = max(float(tau), 180.0)

    k_med = min(7, k_eff)
    med_topk = torch.median(top_ages[:k_med])
    inlier_mask = torch.abs(top_ages - med_topk) <= 24.0
    if inlier_mask.sum() >= 3:
        filt_sims = top_sims[inlier_mask]
        filt_ages = top_ages[inlier_mask]
        logits = (filt_sims - filt_sims.max()) * tau_sharp
        weights = torch.softmax(logits, dim=0)
        inlier_soft = float(torch.sum(weights * filt_ages).item())
        if abs(top1_age - float(med_topk.item())) <= 18.0:
            soft_age = 0.60 * top1_age + 0.40 * inlier_soft
        else:
            soft_age = inlier_soft
    else:
        logits = (top_sims - top_sims.max()) * tau_sharp
        weights = torch.softmax(logits, dim=0)
        soft_age = float(torch.sum(weights * top_ages).item())

    full_logits = (top_sims - top_sims.max()) * 25.0
    full_w = torch.softmax(full_logits, dim=0)
    variance = float(torch.sum(full_w * (top_ages - soft_age) ** 2).item())
    sigma = math.sqrt(max(1.0, variance))

    order_in_k = torch.argsort(torch.abs(top_ages - soft_age)).tolist()
    best_idx = int(top_idxs[order_in_k[0]].item())
    best_id = db_ids[best_idx] if db_ids is not None else best_idx
    candidate_ids = [db_ids[int(top_idxs[j].item())] for j in order_in_k] if db_ids is not None else [best_id]

    return {
        "soft_age": soft_age,
        "top1_age": top1_age,
        "sigma": sigma,
        "best_id": best_id,
        "candidate_ids": candidate_ids,
        "top_sim": float(top_sims[0].item()),
    }


# ==============================================================================
# STEP 3 (MODULE 3): BLIND STAGE CLASSIFIER, SATURATION GATE & DYNAMIC WEIGHTING
# ==============================================================================
def classify_biological_stage_blind(soft_ages, morph_summary, gender):
    """Classify the hand radiograph into Biological Stage 1..5 WITHOUT ground-truth age."""
    vals = [soft_ages[r] for r in ROI_INDICES if r in soft_ages]
    a_init = float(np.median(vals)) if vals else 120.0
    wrist_age = float(soft_ages.get(16, a_init))
    pip_ages = [soft_ages[r] for r in ANATOMICAL_GROUPS["PIP"]["rois"] if r in soft_ages]
    pip_mean = float(np.mean(pip_ages)) if pip_ages else a_init

    is_female = (gender.lower() == "female")
    t_stage1 = 36.0 if is_female else 48.0
    t_stage2 = 108.0 if is_female else 126.0
    t_stage3 = 150.0 if is_female else 168.0
    t_stage4 = 174.0 if is_female else 192.0

    f_dip = morph_summary.get("mean_f_dip", 0.70)
    f_pip = morph_summary.get("mean_f_pip", 0.70)
    sesamoid = morph_summary.get("sesamoid_present", 0)
    mean_rw = morph_summary.get("mean_rw", 0.85)

    if a_init < t_stage1:
        return 1, a_init
    if a_init >= t_stage4 or (
        a_init >= t_stage4 - 8.0 and pip_mean >= t_stage4 - 8.0 and wrist_age >= t_stage4 - 4.0 and f_dip >= 0.80 and f_pip >= 0.80
    ):
        return 5, a_init
    if a_init >= t_stage3 or (a_init >= t_stage3 - 6.0 and f_dip >= 0.82):
        return 4, a_init
    if a_init >= t_stage2 or (sesamoid == 1 and mean_rw >= 0.96 and a_init >= t_stage2 - 6.0):
        return 3, a_init
    return 2, a_init


def blind_stage_and_aggregate(
    soft_ages,
    sigmas,
    morph_feats,
    gender,
    rf_model=None,
    learned_config=None,
    top1_ages=None,
):
    """
    Step 3:
      1. Detect biological Stage 1..5 blindly.
      2. Gate out saturated ROIs and statistical outliers (via MAD & sigma_r).
      3. Combine Stage-Weighted Clinical Aggregation with Calibrated RF/Ensemble.
    """
    morph_summary = morph_feats.get("summary", {}) if morph_feats else {}
    stage, a_init = classify_biological_stage_blind(soft_ages, morph_summary, gender)

    age_list = np.array([soft_ages[r] for r in ROI_INDICES if r in soft_ages], dtype=np.float64)
    mad = float(np.median(np.abs(age_list - a_init))) + 1e-5
    robust_std = max(8.0, 1.4826 * mad)

    group_weights = dict(GASKIN_STAGE_PRIORS[stage])
    if learned_config and "stage_group_weights" in learned_config and stage in (2, 3, 4):
        s_str = str(stage)
        if s_str in learned_config["stage_group_weights"]:
            lw = learned_config["stage_group_weights"][s_str]
            for g_k in group_weights:
                if g_k in lw and group_weights[g_k] > 0:
                    group_weights[g_k] = 0.65 * group_weights[g_k] + 0.35 * float(lw[g_k])

    g_sum = sum(group_weights.values()) + 1e-8
    for g_k in group_weights:
        group_weights[g_k] /= g_sum

    roi_status = {}
    roi_raw_w = {}
    group_ages = {}

    all_sigs = [float(sigmas.get(r, 12.0)) for r in ROI_INDICES]
    med_sig_sq = float(np.median(all_sigs)) ** 2 + 25.0

    wrist_age = float(soft_ages.get(16, a_init))
    for g_k, g_info in ANATOMICAL_GROUPS.items():
        g_w = group_weights.get(g_k, 0.0)
        member_rois = g_info["rois"]
        if g_w <= 1e-4:
            for r in member_rois:
                roi_status[r] = "SATURATED"
                roi_raw_w[r] = 0.0
            group_ages[g_k] = a_init
            continue

        valid_members = []
        member_inv_var = []
        for r in member_rois:
            r_age = float(soft_ages.get(r, a_init))
            r_sig = float(sigmas.get(r, 12.0))
            dev = abs(r_age - a_init)

            is_saturated = False
            if stage >= 4 and g_k in ("DIP", "Carpal", "Thumb") and (a_init - r_age > 16.0) and (wrist_age - r_age > 16.0):
                is_saturated = True

            max_dev = 18.0 if r == 15 else max(22.0, 2.0 * robust_std)
            is_outlier = (dev > max_dev) and (r != 16 or stage < 5)

            if is_saturated:
                roi_status[r] = "SATURATED"
                roi_raw_w[r] = 0.0
            elif is_outlier:
                roi_status[r] = "OUTLIER"
                roi_raw_w[r] = 0.0
            else:
                roi_status[r] = "ACTIVE"
                inv_v = med_sig_sq / (r_sig ** 2 + 25.0)
                valid_members.append((r, r_age))
                member_inv_var.append(inv_v)

        if len(valid_members) == 0:
            group_ages[g_k] = a_init
            continue

        iv_sum = sum(member_inv_var) + 1e-8
        mean_group_conf = float(np.mean(member_inv_var))
        g_age_accum = 0.0
        for (r, r_age), iv in zip(valid_members, member_inv_var):
            norm_iv = iv / iv_sum
            g_age_accum += norm_iv * r_age
            roi_raw_w[r] = g_w * norm_iv * min(1.3, max(0.25, mean_group_conf))
        group_ages[g_k] = g_age_accum

    if sum(roi_raw_w.values()) <= 1e-6:
        sorted_rois = sorted(ROI_INDICES, key=lambda r: abs(soft_ages.get(r, a_init) - a_init))
        for r in sorted_rois[:4]:
            roi_status[r] = "ACTIVE"
            roi_raw_w[r] = 0.25

    total_w = sum(roi_raw_w.values()) + 1e-8
    roi_weights = {r: float(roi_raw_w.get(r, 0.0) / total_w) for r in ROI_INDICES}

    effective_group_weights = {
        g_k: float(sum(roi_weights.get(r, 0.0) for r in g_info["rois"]))
        for g_k, g_info in ANATOMICAL_GROUPS.items()
    }

    raw_clinical_age = sum(roi_weights[r] * float(soft_ages.get(r, a_init)) for r in ROI_INDICES)
    if gender.lower() == "female":
        prior_shift = float(np.clip(0.14 * (a_init - 75.0), -6.0, 11.0))
    else:
        prior_shift = float(np.clip(0.08 * (a_init - 75.0), -5.0, 6.0))
    clinical_age = raw_clinical_age - prior_shift

    esa_months = clinical_age
    if rf_model is not None:
        base_rf_ages = top1_ages if top1_ages is not None else soft_ages
        if getattr(rf_model, "is_debiased", False):
            raw_features = [float(base_rf_ages.get(r, soft_ages.get(r, a_init))) for r in ROI_INDICES]
            rf_pred = float(rf_model.predict(np.array([raw_features]))[0])
        else:
            cleaned_features = []
            for r in ROI_INDICES:
                raw_r_age = float(base_rf_ages.get(r, soft_ages.get(r, a_init)))
                if roi_status.get(r) in ("SATURATED", "OUTLIER") and stage >= 4:
                    cleaned_features.append(raw_clinical_age)
                elif roi_status.get(r) == "OUTLIER" or abs(raw_r_age - a_init) > 24.0:
                    cleaned_features.append(0.50 * raw_clinical_age + 0.50 * float(soft_ages.get(r, a_init)))
                else:
                    cleaned_features.append(raw_r_age)
            rf_pred = float(rf_model.predict(np.array([cleaned_features]))[0]) - 0.70 * prior_shift

        alpha_clinical = 0.60 if stage == 5 else (0.45 if gender.lower() == "female" else 0.50)
        esa_months = alpha_clinical * clinical_age + (1.0 - alpha_clinical) * rf_pred

    return {
        "esa_months": float(esa_months),
        "clinical_age": float(clinical_age),
        "a_init": float(a_init),
        "stage": int(stage),
        "stage_desc": STAGE_DESCRIPTIONS[stage],
        "roi_weights": roi_weights,
        "roi_status": roi_status,
        "group_ages": group_ages,
        "group_weights": effective_group_weights,
    }


# ==============================================================================
# STEP 4: SIMPLIFIED CLINICAL DASHBOARD RENDERER (WINDOWS / MACOS / LINUX)
# ==============================================================================
_FONT_CACHE = {}


def _get_pil_font(size, bold=True):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    from PIL import ImageFont

    candidates = (
        [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/tahomabd.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for p in candidates:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                _FONT_CACHE[key] = f
                return f
            except Exception:
                pass
    f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


def render_clinical_dashboard(
    pat_id,
    gender,
    real_age,
    final_img_bgr,
    valid_bboxes,
    patient_crops,
    atlas_crops,
    soft_ages,
    sigmas,
    morph_feats,
    agg_result,
    save_path="",
    use_fast_pil=True,
):
    """
    Render the Simplified Clinical Dashboard showing:
      - Top 1-line summary: ID | Real Age (Tuổi thật) | Predicted Age (Tuổi dự đoán) | MAE (Độ lệch)
      - Left: 1 large input hand X-ray with 16 ROI boxes drawn
      - Right: 4x4 ROI Grid comparing (Input ROI | Atlas ROI)
    """
    esa = agg_result["esa_months"]
    mae = abs(esa - real_age)

    final_img_rgb = cv2.cvtColor(final_img_bgr, cv2.COLOR_BGR2RGB)
    masked_img = final_img_rgb.copy()
    overlay = final_img_rgb.copy()

    for idx, bbox in valid_bboxes.items():
        g_key = ROI_TO_GROUP.get(idx, "DIP")
        rgb_col = ANATOMICAL_GROUPS[g_key]["color_rgb"]
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), rgb_col, -1)

    cv2.addWeighted(overlay, 0.25, masked_img, 0.75, 0, masked_img)

    for idx, bbox in valid_bboxes.items():
        g_key = ROI_TO_GROUP.get(idx, "DIP")
        rgb_col = ANATOMICAL_GROUPS[g_key]["color_rgb"]
        cv2.rectangle(masked_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), rgb_col, 2)
        cv2.putText(
            masked_img,
            f"{idx}",
            (bbox[0] + 3, bbox[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
        )

    header_text = (
        f"Real Age: {real_age:.1f} mos ({real_age/12.0:.1f} yrs)   |   "
        f"Predicted Age: {esa:.1f} mos ({esa/12.0:.1f} yrs)   |   "
        f"Error (MAE): {mae:.2f} mos"
    )
    err_color = (80, 235, 130) if mae <= 12.0 else ((255, 200, 70) if mae <= 24.0 else (255, 95, 95))

    if use_fast_pil:
        try:
            from PIL import Image, ImageDraw

            W_CANVAS, H_CANVAS = 2360, 1020
            canvas = np.full((H_CANVAS, W_CANVAS, 3), (15, 23, 42), dtype=np.uint8)

            cv2.rectangle(canvas, (20, 16), (W_CANVAS - 20, 82), (30, 41, 59), -1)
            cv2.rectangle(canvas, (20, 16), (W_CANVAS - 20, 82), (51, 65, 85), 2)

            hand_size = 880
            hand_disp = cv2.resize(masked_img, (hand_size, hand_size), interpolation=cv2.INTER_LINEAR)
            canvas[108 : 108 + hand_size, 25 : 25 + hand_size] = hand_disp
            cv2.rectangle(canvas, (25, 108), (25 + hand_size, 108 + hand_size), (71, 85, 105), 2)

            grid_x0, grid_y0 = 935, 108
            cell_w, cell_h = 330, 184
            pad_x, pad_y = 22, 42

            roi_titles = []
            for i, roi_idx in enumerate(ROI_INDICES):
                r_i = i // 4
                c_i = i % 4
                x0 = grid_x0 + c_i * (cell_w + pad_x)
                y0 = grid_y0 + r_i * (cell_h + pad_y) + 26

                pat_c = patient_crops.get(roi_idx, np.zeros((224, 224, 3), dtype=np.uint8))
                atl_c = atlas_crops.get(roi_idx, np.zeros((224, 224, 3), dtype=np.uint8))
                comb = cv2.resize(np.hstack((pat_c, atl_c)), (cell_w, cell_h))
                canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = comb

                g_key = ROI_TO_GROUP.get(roi_idx, "DIP")
                g_rgb = ANATOMICAL_GROUPS[g_key]["color_rgb"]
                r_age = soft_ages.get(roi_idx, 0.0)

                cv2.rectangle(canvas, (x0, y0), (x0 + cell_w, y0 + cell_h), g_rgb, 2)
                cv2.line(canvas, (x0 + cell_w // 2, y0), (x0 + cell_w // 2, y0 + cell_h), (71, 85, 105), 1)
                roi_titles.append((x0, y0 - 22, f"ROI {roi_idx}: Input vs Atlas ({r_age:.0f}m)", (241, 245, 249)))

            pil_img = Image.fromarray(canvas)
            draw = ImageDraw.Draw(pil_img)
            f_h1 = _get_pil_font(24, bold=True)
            f_roi = _get_pil_font(15, bold=True)

            draw.text((38, 34), header_text, fill=err_color, font=f_h1)
            for rx, ry, r_txt, r_col in roi_titles:
                draw.text((rx + 2, ry), r_txt, fill=r_col, font=f_roi)

            out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            imwrite_unicode(save_path, out_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            return
        except Exception:
            pass

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(24, 11), facecolor="#0F172A")
    gs = gridspec.GridSpec(4, 5, width_ratios=[1.8, 1, 1, 1, 1], hspace=0.28, wspace=0.15)

    ax_left = plt.subplot(gs[:, 0])
    ax_left.set_facecolor("#0F172A")
    ax_left.imshow(masked_img)
    ax_left.axis("off")
    ax_left.set_title(header_text, fontsize=14, fontweight="bold", color="#10B981" if mae <= 12 else "#F59E0B", pad=12)

    for i, roi_idx in enumerate(ROI_INDICES):
        row_i = i // 4
        col_i = (i % 4) + 1
        ax_r = plt.subplot(gs[row_i, col_i])
        ax_r.set_facecolor("#1E293B")
        ax_r.set_xticks([])
        ax_r.set_yticks([])

        pat_c = patient_crops.get(roi_idx, np.zeros((224, 224, 3), dtype=np.uint8))
        atl_c = atlas_crops.get(roi_idx, np.zeros((224, 224, 3), dtype=np.uint8))
        comb = np.hstack((pat_c, atl_c))
        ax_r.imshow(comb)

        r_age = soft_ages.get(roi_idx, 0.0)
        ax_r.set_title(f"ROI {roi_idx}: Input vs Atlas ({r_age:.0f}m)", fontsize=10, fontweight="bold", color="#F8FAFC", pad=4)

    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=130)
    plt.close(fig)
