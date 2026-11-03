import os
import glob
import json
import math
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anatomical_branch.clinical_algorithm import (
    imread_unicode,
    extract_roi_morphology,
    soft_topk_retrieval,
    blind_stage_and_aggregate,
)
from anatomical_branch.xai_explainer import (
    extract_vecs_and_spatial_maps,
    generate_vietnamese_xai_summary,
    render_xai_dashboard,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
SAMPLES_DIR = os.path.join(ROOT_DIR, "samples")
XAI_MANUAL_DIR = os.path.join(ROOT_DIR, "outputs", "manual_results")

SAVED_JSON_PATH = os.path.join(XAI_MANUAL_DIR, "manual_annotations.json")
SAVED_CSV_PATH = os.path.join(XAI_MANUAL_DIR, "manual_xai_results.csv")

CLINICAL_CONFIG = {}
cfg_path = os.path.join(WEIGHTS_DIR, "clinical_stage_config.json")
if os.path.exists(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        CLINICAL_CONFIG = json.load(f)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

ROI_INDICES = list(range(14)) + [15, 16]
JOINT_INDICES = list(range(14))
WRIST_INDICES = [15, 16]

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

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_rf_v2_features(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    med = np.median(X, axis=1, keepdims=True)
    X_clamped = np.clip(X, med - 24.0, med + 24.0)
    q25 = np.percentile(X_clamped, 25, axis=1, keepdims=True)
    q75 = np.percentile(X_clamped, 75, axis=1, keepdims=True)
    s = np.sort(X, axis=1)
    tmean_4 = np.mean(s[:, 2:-2], axis=1, keepdims=True)
    tmean_6 = np.mean(s[:, 3:-3], axis=1, keepdims=True)
    wrist_m = np.mean(X_clamped[:, 14:16], axis=1, keepdims=True)
    joint_med = np.median(X_clamped[:, :14], axis=1, keepdims=True)
    return np.hstack([X_clamped, med, tmean_4, tmean_6, q25, q75, wrist_m, joint_med])


def predict_rf_v2_bundle(ens_bundle, X_raw):
    X_feat = build_rf_v2_features(X_raw)
    raw_p = 0.5 * ens_bundle["rf"].predict(X_feat)[0] + 0.5 * ens_bundle["huber"].predict(X_feat)[0]
    cal_p = ens_bundle["slope"] * raw_p + ens_bundle["intercept"]
    return float(np.clip(cal_p, 1.0, 228.0))


class CalibratedRFAdapter:
    def __init__(self, rf_v1, rf_v2_bundle=None, w_v1=0.5, is_debiased=False):
        self.rf_v1 = rf_v1
        self.rf_v2_bundle = rf_v2_bundle
        self.w_v1 = w_v1
        self.is_debiased = is_debiased

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        p1 = float(self.rf_v1.predict(X_arr)[0])
        if self.rf_v2_bundle is None or not isinstance(self.rf_v2_bundle, dict) or self.w_v1 >= 0.999:
            return np.array([p1])
        p2 = predict_rf_v2_bundle(self.rf_v2_bundle, X_arr)
        return np.array([self.w_v1 * p1 + (1.0 - self.w_v1) * p2])


class EncoderResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


print("Loading Encoders, Calibrated Regressor and Vector Atlas DB (No YOLO needed)...")
models = {"male": {}, "female": {}}
db_tensors = {"male": {}, "female": {}}
db_ages = {"male": {}, "female": {}}
db_ids = {"male": {}, "female": {}}
db_id_to_row = {"male": {}, "female": {}}

for gender in ["male", "female"]:
    m_j = EncoderResNet50().to(DEVICE)
    m_j.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, f"encoder_{gender}_joints_ep5.pt"), map_location=DEVICE))
    m_j.eval()
    models[gender]["joints"] = m_j

    m_w = EncoderResNet50().to(DEVICE)
    m_w.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, f"encoder_{gender}_wrists_ep5.pt"), map_location=DEVICE))
    m_w.eval()
    models[gender]["wrists"] = m_w

    rf_v1 = joblib.load(os.path.join(WEIGHTS_DIR, f"{gender}_rf.pkl"))
    rf_v2_path = os.path.join(WEIGHTS_DIR, f"{gender}_rf_v2.pkl")
    rf_v2_bundle = joblib.load(rf_v2_path) if os.path.exists(rf_v2_path) else None

    if gender == "male":
        models[gender]["rf"] = CalibratedRFAdapter(rf_v1, rf_v2_bundle, w_v1=0.0, is_debiased=True)
    else:
        models[gender]["rf"] = CalibratedRFAdapter(rf_v1, rf_v2_bundle, w_v1=1.0, is_debiased=False)

    db_dir = os.path.join(WEIGHTS_DIR, "atlas_db", gender)
    for idx in ROI_INDICES:
        db_path = os.path.join(db_dir, f"roi_{idx}.pt")
        if os.path.exists(db_path):
            roi_db = torch.load(db_path, map_location="cpu")
            vectors, ages, ids = [], [], []
            for db_img_id, db_data in roi_db.items():
                vectors.append(db_data["vector"].float())
                ages.append(float(db_data["age"]))
                ids.append(db_img_id)
            if len(vectors) > 0:
                db_tensors[gender][idx] = torch.stack(vectors)
                db_ages[gender][idx] = torch.tensor(ages, dtype=torch.float32)
                db_ids[gender][idx] = ids
                db_id_to_row[gender][idx] = {cid: row_i for row_i, cid in enumerate(ids)}

visual_bank_path = os.path.join(WEIGHTS_DIR, "atlas_visual_crops.pt")
_visual_bank = torch.load(visual_bank_path, map_location="cpu") if os.path.exists(visual_bank_path) else {"male": {}, "female": {}}
_crop_cache = {}


def get_atlas_crop(gender, roi_idx, target_age):
    a_int = int(round(float(target_age)))
    ckey = (gender, roi_idx, a_int)
    if ckey in _crop_cache:
        return _crop_cache[ckey]
    g_bank = _visual_bank.get(gender, {}).get(roi_idx, {})
    if not g_bank:
        return None
    buf_bytes = g_bank.get(a_int)
    if buf_bytes is None:
        nearest_a = min(g_bank.keys(), key=lambda k: abs(int(k) - a_int))
        buf_bytes = g_bank[nearest_a]
    arr = np.frombuffer(buf_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _crop_cache[ckey] = rgb
    return rgb


def preprocess_manual_keypoints_to_512(orig_img, manual_kpts):
    """Chuẩn hoá ảnh 512x512 và tạo 16 vùng ROI từ 17 điểm keypoint chấm bằng tay (KHÔNG dùng YOLO)."""
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

    bboxes = {}
    half_s = (0.8 * dist_6_7) / 2.0
    for i in range(14):
        px, py = kpts_512[i][0], kpts_512[i][1]
        bboxes[i] = (px - half_s, py - half_s, px + half_s, py + half_s)

    w1, h1 = 0.45 * L, 0.27 * L
    right_edge1 = pt14[0] + 0.05 * L
    bboxes[15] = (right_edge1 - w1, pt15[1] - h1 / 2.0, right_edge1, pt15[1] + h1 / 2.0)

    w2, h2 = 0.40 * L, 0.20 * L
    bboxes[16] = (pt16[0] - w2 / 2.0, pt16[1] - h2 / 2.0, pt16[0] + w2 / 2.0, pt16[1] + h2 / 2.0)

    return final_img, kpts_512, bboxes


def run_xai_only_from_manual_512(final_img, kpts_512, bboxes, case_label, gender, real_age, save_xai_path, orig_img=None):
    """Chạy thuật toán dự đoán tuổi xương + kết xuất trực tiếp ảnh xai.png & ui.png ẩn danh."""
    h_fin, w_fin = final_img.shape[:2]
    patient_crops = {}
    valid_bboxes = {}

    for roi_idx in ROI_INDICES:
        if roi_idx not in bboxes:
            continue
        x1, y1, x2, y2 = bboxes[roi_idx]
        ix1, iy1 = int(max(0, round(x1))), int(max(0, round(y1)))
        ix2, iy2 = int(min(w_fin, round(x2))), int(min(h_fin, round(y2)))

        crop = final_img[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            continue

        patient_crops[roi_idx] = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (224, 224))
        valid_bboxes[roi_idx] = (ix1, iy1, ix2, iy2)

    if len(patient_crops) < 16:
        return None

    morph_feats = extract_roi_morphology(patient_crops, valid_bboxes=bboxes, keypoints=kpts_512)
    soft_ages, top1_ages, sigmas, atlas_crops = {}, {}, {}, {}
    top_sims, vec_map, fmap_map, atlas_vec_map = {}, {}, {}, {}

    with torch.no_grad():
        joint_batch = torch.stack([transform(patient_crops[r]) for r in JOINT_INDICES]).to(DEVICE)
        wrist_batch = torch.stack([transform(patient_crops[r]) for r in WRIST_INDICES]).to(DEVICE)

        joint_vecs, joint_fmaps = extract_vecs_and_spatial_maps(models[gender]["joints"], joint_batch)
        wrist_vecs, wrist_fmaps = extract_vecs_and_spatial_maps(models[gender]["wrists"], wrist_batch)

        for i, r in enumerate(JOINT_INDICES):
            vec_map[r] = joint_vecs[i]
            fmap_map[r] = joint_fmaps[i]
        for i, r in enumerate(WRIST_INDICES):
            vec_map[r] = wrist_vecs[i]
            fmap_map[r] = wrist_fmaps[i]

        for roi_idx in ROI_INDICES:
            vec = vec_map[roi_idx]
            if roi_idx in db_tensors[gender]:
                ret = soft_topk_retrieval(
                    vec,
                    db_tensors[gender][roi_idx],
                    db_ages[gender][roi_idx],
                    db_ids[gender][roi_idx],
                    k=10,
                    tau=180.0,
                )
                soft_ages[roi_idx] = ret["soft_age"]
                top1_ages[roi_idx] = ret["top1_age"]
                sigmas[roi_idx] = ret["sigma"]
                top_sims[roi_idx] = ret.get("top_sim", 0.99)

                best_cid = ret["best_id"]
                if best_cid in db_id_to_row[gender].get(roi_idx, {}):
                    row_i = db_id_to_row[gender][roi_idx][best_cid]
                    atlas_vec_map[roi_idx] = db_tensors[gender][roi_idx][row_i]

                acrop = get_atlas_crop(gender, roi_idx, soft_ages[roi_idx])
                if acrop is not None:
                    atlas_crops[roi_idx] = acrop

    agg_res = blind_stage_and_aggregate(
        soft_ages,
        sigmas,
        morph_feats,
        gender,
        rf_model=models[gender]["rf"],
        learned_config=CLINICAL_CONFIG.get(gender, {}),
        top1_ages=top1_ages,
    )
    pred_age = agg_res["esa_months"]
    mae = abs(pred_age - real_age)

    render_xai_dashboard(
        pat_id=case_label,
        gender=gender,
        real_age=real_age,
        final_img_bgr=final_img,
        valid_bboxes=valid_bboxes,
        patient_crops=patient_crops,
        atlas_crops=atlas_crops,
        fmaps_map=fmap_map,
        query_vecs_map=vec_map,
        atlas_vecs_map=atlas_vec_map,
        top_sims=top_sims,
        soft_ages=soft_ages,
        sigmas=sigmas,
        morph_feats=morph_feats,
        agg_result=agg_res,
        save_path=save_xai_path,
        orig_img_bgr=orig_img,
    )
    xai_lines = generate_vietnamese_xai_summary(gender, agg_res, morph_feats)

    return {
        "gender": gender,
        "stage": int(agg_res["stage"]),
        "real_age": round(float(real_age), 2),
        "predicted_age": round(float(pred_age), 2),
        "mae": round(float(mae), 2),
        "xai_path": save_xai_path,
        "xai_lines": xai_lines,
    }


def save_progress(saved_records):
    with open(SAVED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(saved_records, f, indent=2, ensure_ascii=False)

    rows = []
    for _, rec in saved_records.items():
        rows.append({
            "real_age_months": rec["real_age"],
            "predicted_age_months": rec["predicted_age"],
            "mae_months": rec["mae"],
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(SAVED_CSV_PATH, index=False, encoding="utf-8")


def load_saved_progress():
    if os.path.exists(SAVED_JSON_PATH):
        try:
            with open(SAVED_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Interactive 17-Keypoint Annotation & Direct XAI Export Tool")
    parser.add_argument("--image", type=str, default=None, help="Path to hand X-ray image or sample filename in samples/")
    parser.add_argument("--gender", type=str, default=None, choices=["male", "female"], help="Filter by biological sex")
    parser.add_argument("--real_age", type=float, default=None, help="Real bone age in months (optional)")
    parser.add_argument("--redo", action="store_true", help="Re-annotate previously saved images")
    args = parser.parse_args()

    os.makedirs(XAI_MANUAL_DIR, exist_ok=True)
    sample_meta_csv = os.path.join(SAMPLES_DIR, "sample_metadata.csv")
    meta_dict = {}
    if os.path.exists(sample_meta_csv):
        df_sm = pd.read_csv(sample_meta_csv)
        for _, row in df_sm.iterrows():
            stem = os.path.splitext(str(row["sample_file"]))[0]
            meta_dict[stem] = {
                "gender": str(row["gender"]).strip().lower(),
                "real_age": float(row["real_age_months"]),
            }

    saved_records = load_saved_progress()
    candidates = []

    if args.image is not None and os.path.exists(args.image):
        stem = os.path.splitext(os.path.basename(args.image))[0]
        g = args.gender or meta_dict.get(stem, {}).get("gender", "female")
        ra = args.real_age if args.real_age is not None else meta_dict.get(stem, {}).get("real_age", 84.0)
        candidates.append((stem, g, float(ra), args.image))
    else:
        for img_path in sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.png"))):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            if args.image is not None and args.image not in stem:
                continue
            info = meta_dict.get(stem, {"gender": "female", "real_age": 84.0})
            if args.gender is not None and info["gender"] != args.gender:
                continue
            if (not args.redo) and (args.image is None) and (stem in saved_records):
                continue
            candidates.append((stem, info["gender"], info["real_age"], img_path))

    if len(candidates) == 0:
        print("No remaining images to annotate. Pass --redo to re-annotate.")
        return

    print(f"\nStarting manual 17-keypoint XAI session for {len(candidates)} image(s) -> Output: {XAI_MANUAL_DIR}/")
    clicked_points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_points
        scale = param["scale"]
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) < 17:
                clicked_points.append([x / scale, y / scale])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(clicked_points) > 0:
                clicked_points.pop()

    win_name = "Manual Keypoint XAI Annotation (0 -> 16)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    for idx_img, (case_label, gender, real_age, img_path) in enumerate(candidates):
        orig_img = imread_unicode(img_path)
        if orig_img is None:
            continue

        h, w = orig_img.shape[:2]
        disp_h = 840
        scale = disp_h / float(h)
        disp_w = int(round(w * scale))
        disp_img = cv2.resize(orig_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        prev_kpts = saved_records.get(case_label, {}).get("raw_keypoints", None)
        clicked_points = []
        cv2.setMouseCallback(win_name, mouse_callback, param={"scale": scale})

        skip_image = False
        quit_session = False

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
            info_line1 = f"[{idx_img+1}/{len(candidates)}] ({gender.upper()}, Real: {real_age:.0f}m) | Saved: {len(saved_records)} cases"
            info_line2 = f"Annotating [{cur_idx}/17] -> {next_guide}"
            info_line3 = "Left=Click | Right=Undo | R=Reset | S/ESC=Skip | Q=Quit"
            cv2.putText(temp_img, info_line1, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 120), 2, cv2.LINE_AA)
            cv2.putText(temp_img, info_line2, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 235, 255), 2, cv2.LINE_AA)
            cv2.putText(temp_img, info_line3, (12, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 210, 225), 1, cv2.LINE_AA)

            cv2.imshow(win_name, temp_img)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), ord("Q")):
                quit_session = True
                break
            elif key in (27, ord("s"), ord("S")):
                skip_image = True
                break
            elif key in (ord("r"), ord("R")):
                clicked_points = []
            elif key in (ord("l"), ord("L")) and prev_kpts is not None:
                clicked_points = [[float(p[0]), float(p[1])] for p in prev_kpts[:17]]

            if len(clicked_points) == 17:
                cv2.imshow(win_name, temp_img)
                cv2.waitKey(250)
                break

        if quit_session:
            break
        if skip_image or len(clicked_points) < 17:
            continue

        final_img, kpts_512, bboxes = preprocess_manual_keypoints_to_512(orig_img, clicked_points)
        if final_img is None:
            continue

        save_xai_path = os.path.join(XAI_MANUAL_DIR, case_label, "xai.png")
        res = run_xai_only_from_manual_512(
            final_img=final_img,
            kpts_512=kpts_512,
            bboxes=bboxes,
            case_label=case_label,
            gender=gender,
            real_age=real_age,
            save_xai_path=save_xai_path,
            orig_img=orig_img,
        )
        if res is not None:
            saved_records[case_label] = {
                "gender": gender,
                "stage": res["stage"],
                "real_age": res["real_age"],
                "predicted_age": res["predicted_age"],
                "mae": res["mae"],
                "xai_path": res["xai_path"],
                "raw_keypoints": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in clicked_points],
            }
            save_progress(saved_records)
            print(f"  -> Real={res['real_age']:.1f}m | Pred={res['predicted_age']:.1f}m | MAE={res['mae']:.2f}m -> {res['xai_path']}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
