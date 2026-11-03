import os
import glob
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from anatomical_branch.clinical_algorithm import (
    ROI_INDICES,
    ANATOMICAL_GROUPS,
    ROI_TO_GROUP,
    STAGE_DESCRIPTIONS,
    imread_unicode,
    imwrite_unicode,
    _get_pil_font,
)

_RAW_SAMPLE_MAP = None


def _get_raw_sample_image(case_key):
    global _RAW_SAMPLE_MAP
    if _RAW_SAMPLE_MAP is None:
        _RAW_SAMPLE_MAP = {}
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        samples_dir = os.path.join(root_dir, "samples")
        if os.path.isdir(samples_dir):
            for p in glob.glob(os.path.join(samples_dir, "**", "*.png"), recursive=True):
                stem = os.path.splitext(os.path.basename(p))[0]
                _RAW_SAMPLE_MAP[stem] = p
    raw_path = _RAW_SAMPLE_MAP.get(str(case_key))
    if raw_path and os.path.exists(raw_path):
        return imread_unicode(raw_path)
    return None


def _pad_to_square_rgb(img_rgb):
    h, w = img_rgb.shape[:2]
    if h == w:
        return img_rgb
    side = max(h, w)
    out = np.zeros((side, side, 3), dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = img_rgb
    return out


def extract_vecs_and_spatial_maps(encoder_model, batch_tensor):
    """
    Pass a batch of ROI tensors [B, 3, 224, 224] through EncoderResNet50:
      - Returns pooled 2048-D vectors [B, 2048]
      - Returns final conv layer4 spatial feature maps [B, 2048, 7, 7]
    """
    x = batch_tensor
    for layer in list(encoder_model.features.children())[:-1]:
        x = layer(x)
    fmap_7x7 = x
    pooled = encoder_model.features[-1](fmap_7x7).view(fmap_7x7.size(0), -1)
    return pooled.cpu(), fmap_7x7.cpu()


def compute_roi_only_gradcam(crop_rgb, fmap_2048x7x7, query_vec, atlas_vec):
    """
    Compute Cosine-Similarity Grad-CAM strictly inside the ROI bone structure:
      - Weights each of the 2048 channels of layer4 by its contribution to the Cosine Similarity
        between the patient ROI vector and the matched Atlas prototype vector.
      - Applies a soft BoneMask so dark background / non-ROI artifacts are 100% suppressed.
    """
    q_norm = F.normalize(query_vec.float().unsqueeze(0), dim=1).squeeze(0)
    a_norm = F.normalize(atlas_vec.float().unsqueeze(0), dim=1).squeeze(0)

    channel_weights = F.relu(q_norm * a_norm)  # [2048]
    cam_7x7 = torch.einsum("k,kij->ij", channel_weights, fmap_2048x7x7.float())
    cam_7x7 = F.relu(cam_7x7).numpy()

    cam_224 = cv2.resize(cam_7x7, (224, 224), interpolation=cv2.INTER_CUBIC)
    if cam_224.max() > cam_224.min() + 1e-6:
        cam_224 = (cam_224 - cam_224.min()) / (cam_224.max() - cam_224.min())
    else:
        cam_224 = np.zeros((224, 224), dtype=np.float32)

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    p25 = float(np.percentile(gray, 25))
    p85 = float(np.percentile(gray, 85))
    bone_gate = np.clip((gray - p25) / max(15.0, p85 - p25), 0.0, 1.0)

    yy, xx = np.mgrid[0:224, 0:224]
    dist_c = np.sqrt(((xx - 112.0) / 96.0) ** 2 + ((yy - 112.0) / 96.0) ** 2)
    center_gate = np.clip(1.15 - 0.55 * dist_c, 0.20, 1.0)

    cam_masked = cam_224 * (0.30 + 0.70 * bone_gate) * center_gate
    if cam_masked.max() > 1e-6:
        cam_masked = cam_masked / cam_masked.max()

    heat_u8 = np.uint8(255.0 * cam_masked)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    alpha = (0.58 * cam_masked)[..., None]
    overlay_rgb = np.clip((1.0 - alpha) * crop_rgb.astype(np.float32) + alpha * heat_rgb.astype(np.float32), 0, 255).astype(np.uint8)
    return overlay_rgb, cam_masked


def draw_morphology_xai_crop(roi_idx, crop_rgb, morph_feats):
    """
    Draw Layer-2 Morphological XAI directly onto the 224x224 ROI crop:
      - ROI 0..13: Epiphyseal width W_e (Cyan), Metaphyseal width W_m (Yellow), Physis line (Magenta)
      - ROI 13: Sesamoid detection box (Orange)
      - ROI 15: Carpal bone contours (Green)
      - ROI 16: Distal radius growth plate fusion band (Pink/Cyan)
    """
    vis = crop_rgb.copy()
    per_roi = morph_feats.get("per_roi", {}).get(roi_idx, {})

    if roi_idx <= 12:
        if "meta_pts" in per_roi and "epi_pts" in per_roi and "physis_pts" in per_roi:
            ux = float(per_roi.get("ux", 0.0))
            uy = float(per_roi.get("uy", 1.0))
            cx, cy = per_roi.get("keypoint_pt", (112.0, 112.0))
            ax_p1 = (int(round(cx - 95 * ux)), int(round(cy - 95 * uy)))
            ax_p2 = (int(round(cx + 95 * ux)), int(round(cy + 95 * uy)))
            cv2.line(vis, ax_p1, ax_p2, (90, 200, 60), 1, cv2.LINE_AA)

            pp1 = (int(round(per_roi["physis_pts"][0][0])), int(round(per_roi["physis_pts"][0][1])))
            pp2 = (int(round(per_roi["physis_pts"][1][0])), int(round(per_roi["physis_pts"][1][1])))
            cv2.line(vis, pp1, pp2, (244, 114, 182), 2, cv2.LINE_AA)

            mp1 = (int(round(per_roi["meta_pts"][0][0])), int(round(per_roi["meta_pts"][0][1])))
            mp2 = (int(round(per_roi["meta_pts"][1][0])), int(round(per_roi["meta_pts"][1][1])))
            cv2.line(vis, mp1, mp2, (250, 204, 21), 2, cv2.LINE_AA)
            cv2.circle(vis, mp1, 3, (250, 204, 21), -1, cv2.LINE_AA)
            cv2.circle(vis, mp2, 3, (250, 204, 21), -1, cv2.LINE_AA)

            ep1 = (int(round(per_roi["epi_pts"][0][0])), int(round(per_roi["epi_pts"][0][1])))
            ep2 = (int(round(per_roi["epi_pts"][1][0])), int(round(per_roi["epi_pts"][1][1])))
            cv2.line(vis, ep1, ep2, (56, 189, 248), 2, cv2.LINE_AA)
            cv2.circle(vis, ep1, 3, (56, 189, 248), -1, cv2.LINE_AA)
            cv2.circle(vis, ep2, 3, (56, 189, 248), -1, cv2.LINE_AA)

            kpt_c = (int(round(cx)), int(round(cy)))
            cv2.circle(vis, kpt_c, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, kpt_c, 2, (239, 68, 68), -1, cv2.LINE_AA)
        else:
            y_epi = int(per_roi.get("y_epi", 96))
            y_physis = int(per_roi.get("y_physis", 82))
            y_meta = int(per_roi.get("y_meta", 70))
            x1_e, x2_e = per_roi.get("epi_x", (80, 144))
            x1_m, x2_m = per_roi.get("meta_x", (75, 149))
            cv2.line(vis, (x1_m, y_physis), (x2_m, y_physis), (244, 114, 182), 2)
            cv2.line(vis, (x1_m, y_meta), (x2_m, y_meta), (250, 204, 21), 2)
            cv2.line(vis, (x1_e, y_epi), (x2_e, y_epi), (56, 189, 248), 2)

    elif roi_idx == 13:
        ses_p = int(per_roi.get("sesamoid_present", 0))
        col_s = (16, 185, 129) if ses_p else (245, 158, 11)
        if "ses_pts" in per_roi and len(per_roi["ses_pts"]) >= 3:
            s_poly = np.array([[int(round(p[0])), int(round(p[1]))] for p in per_roi["ses_pts"]], dtype=np.int32)
            cv2.polylines(vis, [s_poly], isClosed=True, color=col_s, thickness=2, lineType=cv2.LINE_AA)
        else:
            cv2.rectangle(vis, (18, 95), (82, 165), col_s, 2)

    elif roi_idx == 15:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6))
        blur = cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0)
        x_l, x_r, y_t, y_b = 30, 195, 30, 195
        cavity = blur[y_t:y_b, x_l:x_r]
        otsu_t, _ = cv2.threshold(cavity, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_val = max(float(otsu_t), float(np.percentile(cavity, 58)))
        raw_mask = np.zeros_like(gray)
        raw_mask[y_t:y_b, x_l:x_r] = (cavity > thr_val).astype(np.uint8) * 255
        clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.rectangle(vis, (x_l, y_t), (x_r, y_b), (71, 85, 105), 1)
        for c in cnts:
            if cv2.contourArea(c) < 60:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if by <= y_t + 2 or (by + bh) >= y_b - 2 or bx <= x_l + 1 or (bx + bw) >= x_r - 1:
                continue
            cv2.drawContours(vis, [c], -1, (16, 185, 129), 2)

    elif roi_idx == 16:
        cv2.rectangle(vis, (75, 75), (150, 135), (244, 114, 182), 2)
        cv2.line(vis, (75, 105), (150, 105), (56, 189, 248), 2)

    return vis


def generate_english_xai_summary(gender, agg_result, morph_feats):
    """Generate a concise, clinically interpretable English explanation of the AI decision."""
    stage = agg_result["stage"]
    s_desc = STAGE_DESCRIPTIONS.get(stage, f"STAGE {stage}")
    ms = morph_feats.get("summary", {})
    n_carp = ms.get("n_carpal", 0)
    rw = ms.get("mean_rw", 0.85)
    f_dip = ms.get("mean_f_dip", 0.70)
    ses = ms.get("sesamoid_score", 0.0)
    w_fus = ms.get("wrist_fusion", 0.70)

    gw = agg_result.get("group_weights", {})
    top_groups = sorted(gw.items(), key=lambda kv: kv[1], reverse=True)
    top_str = ", ".join(f"{k} ({v*100:.0f}%)" for k, v in top_groups if v >= 0.08)

    statuses = agg_result.get("roi_status", {})
    outliers = [str(r) for r, st in statuses.items() if st == "OUTLIER"]
    saturated = [str(r) for r, st in statuses.items() if st == "SATURATED"]

    gate_parts = []
    if outliers:
        gate_parts.append(f"Gated OUTLIER: ROI [{', '.join(outliers)}]")
    if saturated:
        gate_parts.append(f"Gated SATURATED: ROI [{', '.join(saturated)}]")
    gate_str = " | ".join(gate_parts) if gate_parts else "All 16 anatomical ROIs are valid (ACTIVE)"

    line1 = (
        f"Morphological & Biological Stage ({s_desc}): "
        f"Carpal Count [1A]={n_carp}/8 | Epiphyseal/Metaphyseal Ratio [1B] R_w={rw:.2f} | "
        f"DIP Fusion [1C]={f_dip:.2f} | Sesamoid [1D]={ses:.2f} | Wrist Fusion [1E]={w_fus:.2f}"
    )
    line2 = f"Decision Weight Attribution: Prioritized {top_str}   |   Quality Control: {gate_str}"
    return line1, line2


generate_vietnamese_xai_summary = generate_english_xai_summary


def render_xai_dashboard(
    pat_id,
    gender,
    real_age,
    final_img_bgr,
    valid_bboxes,
    patient_crops,
    atlas_crops,
    fmaps_map,
    query_vecs_map,
    atlas_vecs_map,
    top_sims,
    soft_ages,
    sigmas,
    morph_feats,
    agg_result,
    save_path,
    orig_img_bgr=None,
):
    """
    Render the Anonymized 4-Layer Explainable AI (XAI) Dashboard (2620 x 1240):
      - Displays ONLY Predicted Age, Real Age, and Error (MAE) in headers (no patient ID or name).
      - Strictly focuses Grad-CAM heatmaps ONLY inside the ROI boxes on the hand X-ray.
      - Displays 2-panel ROI explainability: [1. Grad-CAM ROI | 2. Atlas Sim].
      - Displays Group Weight Attribution bar chart.
    """
    esa = agg_result["esa_months"]
    mae = abs(esa - real_age) if real_age is not None else 0.0
    roi_weights = agg_result.get("roi_weights", {})
    roi_status = agg_result.get("roi_status", {})
    group_weights = agg_result.get("group_weights", {})
    group_ages = agg_result.get("group_ages", {})

    hand_rgb = cv2.cvtColor(final_img_bgr, cv2.COLOR_BGR2RGB).copy()
    roi_gradcam_crops = {}

    for roi_idx in ROI_INDICES:
        pat_c = patient_crops.get(roi_idx, np.zeros((224, 224, 3), dtype=np.uint8))

        if roi_idx in fmaps_map and roi_idx in query_vecs_map and roi_idx in atlas_vecs_map:
            gcam_rgb, _ = compute_roi_only_gradcam(
                pat_c, fmaps_map[roi_idx], query_vecs_map[roi_idx], atlas_vecs_map[roi_idx]
            )
        else:
            gcam_rgb = pat_c.copy()
        roi_gradcam_crops[roi_idx] = gcam_rgb

        if roi_idx in valid_bboxes:
            x1, y1, x2, y2 = valid_bboxes[roi_idx]
            bw, bh = x2 - x1, y2 - y1
            if bw > 2 and bh > 2:
                patch_gcam = cv2.resize(gcam_rgb, (bw, bh), interpolation=cv2.INTER_LINEAR)
                hand_rgb[y1:y2, x1:x2] = patch_gcam

    hand_gradcam_clean_rgb = hand_rgb.copy()

    for idx, bbox in valid_bboxes.items():
        g_key = ROI_TO_GROUP.get(idx, "DIP")
        st = roi_status.get(idx, "ACTIVE")
        rgb_col = ANATOMICAL_GROUPS[g_key]["color_rgb"] if st == "ACTIVE" else (148, 163, 184)
        cv2.rectangle(hand_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), rgb_col, 2)
        cv2.putText(
            hand_rgb,
            f"{idx}",
            (bbox[0] + 3, bbox[1] + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
        )

    W_CANVAS, H_CANVAS = 2620, 1240
    canvas = np.full((H_CANVAS, W_CANVAS, 3), (15, 23, 42), dtype=np.uint8)

    cv2.rectangle(canvas, (20, 24), (W_CANVAS - 20, 116), (30, 41, 59), -1)
    cv2.rectangle(canvas, (20, 24), (W_CANVAS - 20, 116), (51, 65, 85), 2)

    hand_size = 760
    hand_disp = cv2.resize(hand_rgb, (hand_size, hand_size), interpolation=cv2.INTER_LINEAR)
    canvas[146 : 146 + hand_size, 24 : 24 + hand_size] = hand_disp
    cv2.rectangle(canvas, (24, 146), (24 + hand_size, 146 + hand_size), (71, 85, 105), 2)

    box_y1, box_y2 = 922, 1220
    cv2.rectangle(canvas, (24, box_y1), (24 + hand_size, box_y2), (30, 41, 59), -1)
    cv2.rectangle(canvas, (24, box_y1), (24 + hand_size, box_y2), (51, 65, 85), 2)

    bar_texts = []
    for gi, (g_k, g_info) in enumerate(ANATOMICAL_GROUPS.items()):
        by = box_y1 + 44 + gi * 41
        gw_val = float(group_weights.get(g_k, 0.0))
        ga_val = float(group_ages.get(g_k, 0.0))
        g_col = g_info["color_rgb"]

        bar_x0 = 260
        max_bar_w = 340
        cur_w = int(round(gw_val * max_bar_w))
        cv2.rectangle(canvas, (bar_x0, by), (bar_x0 + max_bar_w, by + 22), (15, 23, 42), -1)
        if cur_w > 0:
            cv2.rectangle(canvas, (bar_x0, by), (bar_x0 + cur_w, by + 22), g_col, -1)
        cv2.rectangle(canvas, (bar_x0, by), (bar_x0 + max_bar_w, by + 22), (71, 85, 105), 1)
        bar_texts.append((38, by + 2, g_info["name"], g_col))
        bar_texts.append((bar_x0 + max_bar_w + 14, by + 2, f"{gw_val*100:4.1f}%  ({ga_val:5.1f}m)", (241, 245, 249)))

    grid_x0, grid_y0 = 812, 146
    cell_w, cell_h = 426, 216
    pad_x, pad_y = 22, 54

    roi_titles = []
    roi_subtitles = []
    for i, roi_idx in enumerate(ROI_INDICES):
        r_i = i // 4
        c_i = i % 4
        x0 = grid_x0 + c_i * (cell_w + pad_x)
        y0 = grid_y0 + r_i * (cell_h + pad_y) + 38

        g_crop = roi_gradcam_crops[roi_idx]
        a_crop = atlas_crops.get(roi_idx, np.zeros((224, 224, 3), dtype=np.uint8))

        dual_stack = np.hstack((g_crop, a_crop))
        dual_disp = cv2.resize(dual_stack, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = dual_disp

        g_key = ROI_TO_GROUP.get(roi_idx, "DIP")
        st = roi_status.get(roi_idx, "ACTIVE")
        g_rgb = ANATOMICAL_GROUPS[g_key]["color_rgb"] if st == "ACTIVE" else (148, 163, 184)

        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w, y0 + cell_h), g_rgb, 2)
        cv2.line(canvas, (x0 + cell_w // 2, y0), (x0 + cell_w // 2, y0 + cell_h), (71, 85, 105), 1)

        r_age = soft_ages.get(roi_idx, 0.0)
        r_sig = sigmas.get(roi_idx, 0.0)
        r_w = roi_weights.get(roi_idx, 0.0) * 100.0
        r_sim = top_sims.get(roi_idx, 0.99) * 100.0

        st_color = (74, 222, 128) if st == "ACTIVE" else ((251, 191, 36) if st == "SATURATED" else (248, 113, 113))
        roi_titles.append((x0, y0 - 36, f"ROI {roi_idx}: {r_age:.0f}m (±{r_sig:.0f}m) | W={r_w:.1f}% [{st}]", st_color))
        roi_subtitles.append((x0, y0 - 18, f"[1. Grad-CAM ROI]                   [2. Atlas Sim {r_sim:.1f}%]", (148, 163, 184)))

    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    f_h1 = _get_pil_font(25, bold=True)
    f_sec = _get_pil_font(16, bold=True)
    f_roi = _get_pil_font(14, bold=True)
    f_small = _get_pil_font(12, bold=False)

    # Anonymized Header: ONLY Real Age, Predicted Age, and Error (MAE) — NO ID or Patient Name
    if real_age is not None and real_age > 0:
        header_text = (
            f"[XAI DASHBOARD]   "
            f"Real Age: {real_age:.1f} mos ({real_age/12.0:.1f} yrs)   |   "
            f"Predicted Age: {esa:.1f} mos ({esa/12.0:.1f} yrs)   |   "
            f"Error (MAE): {mae:.2f} mos"
        )
        err_color = (80, 235, 130) if mae <= 12.0 else ((255, 200, 70) if mae <= 24.0 else (255, 95, 95))
    else:
        header_text = (
            f"[XAI DASHBOARD]   "
            f"Predicted Age: {esa:.1f} mos ({esa/12.0:.1f} yrs)   |   "
            f"90% Conformal CI: [{max(0.0, esa - 19.2):.1f}, {esa + 19.2:.1f}] mos"
        )
        err_color = (80, 235, 130)

    draw.text((36, 54), header_text, fill=err_color, font=f_h1)

    draw.text((38, box_y1 + 12), "XAI TIER 4 — DECISION WEIGHT ATTRIBUTION BY ANATOMICAL GROUP:", fill=(248, 250, 252), font=f_sec)
    for bx, by, btxt, bcol in bar_texts:
        draw.text((bx, by), btxt, fill=bcol, font=f_roi)

    for rx, ry, r_txt, r_col in roi_titles:
        draw.text((rx + 2, ry), r_txt, fill=r_col, font=f_roi)
    for rx, ry, r_txt, r_col in roi_subtitles:
        draw.text((rx + 2, ry), r_txt, fill=r_col, font=f_small)

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    base_name = os.path.basename(save_path).lower()
    if base_name in ("xai.png", "ui.png", "gradcam.png"):
        case_dir = os.path.dirname(save_path)
    else:
        case_dir = os.path.join(os.path.dirname(save_path), str(pat_id))
    os.makedirs(case_dir, exist_ok=True)
    xai_file_path = os.path.join(case_dir, "xai.png")
    ui_file_path = os.path.join(case_dir, "ui.png")

    imwrite_unicode(xai_file_path, out_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    # Render 2-Panel UI Image (ui.png) without ID or Patient Name
    raw_bgr = orig_img_bgr if orig_img_bgr is not None else _get_raw_sample_image(pat_id)
    if raw_bgr is not None:
        orig_rgb = _pad_to_square_rgb(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
    else:
        orig_rgb = cv2.cvtColor(final_img_bgr, cv2.COLOR_BGR2RGB).copy()

    W_GC, H_GC = 1600, 900
    gc_canvas = np.full((H_GC, W_GC, 3), (15, 23, 42), dtype=np.uint8)

    cv2.rectangle(gc_canvas, (24, 20), (W_GC - 24, 104), (30, 41, 59), -1)
    cv2.rectangle(gc_canvas, (24, 20), (W_GC - 24, 104), (51, 65, 85), 2)

    panel_size = 750
    y_panel = 126
    x_left = 24
    x_right = 826

    orig_disp = cv2.resize(orig_rgb, (panel_size, panel_size), interpolation=cv2.INTER_LINEAR)
    gcam_disp = cv2.resize(hand_gradcam_clean_rgb, (panel_size, panel_size), interpolation=cv2.INTER_LINEAR)

    gc_canvas[y_panel : y_panel + panel_size, x_left : x_left + panel_size] = orig_disp
    gc_canvas[y_panel : y_panel + panel_size, x_right : x_right + panel_size] = gcam_disp

    cv2.rectangle(gc_canvas, (x_left, y_panel), (x_left + panel_size, y_panel + panel_size), (71, 85, 105), 2)
    cv2.rectangle(gc_canvas, (x_right, y_panel), (x_right + panel_size, y_panel + panel_size), (71, 85, 105), 2)

    gc_pil = Image.fromarray(gc_canvas)
    gc_draw = ImageDraw.Draw(gc_pil)
    f_gc_h1 = _get_pil_font(25, bold=True)

    if real_age is not None and real_age > 0:
        ui_header_text = (
            f"Real Age: {real_age:.1f} mos ({real_age/12.0:.1f} yrs)     |     "
            f"Predicted Age: {esa:.1f} mos ({esa/12.0:.1f} yrs)     |     "
            f"Error (MAE): {mae:.2f} mos"
        )
    else:
        ui_header_text = (
            f"Predicted Age: {esa:.1f} mos ({esa/12.0:.1f} yrs)"
        )
    gc_draw.text((W_GC // 2, 62), ui_header_text, fill=(248, 250, 252), font=f_gc_h1, anchor="mm")

    gc_out_bgr = cv2.cvtColor(np.array(gc_pil), cv2.COLOR_RGB2BGR)
    imwrite_unicode(ui_file_path, gc_out_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
