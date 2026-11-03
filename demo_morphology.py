import os
import argparse
import cv2
import numpy as np

from anatomical_branch.clinical_algorithm import (
    ANATOMICAL_GROUPS,
    ROI_TO_GROUP,
    ROI_INDICES,
    extract_roi_morphology,
    imread_unicode,
)
from preprocess_hospital import preprocess_single_raw_image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(ROOT_DIR, "samples")
OUT_DEMO_DIR = os.path.join(ROOT_DIR, "outputs", "xai_results")
os.makedirs(OUT_DEMO_DIR, exist_ok=True)

FINGER_CHAINS = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (9, 10, 11),
    (12, 13),
]


def render_demo_module_1bcd(sample_name="sample_male_01.png", scale=2, force_interactive=False):
    img_path = sample_name if os.path.exists(sample_name) else os.path.join(SAMPLES_DIR, sample_name)
    orig_img = imread_unicode(img_path)
    if orig_img is None:
        raise FileNotFoundError(f"Sample image not found: {img_path}")

    img_bgr, kpts, sq_list = preprocess_single_raw_image(orig_img, img_path=img_path, force_interactive=force_interactive)
    if img_bgr is None:
        raise RuntimeError("Manual 17-keypoint annotation (0 -> 16) was cancelled or incomplete.")

    H, W = img_bgr.shape[:2]
    vis = cv2.resize(img_bgr, (W * scale, H * scale), interpolation=cv2.INTER_CUBIC)
    squares = {int(sq["point_index"]): tuple(sq["bbox"]) for sq in sq_list}

    patient_crops = {}
    for r_idx in ROI_INDICES:
        if r_idx in squares:
            x1, y1, x2, y2 = squares[r_idx]
            crop_bgr = img_bgr[max(0, int(round(y1))) : min(H, int(round(y2))), max(0, int(round(x1))) : min(W, int(round(x2)))]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            patient_crops[r_idx] = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_CUBIC)

    morph = extract_roi_morphology(patient_crops, valid_bboxes=squares, keypoints=kpts)

    for r_idx, bbox in squares.items():
        x1, y1, x2, y2 = [int(round(v * scale)) for v in bbox]
        g_key = ROI_TO_GROUP.get(r_idx, "Carpal")
        rgb = ANATOMICAL_GROUPS[g_key]["color_rgb"]
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(vis, (x1, y1), (x2, y2), bgr, 2, cv2.LINE_AA)

    for chain in FINGER_CHAINS:
        for i in range(len(chain) - 1):
            idx_a, idx_b = chain[i], chain[i + 1]
            xa, ya = int(round(kpts[idx_a][0] * scale)), int(round(kpts[idx_a][1] * scale))
            xb, yb = int(round(kpts[idx_b][0] * scale)), int(round(kpts[idx_b][1] * scale))
            cv2.line(vis, (xa, ya), (xb, yb), (60, 235, 100), 2, cv2.LINE_AA)

    roi_cards = []
    for r_idx in range(14):
        x1, y1, x2, y2 = squares[r_idx]
        res = morph["per_roi"][r_idx]
        card = cv2.cvtColor(patient_crops[r_idx], cv2.COLOR_RGB2BGR)

        cx_224, cy_224 = res["keypoint_pt"]
        ux, uy = res["ux"], res["uy"]
        seg_a, seg_b = res["seg"]
        deg = res["angle_deg"]

        ax_p1 = (int(round(cx_224 - 95 * ux)), int(round(cy_224 - 95 * uy)))
        ax_p2 = (int(round(cx_224 + 95 * ux)), int(round(cy_224 + 95 * uy)))
        cv2.line(card, ax_p1, ax_p2, (60, 200, 90), 1, cv2.LINE_AA)

        pp1 = (int(round(res["physis_pts"][0][0])), int(round(res["physis_pts"][0][1])))
        pp2 = (int(round(res["physis_pts"][1][0])), int(round(res["physis_pts"][1][1])))
        cv2.line(card, pp1, pp2, (182, 114, 244), 2, cv2.LINE_AA)

        mp1 = (int(round(res["meta_pts"][0][0])), int(round(res["meta_pts"][0][1])))
        mp2 = (int(round(res["meta_pts"][1][0])), int(round(res["meta_pts"][1][1])))
        cv2.line(card, mp1, mp2, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.circle(card, mp1, 4, (0, 215, 255), -1, cv2.LINE_AA)
        cv2.circle(card, mp2, 4, (0, 215, 255), -1, cv2.LINE_AA)

        ep1 = (int(round(res["epi_pts"][0][0])), int(round(res["epi_pts"][0][1])))
        ep2 = (int(round(res["epi_pts"][1][0])), int(round(res["epi_pts"][1][1])))
        cv2.line(card, ep1, ep2, (255, 215, 0), 2, cv2.LINE_AA)
        cv2.circle(card, ep1, 4, (255, 215, 0), -1, cv2.LINE_AA)
        cv2.circle(card, ep2, 4, (255, 215, 0), -1, cv2.LINE_AA)

        if r_idx == 13 and "ses_pts" in res and len(res["ses_pts"]) >= 3:
            s_poly = np.array([[int(round(p[0])), int(round(p[1]))] for p in res["ses_pts"]], dtype=np.int32)
            ses_col = (120, 240, 80) if res.get("sesamoid_present", 0) else (0, 165, 255)
            cv2.polylines(card, [s_poly], isClosed=True, color=ses_col, thickness=2, lineType=cv2.LINE_AA)

        kpt_c = (int(round(cx_224)), int(round(cy_224)))
        cv2.circle(card, kpt_c, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(card, kpt_c, 3, (30, 30, 245), -1, cv2.LINE_AA)

        cv2.rectangle(card, (0, 0), (224, 30), (20, 25, 35), -1)
        title_str = f"ROI {r_idx} (ortho {seg_a}->{seg_b}) | {deg:+.1f}d"
        cv2.putText(card, title_str, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (240, 245, 255), 1, cv2.LINE_AA)

        cv2.rectangle(card, (0, 192), (224, 224), (20, 25, 35), -1)
        if r_idx == 13:
            foot_str = f"1B:Rw={res['rw']:.2f} | 1C:Fr={res['fusion_score']:.2f} | 1D:Ses={res['sesamoid_score']:.2f}"
        else:
            foot_str = f"1B: Rw={res['rw']:.2f} ({res['w_epi']:.0f}/{res['w_meta']:.0f}) | 1C: Fr={res['fusion_score']:.2f}"
        cv2.putText(card, foot_str, (5, 213), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 240, 255), 1, cv2.LINE_AA)
        roi_cards.append((r_idx, card))

    cols, rows = 5, 3
    pad = 12
    cell_w, cell_h = 224, 224
    header_h = 68
    grid_w = cols * cell_w + (cols + 1) * pad
    grid_h = rows * cell_h + (rows + 1) * pad
    hand_disp_size = grid_h
    hand_disp = cv2.resize(vis, (hand_disp_size, hand_disp_size), interpolation=cv2.INTER_AREA)

    total_w = pad + hand_disp_size + pad + grid_w
    total_h = header_h + grid_h + pad
    canvas = np.full((total_h, total_w, 3), (15, 18, 26), dtype=np.uint8)

    ms = morph["summary"]
    cv2.putText(
        canvas,
        "DEMO MODULE 1B - 1C - 1D (ORTHOGONAL FINGER-AXIS MORPHOLOGICAL MEASUREMENT)",
        (pad, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 250, 255),
        2,
        cv2.LINE_AA,
    )
    summary_line = (
        f"1B Mean Rw: {ms['mean_rw']:.2f} (Yellow=W_meta, Cyan=W_epi)   |   "
        f"1C Fusion (Pink): DIP={ms['mean_f_dip']:.2f}, PIP={ms['mean_f_pip']:.2f}, MCP={ms['mean_f_mcp']:.2f}   |   "
        f"1D Sesamoid (ROI 13): score={ms['sesamoid_score']:.2f}"
    )
    cv2.putText(canvas, summary_line, (pad, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 235, 255), 1, cv2.LINE_AA)

    canvas[header_h : header_h + hand_disp_size, pad : pad + hand_disp_size] = hand_disp
    cv2.rectangle(canvas, (pad, header_h), (pad + hand_disp_size, header_h + hand_disp_size), (80, 95, 120), 2)

    gx0 = pad + hand_disp_size + pad
    for i, (r_idx, card) in enumerate(roi_cards):
        r = i // cols
        c = i % cols
        y0 = header_h + pad + r * (cell_h + pad)
        x0 = gx0 + pad + c * (cell_w + pad)
        canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = card
        g_key = ROI_TO_GROUP.get(r_idx, "Carpal")
        rgb = ANATOMICAL_GROUPS[g_key]["color_rgb"]
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w, y0 + cell_h), bgr, 2)

    out_demo_path = os.path.join(OUT_DEMO_DIR, "morphology_demo.png")
    cv2.imwrite(out_demo_path, canvas)
    print(f"Saved 1B-1C-1D morphology measurement visualization -> {out_demo_path}")
    return out_demo_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Demo of Morphological Modules 1B, 1C, 1D")
    parser.add_argument("--image", type=str, default="sample_male_01.png")
    args = parser.parse_args()
    render_demo_module_1bcd(args.image)
