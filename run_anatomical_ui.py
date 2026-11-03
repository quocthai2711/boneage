import os
import argparse
import glob
import json
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
import cv2
import pandas as pd
import numpy as np
import joblib

from anatomical_branch.clinical_algorithm import (
    imread_unicode,
    extract_roi_morphology,
    soft_topk_retrieval,
    blind_stage_and_aggregate,
    render_clinical_dashboard,
)
from anatomical_branch.xai_explainer import (
    extract_vecs_and_spatial_maps,
    generate_vietnamese_xai_summary,
    render_xai_dashboard,
)
from preprocess_hospital import preprocess_single_raw_image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
SAMPLES_DIR = os.path.join(ROOT_DIR, "samples")
UI_OUTPUT = os.path.join(ROOT_DIR, "outputs", "ui_results")
XAI_OUTPUT = os.path.join(ROOT_DIR, "outputs", "xai_results")

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

ROI_INDICES = list(range(14)) + [15, 16]
JOINT_INDICES = list(range(14))
WRIST_INDICES = [15, 16]

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


def predict_from_standardized_512(final_img, bboxes, case_label, gender, real_age, save_ui_path, enable_xai=False, save_xai_path=None, orig_img=None):
    h_fin, w_fin = final_img.shape[:2]
    patient_crops = {}
    valid_bboxes = {}

    for roi_idx in ROI_INDICES:
        if roi_idx not in bboxes:
            continue
        x1, y1, x2, y2 = bboxes[roi_idx]
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(w_fin, x2)), int(min(h_fin, y2))

        crop = final_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        patient_crops[roi_idx] = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (224, 224))
        valid_bboxes[roi_idx] = (x1, y1, x2, y2)

    if len(patient_crops) < 16:
        return None

    morph_feats = extract_roi_morphology(patient_crops, valid_bboxes=valid_bboxes)
    soft_ages, top1_ages, sigmas, atlas_crops = {}, {}, {}, {}
    top_sims, vec_map, fmap_map, atlas_vec_map = {}, {}, {}, {}

    with torch.no_grad():
        joint_batch = torch.stack([transform(patient_crops[r]) for r in JOINT_INDICES]).to(DEVICE)
        wrist_batch = torch.stack([transform(patient_crops[r]) for r in WRIST_INDICES]).to(DEVICE)

        if enable_xai:
            joint_vecs, joint_fmaps = extract_vecs_and_spatial_maps(models[gender]["joints"], joint_batch)
            wrist_vecs, wrist_fmaps = extract_vecs_and_spatial_maps(models[gender]["wrists"], wrist_batch)
            for i, r in enumerate(JOINT_INDICES):
                fmap_map[r] = joint_fmaps[i]
            for i, r in enumerate(WRIST_INDICES):
                fmap_map[r] = wrist_fmaps[i]
        else:
            joint_vecs = models[gender]["joints"](joint_batch).cpu()
            wrist_vecs = models[gender]["wrists"](wrist_batch).cpu()

        for i, r in enumerate(JOINT_INDICES):
            vec_map[r] = joint_vecs[i]
        for i, r in enumerate(WRIST_INDICES):
            vec_map[r] = wrist_vecs[i]

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

    render_clinical_dashboard(
        pat_id=case_label,
        gender=gender,
        real_age=real_age,
        final_img_bgr=final_img,
        valid_bboxes=valid_bboxes,
        patient_crops=patient_crops,
        atlas_crops=atlas_crops,
        soft_ages=soft_ages,
        sigmas=sigmas,
        morph_feats=morph_feats,
        agg_result=agg_res,
        save_path=save_ui_path,
        use_fast_pil=True,
    )

    xai_lines = None
    if enable_xai and save_xai_path:
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
        "real_age_months": round(float(real_age), 2),
        "predicted_age_months": round(float(pred_age), 2),
        "mae_months": round(float(mae), 2),
        "xai_lines": xai_lines,
    }


def main():
    parser = argparse.ArgumentParser(description="Anatomical Branch UI & 4-Layer XAI Dashboard")
    parser.add_argument("--image", type=str, default=None, help="Path to hand X-ray image or sample filename in samples/")
    parser.add_argument("--gender", type=str, default=None, choices=["male", "female"])
    parser.add_argument("--real_age", type=float, default=None)
    parser.add_argument("--xai", action="store_true", help="Also export 4-layer XAI dashboard")
    parser.add_argument("--redo", action="store_true", help="Force interactive manual 17-keypoint annotation (0 -> 16)")
    args = parser.parse_args()

    os.makedirs(UI_OUTPUT, exist_ok=True)
    os.makedirs(XAI_OUTPUT, exist_ok=True)

    sample_meta_csv = os.path.join(SAMPLES_DIR, "sample_metadata.csv")
    meta_map = {}
    if os.path.exists(sample_meta_csv):
        df_sm = pd.read_csv(sample_meta_csv)
        for _, r in df_sm.iterrows():
            stem = os.path.splitext(str(r["sample_file"]))[0]
            meta_map[stem] = (str(r["gender"]).lower(), float(r["real_age_months"]))

    targets = []
    if args.image is not None:
        img_target = args.image
        if not os.path.exists(img_target):
            cand = os.path.join(SAMPLES_DIR, img_target)
            if os.path.exists(cand):
                img_target = cand
        targets.append(img_target)
    else:
        targets = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.png")))

    rows = []
    for img_p in targets:
        orig_img = imread_unicode(img_p)
        if orig_img is None:
            continue
        stem = os.path.splitext(os.path.basename(img_p))[0]
        g = args.gender or meta_map.get(stem, ("female", 84.0))[0]
        ra = args.real_age if args.real_age is not None else meta_map.get(stem, ("female", 84.0))[1]
        final_img, _, squares = preprocess_single_raw_image(
            orig_img, img_path=img_p, gender=g, real_age=ra, force_interactive=args.redo
        )
        if final_img is None:
            continue
        bboxes = {int(s["point_index"]): tuple(s["bbox"]) for s in squares}
        ui_p = os.path.join(UI_OUTPUT, f"{stem}.png")
        xai_p = os.path.join(XAI_OUTPUT, stem, "xai.png") if args.xai else None
        res = predict_from_standardized_512(final_img, bboxes, stem, g, float(ra), ui_p, enable_xai=args.xai, save_xai_path=xai_p, orig_img=orig_img)
        if res is not None:
            rows.append({
                "real_age_months": res["real_age_months"],
                "predicted_age_months": res["predicted_age_months"],
                "mae_months": res["mae_months"],
            })
            print(f"Real Age: {res['real_age_months']:.1f}m | Predicted Age: {res['predicted_age_months']:.1f}m | MAE: {res['mae_months']:.2f}m -> {ui_p}")

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(UI_OUTPUT, "anatomical_results.csv"), index=False)


if __name__ == "__main__":
    main()
