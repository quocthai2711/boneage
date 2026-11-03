import os
import sys
import glob
import json
import argparse
import importlib.util
import joblib
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GLOBAL_DIR = os.path.join(ROOT_DIR, "global_branch")
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
SAMPLES_DIR = os.path.join(ROOT_DIR, "samples")
XAI_OUTPUT = os.path.join(ROOT_DIR, "outputs", "xai_results")
os.makedirs(XAI_OUTPUT, exist_ok=True)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from anatomical_branch.clinical_algorithm import (
    imread_unicode,
    extract_roi_morphology,
    soft_topk_retrieval,
    blind_stage_and_aggregate,
    ANATOMICAL_GROUPS,
    ROI_INDICES,
)
from anatomical_branch.xai_explainer import (
    extract_vecs_and_spatial_maps,
    render_xai_dashboard,
)
from preprocess_hospital import preprocess_single_raw_image


class EncoderResNet50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = torch.nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


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


def plot_dist(maes, label_str, save_paths):
    if len(maes) == 0:
        return
    maes_arr = np.array(maes, dtype=np.float64)
    mean_mae = float(np.mean(maes_arr))
    med_mae = float(np.median(maes_arr))
    below_cnt = int(np.sum(maes_arr < mean_mae))
    above_cnt = int(np.sum(maes_arr >= mean_mae))
    below_pct = 100.0 * below_cnt / len(maes_arr)
    above_pct = 100.0 * above_cnt / len(maes_arr)

    plt.figure(figsize=(10, 6))
    plt.hist(maes_arr, bins=20, color="skyblue", edgecolor="black", alpha=0.85)
    plt.axvline(
        mean_mae,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean MAE: {mean_mae:.2f}m | Median: {med_mae:.2f}m\n< Mean: {below_pct:.1f}% ({below_cnt})\n> Mean: {above_pct:.1f}% ({above_cnt})",
    )
    plt.title(
        f"MAE Distribution - {label_str.upper()} (Total: {len(maes_arr)})\n"
        f"Mean: {mean_mae:.2f}m | Median: {med_mae:.2f}m | < Mean: {below_pct:.1f}%"
    )
    plt.xlabel("MAE (months)")
    plt.ylabel("Number of Cases")
    plt.legend(loc="upper right")
    plt.grid(axis="y", alpha=0.75)
    for sp in save_paths:
        plt.savefig(sp, dpi=150, bbox_inches="tight")
    plt.close()


# Import Global Branch model & preprocessing
spec_m = importlib.util.spec_from_file_location("global_models", os.path.join(GLOBAL_DIR, "src", "boneage", "models.py"))
global_models = importlib.util.module_from_spec(spec_m)
spec_m.loader.exec_module(global_models)
GlobalBoneAgeModel = global_models.BoneAgeModel

spec_p = importlib.util.spec_from_file_location("global_prep", os.path.join(GLOBAL_DIR, "src", "boneage", "preprocessing.py"))
global_prep = importlib.util.module_from_spec(spec_p)
spec_p.loader.exec_module(global_prep)
BoneAgePreprocess = global_prep.BoneAgePreprocess

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

JOINT_INDICES = list(range(14))
WRIST_INDICES = [15, 16]
GROUP_KEYS = ["DIP", "PIP", "MCP", "Thumb", "Carpal", "Wrist"]
MORPH_COLS = [
    "n_carpal", "carpal_occupancy", "mean_rw", "mean_f_dip", "mean_f_pip",
    "mean_f_mcp", "sesamoid_score", "sesamoid_present", "wrist_sobel", "wrist_fusion"
]

transform_224 = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_global_ensemble():
    ckpts = sorted(glob.glob(os.path.join(WEIGHTS_DIR, "global_folds", "fold_*", "best.pt")))
    ensemble = []
    for cp in ckpts:
        st = torch.load(cp, map_location=DEVICE, weights_only=False)
        m = GlobalBoneAgeModel(backbone_name=st["backbone"], pretrained=False).to(DEVICE)
        m.load_state_dict(st["model_state"], strict=True)
        m.eval()
        ensemble.append({
            "model": m,
            "target_mean": float(st["target_mean"]),
            "target_std": float(st["target_std"]),
            "preprocess": BoneAgePreprocess(mode=st["preprocess_mode"], target_size=int(st["resolution"]), is_train=False),
        })
    return ensemble


def load_anatomical_models_and_atlas():
    with open(os.path.join(WEIGHTS_DIR, "clinical_stage_config.json"), "r", encoding="utf-8") as f:
        clinical_cfg = json.load(f)

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
        rf_v2 = joblib.load(os.path.join(WEIGHTS_DIR, f"{gender}_rf_v2.pkl"))
        models[gender]["rf"] = CalibratedRFAdapter(rf_v1, rf_v2, w_v1=0.0 if gender == "male" else 1.0, is_debiased=(gender == "male"))

        for idx in ROI_INDICES:
            roi_db = torch.load(os.path.join(WEIGHTS_DIR, "atlas_db", gender, f"roi_{idx}.pt"), map_location="cpu")
            db_tensors[gender][idx] = torch.stack([d["vector"].float() for d in roi_db.values()])
            db_ages[gender][idx] = torch.tensor([float(d["age"]) for d in roi_db.values()], dtype=torch.float32)
            ids_list = list(roi_db.keys())
            db_ids[gender][idx] = ids_list
            db_id_to_row[gender][idx] = {cid: row_i for row_i, cid in enumerate(ids_list)}

    visual_bank_path = os.path.join(WEIGHTS_DIR, "atlas_visual_crops.pt")
    visual_bank = torch.load(visual_bank_path, map_location="cpu") if os.path.exists(visual_bank_path) else {"male": {}, "female": {}}
    crop_cache = {}

    def get_atlas_crop(gender, roi_idx, target_age):
        a_int = int(round(float(target_age)))
        ckey = (gender, roi_idx, a_int)
        if ckey in crop_cache:
            return crop_cache[ckey]
        g_bank = visual_bank.get(gender, {}).get(roi_idx, {})
        if not g_bank:
            return None
        if a_int in g_bank:
            buf_bytes = g_bank[a_int]
        else:
            nearest_a = min(g_bank.keys(), key=lambda k: abs(int(k) - a_int))
            buf_bytes = g_bank[nearest_a]
        arr = np.frombuffer(buf_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crop_cache[ckey] = rgb
        return rgb

    return clinical_cfg, models, db_tensors, db_ages, db_ids, db_id_to_row, get_atlas_crop


def predict_global_single(bgr_img, sex_val, global_ens):
    pil_gray = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY))
    prep = global_ens[0]["preprocess"]
    x = prep(pil_gray).unsqueeze(0).to(DEVICE)
    s = torch.tensor([[float(sex_val)]], dtype=torch.float32, device=DEVICE)
    preds = []
    with torch.no_grad():
        for item in global_ens:
            z = float(item["model"](x, s).cpu().item())
            preds.append(z * item["target_std"] + item["target_mean"])
    return float(np.mean(preds)), float(np.std(preds)), preds


def build_unified_feature_matrices(df):
    p_anat = df["anatomical_pred"].to_numpy(dtype=float)
    p_glob_raw = df["global_raw_pred"].to_numpy(dtype=float)
    p_glob_std = df["global_std_pred"].to_numpy(dtype=float)
    s_anat = df["anatomical_roi_std"].to_numpy(dtype=float)
    s_glob = df["global_raw_std"].to_numpy(dtype=float)
    sex = df["sex"].to_numpy(dtype=float)
    stages = df["stage"].to_numpy(dtype=int)

    X_rois = df[[f"roi_{r}_age" for r in ROI_INDICES]].to_numpy(dtype=float)
    X_sigs = df[[f"roi_{r}_sigma" for r in ROI_INDICES]].to_numpy(dtype=float)
    inv_var = 1.0 / (X_sigs ** 2 + 25.0)
    inv_var_norm = inv_var / np.sum(inv_var, axis=1, keepdims=True)
    precision_weighted_anat = np.sum(X_rois * inv_var_norm, axis=1)

    # Step 1: Cross-Anchor Outlier Gating
    anchor = (0.35 * precision_weighted_anat + 0.65 * p_glob_raw).reshape(-1, 1)
    X_clamped = np.clip(X_rois, anchor - 20.0, anchor + 20.0)
    s_sorted = np.sort(X_clamped, axis=1)
    tmean_4 = np.mean(s_sorted[:, 2:-2], axis=1)
    tmean_6 = np.mean(s_sorted[:, 4:-4], axis=1)
    q25 = np.percentile(X_clamped, 25, axis=1)
    q75 = np.percentile(X_clamped, 75, axis=1)

    # Step 2: 7-Group Anatomical Matrix
    G7_mat = np.zeros((len(df), 7), dtype=float)
    G6_rel = np.zeros((len(df), 6), dtype=float)
    for i in range(len(df)):
        for g_idx, g_k in enumerate(GROUP_KEYS):
            m_cols = [ROI_INDICES.index(r) for r in ANATOMICAL_GROUPS[g_k]["rois"]]
            G7_mat[i, g_idx] = float(np.mean(X_clamped[i, m_cols]))
            G6_rel[i, g_idx] = G7_mat[i, g_idx] - p_glob_raw[i]
        G7_mat[i, 6] = p_glob_raw[i]

    # Step 3A: Direct Meta-Feature Matrix
    group_cols = [f"group_{g_k}" for g_k in GROUP_KEYS]
    meta_feature_cols = (
        ["anatomical_pred", "anatomical_rf_pred", "anatomical_clin_pred", "global_raw_pred", "global_std_pred",
         "anatomical_sigma_mean", "anatomical_roi_std", "global_raw_std", "sex", "stage"]
        + group_cols
        + MORPH_COLS
    )
    X_meta_base = df[meta_feature_cols].to_numpy(dtype=float)
    diff_ag = np.abs(p_anat - p_glob_raw).reshape(-1, 1)
    unc_ratio = (s_anat / (s_glob + 2.0)).reshape(-1, 1)
    X_meta_full = np.hstack([
        X_meta_base, G7_mat, diff_ag, unc_ratio,
        precision_weighted_anat.reshape(-1, 1),
        tmean_4.reshape(-1, 1), tmean_6.reshape(-1, 1),
        q25.reshape(-1, 1), q75.reshape(-1, 1),
    ])

    # Step 3B: Residual Anatomical Correction Feature Matrix
    X_rel_rois = X_clamped - p_glob_raw.reshape(-1, 1)
    X_morph = df[MORPH_COLS].to_numpy(dtype=float)
    X_res_feats = np.column_stack([
        p_glob_raw, p_glob_std - p_glob_raw,
        p_anat - p_glob_raw,
        df["anatomical_clin_pred"].to_numpy() - p_glob_raw,
        precision_weighted_anat - p_glob_raw,
        s_anat, s_glob, df["anatomical_sigma_mean"].to_numpy(),
        sex, stages,
        G6_rel, X_rel_rois, X_morph,
    ])
    return G7_mat, X_meta_full, X_res_feats


def fit_champion_fusion_models(G7_tr, X_meta_tr, X_res_tr, y_tr, p_glob_tr, sex_tr, stages_tr):
    stage_sex_7weights = {}
    for sx in [0.0, 1.0]:
        for st in range(1, 6):
            m_tr = (sex_tr == sx) & (stages_tr == st)
            if np.sum(m_tr) < 10:
                m_tr = (sex_tr == sx)
            G_sub, y_sub = G7_tr[m_tr], y_tr[m_tr]

            def loss_7g(params):
                w7 = np.maximum(0.0, params[:7])
                w7 = w7 / (np.sum(w7) + 1e-8)
                return np.mean(np.abs(G_sub.dot(w7) + params[7] - y_sub)) + 0.05 * np.sum(w7[:6] ** 2)

            res7 = minimize(
                loss_7g,
                np.array([0.08] * 5 + [0.10, 0.50, 0.0]),
                bounds=[(0.0, 1.0)] * 7 + [(-10.0, 10.0)],
                method="SLSQP",
            )
            w7_opt = np.maximum(0.0, res7.x[:7])
            w7_opt /= (np.sum(w7_opt) + 1e-8)
            b7_opt = float(res7.x[7])
            stage_sex_7weights[(sx, st)] = (w7_opt, b7_opt)

    ridge_g = make_pipeline(StandardScaler(), Ridge(alpha=60.0)).fit(X_meta_tr, y_tr)
    rf_g = RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1).fit(X_meta_tr, y_tr)
    et_g = ExtraTreesRegressor(n_estimators=250, max_depth=9, min_samples_leaf=4, max_features=0.6, random_state=42, n_jobs=-1).fit(X_meta_tr, y_tr)
    gbr_g = GradientBoostingRegressor(loss="huber", alpha=0.85, n_estimators=120, max_depth=3, learning_rate=0.05, min_samples_leaf=6, random_state=42).fit(X_meta_tr, y_tr)

    m_tr_f = (sex_tr == 0.0)
    et_fem = ExtraTreesRegressor(n_estimators=300, max_depth=9, min_samples_leaf=4, max_features=0.6, random_state=42, n_jobs=-1).fit(X_meta_tr[m_tr_f], y_tr[m_tr_f])
    gbr_fem = GradientBoostingRegressor(loss="huber", alpha=0.80, n_estimators=120, max_depth=3, learning_rate=0.04, min_samples_leaf=5, subsample=0.8, random_state=42).fit(X_meta_tr[m_tr_f], y_tr[m_tr_f])

    delta_tr = y_tr - p_glob_tr
    huber_r = make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000, alpha=20.0, epsilon=1.35)).fit(X_res_tr, delta_tr)
    ridge_r = make_pipeline(StandardScaler(), Ridge(alpha=150.0)).fit(X_res_tr, delta_tr)
    gbr_r = GradientBoostingRegressor(loss="huber", alpha=0.80, n_estimators=100, max_depth=3, learning_rate=0.04, min_samples_leaf=6, subsample=0.8, random_state=42).fit(X_res_tr, delta_tr)
    rf_r = RandomForestRegressor(n_estimators=250, max_depth=7, min_samples_leaf=5, max_features=0.5, random_state=42, n_jobs=-1).fit(X_res_tr, delta_tr)
    et_r = ExtraTreesRegressor(n_estimators=250, max_depth=8, min_samples_leaf=5, max_features=0.5, random_state=42, n_jobs=-1).fit(X_res_tr, delta_tr)

    return {
        "stage_sex_7weights": stage_sex_7weights,
        "direct": {"ridge_g": ridge_g, "rf_g": rf_g, "et_g": et_g, "gbr_g": gbr_g, "et_fem": et_fem, "gbr_fem": gbr_fem},
        "residual": {"huber_r": huber_r, "ridge_r": ridge_r, "gbr_r": gbr_r, "rf_r": rf_r, "et_r": et_r},
    }


def predict_champion_fusion(bundle, G7_va, X_meta_va, X_res_va, p_glob_va, sex_va, stages_va):
    n = len(p_glob_va)
    p_7group = np.zeros(n, dtype=float)
    for i in range(n):
        key = (float(sex_va[i]), int(stages_va[i]))
        w7, b7 = bundle["stage_sex_7weights"].get(key, (np.array([0.08] * 5 + [0.10, 0.50]), 0.0))
        p_7group[i] = float(G7_va[i].dot(w7) + b7)

    d_models = bundle["direct"]
    pred_global_stack = (
        0.30 * p_7group
        + 0.25 * d_models["gbr_g"].predict(X_meta_va)
        + 0.20 * d_models["et_g"].predict(X_meta_va)
        + 0.15 * d_models["rf_g"].predict(X_meta_va)
        + 0.10 * d_models["ridge_g"].predict(X_meta_va)
    )

    p_direct = np.zeros(n, dtype=float)
    for i in range(n):
        if sex_va[i] == 0.0:
            p_fem_spec = 0.5 * d_models["et_fem"].predict(X_meta_va[i : i + 1])[0] + 0.5 * d_models["gbr_fem"].predict(X_meta_va[i : i + 1])[0]
            p_direct[i] = 0.55 * pred_global_stack[i] + 0.45 * p_fem_spec
        else:
            p_direct[i] = pred_global_stack[i]

    r_models = bundle["residual"]
    delta_pred = (
        0.30 * r_models["gbr_r"].predict(X_res_va)
        + 0.25 * r_models["rf_r"].predict(X_res_va)
        + 0.20 * r_models["et_r"].predict(X_res_va)
        + 0.15 * r_models["huber_r"].predict(X_res_va)
        + 0.10 * r_models["ridge_r"].predict(X_res_va)
    )
    p_residual = p_glob_va + delta_pred

    p_champion = np.zeros(n, dtype=float)
    for i in range(n):
        if int(stages_va[i]) in (1, 4, 5) or sex_va[i] == 1.0:
            p_champion[i] = 0.35 * p_direct[i] + 0.65 * p_residual[i]
        else:
            p_champion[i] = 0.65 * p_direct[i] + 0.35 * p_residual[i]

    return p_champion, p_7group, p_direct, p_residual


def render_single_unified_xai(
    final_img, bboxes, case_label, gender, real_age, unified_pred_age,
    clinical_cfg, models, db_tensors, db_ages, db_ids, db_id_to_row, get_atlas_crop,
    save_xai_path, keypoints=None, orig_img=None
):
    """Render and export the Anonymized 4-Layer XAI Dashboard with the Unified predicted age."""
    h_fin, w_fin = final_img.shape[:2]
    crops, valid_bboxes = {}, {}
    for r in ROI_INDICES:
        if r in bboxes:
            x1, y1, x2, y2 = [int(v) for v in bboxes[r]]
            c = final_img[max(0, y1) : min(h_fin, y2), max(0, x1) : min(w_fin, x2)]
            if c.size > 0:
                crops[r] = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (224, 224))
                valid_bboxes[r] = (x1, y1, x2, y2)
    if len(crops) < 16:
        return None

    morph = extract_roi_morphology(crops, valid_bboxes=valid_bboxes, keypoints=keypoints)
    sa, ta, sg, top_sims, vec_map, fmap_map, atlas_vec_map, atlas_crops = {}, {}, {}, {}, {}, {}, {}, {}

    with torch.no_grad():
        jb = torch.stack([transform_224(crops[r]) for r in JOINT_INDICES]).to(DEVICE)
        wb = torch.stack([transform_224(crops[r]) for r in WRIST_INDICES]).to(DEVICE)
        jv, jf = extract_vecs_and_spatial_maps(models[gender]["joints"], jb)
        wv, wf = extract_vecs_and_spatial_maps(models[gender]["wrists"], wb)
        for i, r in enumerate(JOINT_INDICES):
            vec_map[r] = jv[i]
            fmap_map[r] = jf[i]
        for i, r in enumerate(WRIST_INDICES):
            vec_map[r] = wv[i]
            fmap_map[r] = wf[i]

        for r in ROI_INDICES:
            ret = soft_topk_retrieval(vec_map[r], db_tensors[gender][r], db_ages[gender][r], db_ids[gender][r], k=10, tau=180.0)
            sa[r], ta[r], sg[r] = ret["soft_age"], ret["top1_age"], ret["sigma"]
            top_sims[r] = ret.get("top_sim", 0.99)
            best_cid = ret["best_id"]
            if best_cid in db_id_to_row[gender].get(r, {}):
                row_i = db_id_to_row[gender][r][best_cid]
                atlas_vec_map[r] = db_tensors[gender][r][row_i]
            ac = get_atlas_crop(gender, r, sa[r])
            if ac is not None:
                atlas_crops[r] = ac

    agg = blind_stage_and_aggregate(sa, sg, morph, gender, rf_model=models[gender]["rf"], learned_config=clinical_cfg.get(gender, {}), top1_ages=ta)
    if unified_pred_age is not None:
        agg["esa_months"] = float(unified_pred_age)

    render_xai_dashboard(
        pat_id=case_label, gender=gender, real_age=real_age, final_img_bgr=final_img,
        valid_bboxes=valid_bboxes, patient_crops=crops, atlas_crops=atlas_crops,
        fmaps_map=fmap_map, query_vecs_map=vec_map, atlas_vecs_map=atlas_vec_map,
        top_sims=top_sims, soft_ages=sa, sigmas=sg, morph_feats=morph,
        agg_result=agg, save_path=save_xai_path, orig_img_bgr=orig_img,
    )
    return agg, morph, sa, sg


def run_single_image_inference(
    img_path, gender, real_age,
    clinical_cfg, models, db_tensors, db_ages, db_ids, db_id_to_row, get_atlas_crop,
    global_ens, fusion_bundle, force_interactive=False
):
    orig_img = imread_unicode(img_path)
    if orig_img is None:
        print(f"Failed to read image: {img_path}")
        return None

    case_label = os.path.splitext(os.path.basename(img_path))[0]
    sex_val = 1.0 if gender == "male" else 0.0

    final_img, kpts_512, squares = preprocess_single_raw_image(
        orig_img, img_path=img_path, gender=gender, real_age=real_age, force_interactive=force_interactive
    )
    if final_img is None:
        print(f"Manual 17-keypoint annotation (0 -> 16) was skipped or incomplete for image: {img_path}")
        return None

    glob_raw_p, glob_raw_s, _ = predict_global_single(orig_img, sex_val, global_ens)
    glob_std_p, _, _ = predict_global_single(final_img, sex_val, global_ens)

    bboxes = {int(s["point_index"]): tuple(s["bbox"]) for s in squares}
    h_fin, w_fin = final_img.shape[:2]
    crops, valid_bboxes = {}, {}
    for r in ROI_INDICES:
        if r in bboxes:
            x1, y1, x2, y2 = [int(v) for v in bboxes[r]]
            c = final_img[max(0, y1) : min(h_fin, y2), max(0, x1) : min(w_fin, x2)]
            if c.size > 0:
                crops[r] = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (224, 224))
                valid_bboxes[r] = (x1, y1, x2, y2)

    morph = extract_roi_morphology(crops, valid_bboxes=valid_bboxes, keypoints=kpts_512)
    sa, ta, sg = {}, {}, {}
    with torch.no_grad():
        jb = torch.stack([transform_224(crops[r]) for r in JOINT_INDICES]).to(DEVICE)
        wb = torch.stack([transform_224(crops[r]) for r in WRIST_INDICES]).to(DEVICE)
        jv = models[gender]["joints"](jb).cpu()
        wv = models[gender]["wrists"](wb).cpu()
        vm = {r: jv[i] for i, r in enumerate(JOINT_INDICES)}
        for i, r in enumerate(WRIST_INDICES):
            vm[r] = wv[i]
        for r in ROI_INDICES:
            ret = soft_topk_retrieval(vm[r], db_tensors[gender][r], db_ages[gender][r], db_ids[gender][r], k=10, tau=180.0)
            sa[r], ta[r], sg[r] = ret["soft_age"], ret["top1_age"], ret["sigma"]

    agg = blind_stage_and_aggregate(sa, sg, morph, gender, rf_model=models[gender]["rf"], learned_config=clinical_cfg.get(gender, {}), top1_ages=ta)

    row_dict = {
        "gender": gender, "sex": sex_val, "real_age": real_age if real_age is not None else 0.0,
        "anatomical_pred": float(agg["esa_months"]), "anatomical_rf_pred": float(agg["a_init"]), "anatomical_clin_pred": float(agg["clinical_age"]),
        "stage": int(agg["stage"]), "anatomical_sigma_mean": float(np.mean(list(sg.values()))), "anatomical_roi_std": float(np.std(list(sa.values()))),
        "global_raw_pred": glob_raw_p, "global_raw_std": glob_raw_s, "global_std_pred": glob_std_p,
    }
    for r in ROI_INDICES:
        row_dict[f"roi_{r}_age"] = float(sa[r])
        row_dict[f"roi_{r}_sigma"] = float(sg[r])
    for g_k in GROUP_KEYS:
        row_dict[f"group_{g_k}"] = float(agg["group_ages"][g_k])
    for mc in MORPH_COLS:
        row_dict[mc] = float(morph["summary"][mc])

    df_single = pd.DataFrame([row_dict])
    G7_s, X_meta_s, X_res_s = build_unified_feature_matrices(df_single)
    p_champ, _, _, _ = predict_champion_fusion(
        fusion_bundle, G7_s, X_meta_s, X_res_s,
        np.array([glob_raw_p]), np.array([sex_val]), np.array([int(agg["stage"])])
    )
    final_pred = float(p_champ[0])

    case_out_dir = os.path.join(XAI_OUTPUT, case_label)
    os.makedirs(case_out_dir, exist_ok=True)
    xai_save_path = os.path.join(case_out_dir, "xai.png")
    render_single_unified_xai(
        final_img, bboxes, case_label, gender, real_age, final_pred,
        clinical_cfg, models, db_tensors, db_ages, db_ids, db_id_to_row, get_atlas_crop,
        xai_save_path, keypoints=kpts_512, orig_img=orig_img,
    )
    return {
        "case_dir": case_out_dir,
        "stage": int(agg["stage"]),
        "real_age_months": real_age,
        "anatomical_pred_months": row_dict["anatomical_pred"],
        "global_pred_months": glob_raw_p,
        "global_std_months": glob_raw_s,
        "predicted_age_months": final_pred,
        "mae_months": abs(final_pred - real_age) if real_age is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-End Unified Bone Age Pipeline (Global + Anatomical + 4-Layer XAI)")
    parser.add_argument("--image", type=str, default=None, help="Path to input hand X-ray image or sample filename in samples/")
    parser.add_argument("--gender", type=str, default=None, choices=["male", "female"], help="Biological sex (male / female)")
    parser.add_argument("--real_age", type=float, default=None, help="Optional ground-truth bone age in months (to compute MAE)")
    parser.add_argument("--redo", action="store_true", help="Force interactive manual 17-keypoint annotation (0 -> 16) even if saved coordinates exist")
    args = parser.parse_args()

    cached_csv = os.path.join(WEIGHTS_DIR, "hospital_features_anonymized.csv")
    fusion_pkl_path = os.path.join(WEIGHTS_DIR, "unified_fusion.pkl")
    sample_meta_csv = os.path.join(SAMPLES_DIR, "sample_metadata.csv")

    print("Loading Anatomical Branch (ResNet50 Encoders + Vector Atlas DB) and 5-Fold Global Ensemble...")
    clinical_cfg, models, db_tensors, db_ages, db_ids, db_id_to_row, get_atlas_crop = load_anatomical_models_and_atlas()
    global_ens = load_global_ensemble()
    fusion_bundle = joblib.load(fusion_pkl_path)

    sample_meta_map = {}
    if os.path.exists(sample_meta_csv):
        df_sm = pd.read_csv(sample_meta_csv)
        for _, r in df_sm.iterrows():
            stem = os.path.splitext(str(r["sample_file"]))[0]
            sample_meta_map[stem] = (str(r["gender"]).lower(), float(r["real_age_months"]))
            sample_meta_map[str(r["sample_file"])] = (str(r["gender"]).lower(), float(r["real_age_months"]))

    # --- MODE 1: SINGLE-IMAGE INFERENCE & XAI EXPORT ---
    if args.image is not None:
        img_target = args.image
        if not os.path.exists(img_target):
            cand = os.path.join(SAMPLES_DIR, img_target)
            if os.path.exists(cand):
                img_target = cand
            else:
                matches = glob.glob(os.path.join(SAMPLES_DIR, f"*{img_target}*.png"))
                if matches:
                    img_target = matches[0]
        if not os.path.exists(img_target):
            print(f"Image file not found: {args.image}")
            return

        stem = os.path.splitext(os.path.basename(img_target))[0]
        gender = args.gender
        real_age = args.real_age
        if stem in sample_meta_map:
            if gender is None:
                gender = sample_meta_map[stem][0]
            if real_age is None:
                real_age = sample_meta_map[stem][1]
        if gender is None:
            gender = "male" if "male" in img_target.lower() and "female" not in img_target.lower() else "female"

        res = run_single_image_inference(
            img_target, gender.lower(), real_age,
            clinical_cfg, models, db_tensors, db_ages, db_ids, db_id_to_row, get_atlas_crop,
            global_ens, fusion_bundle, force_interactive=args.redo
        )
        if res is not None:
            print("\n================ BONE AGE PREDICTION & XAI SUMMARY ================")
            if res["real_age_months"] is not None:
                print(f"Real Age            : {res['real_age_months']:.2f} mos ({res['real_age_months']/12.0:.2f} yrs)")
            print(f"Predicted Age       : {res['predicted_age_months']:.2f} mos ({res['predicted_age_months']/12.0:.2f} yrs)")
            if res["mae_months"] is not None:
                print(f"Absolute Error (MAE): {res['mae_months']:.2f} mos")
            print(f"90% Conformal CI    : [{max(0.0, res['predicted_age_months'] - 19.24):.1f}, {res['predicted_age_months'] + 19.24:.1f}] mos")
            print(f"Exported XAI & UI   : {res['case_dir']}/ (xai.png & ui.png)")
            print("===================================================================")
        return

    # --- MODE 2: 5-FOLD OOF EVALUATION & SAMPLE XAI GENERATION ---
    if os.path.exists(cached_csv):
        df = pd.read_csv(cached_csv)
        print(f"=== Running 5-Fold OOF Evaluation ({len(df)} cases) & Exporting Anonymized Results ===")
        y = df["real_age"].to_numpy(dtype=float)
        p_glob_raw = df["global_raw_pred"].to_numpy(dtype=float)
        sex = df["sex"].to_numpy(dtype=float)
        stages = df["stage"].to_numpy(dtype=int)
        folds = df["cv_fold"].to_numpy(dtype=int)

        G7_mat, X_meta_full, X_res_feats = build_unified_feature_matrices(df)
        oof_champion = np.zeros_like(y)

        for f_id in range(5):
            va_idx = np.where(folds == f_id)[0]
            tr_idx = np.where(folds != f_id)[0]
            bundle_fold = fit_champion_fusion_models(
                G7_mat[tr_idx], X_meta_full[tr_idx], X_res_feats[tr_idx],
                y[tr_idx], p_glob_raw[tr_idx], sex[tr_idx], stages[tr_idx]
            )
            p_champ, _, _, _ = predict_champion_fusion(
                bundle_fold, G7_mat[va_idx], X_meta_full[va_idx], X_res_feats[va_idx],
                p_glob_raw[va_idx], sex[va_idx], stages[va_idx]
            )
            oof_champion[va_idx] = p_champ

        full_bundle = fit_champion_fusion_models(G7_mat, X_meta_full, X_res_feats, y, p_glob_raw, sex, stages)
        joblib.dump(full_bundle, fusion_pkl_path)

        # Export anonymized CSV containing only real_age_months, predicted_age_months, mae_months
        df_out = pd.DataFrame({
            "real_age_months": np.round(y, 2),
            "predicted_age_months": np.round(oof_champion, 2),
            "mae_months": np.round(np.abs(oof_champion - y), 2),
        })
        out_csv_xai = os.path.join(XAI_OUTPUT, "prediction_results.csv")
        df_out.to_csv(out_csv_xai, index=False)

        ae = np.abs(oof_champion - y)
        mae_f = ae[df["gender"] == "female"].tolist()
        mae_m = ae[df["gender"] == "male"].tolist()
        plot_dist(mae_m, "male", [os.path.join(XAI_OUTPUT, "mae_distribution_male.png")])
        plot_dist(mae_f, "female", [os.path.join(XAI_OUTPUT, "mae_distribution_female.png")])
        plot_dist(mae_m + mae_f, "all", [os.path.join(XAI_OUTPUT, "mae_distribution_all.png")])

        m412 = (y >= 48) & (y <= 144)
        print("\n================ UNIFIED PIPELINE EVALUATION SUMMARY ================")
        print(f"Total Evaluated Cases : {len(df)} (Female: {len(mae_f)}, Male: {len(mae_m)})")
        print(f"Overall MAE           : {np.mean(ae):.2f} mos (Median AE: {np.median(ae):.2f} mos)")
        print(f"Female MAE (N={len(mae_f):3d})    : {np.mean(mae_f):.2f} mos (Median AE: {np.median(mae_f):.2f} mos)")
        print(f"Male MAE (N={len(mae_m):3d})      : {np.mean(mae_m):.2f} mos (Median AE: {np.median(mae_m):.2f} mos)")
        print(f"Ages 4-12 yrs MAE     : {np.mean(ae[m412]):.2f} mos (N={int(m412.sum())})")
        print(f"Anonymized Results CSV: {out_csv_xai}")
        print("=====================================================================")

    # Export 4-Layer XAI Dashboards for sample images in samples/
    sample_pngs = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.png")))
    for sp in tqdm(sample_pngs, desc="Exporting 4-Layer XAI Dashboards for samples/"):
        stem = os.path.splitext(os.path.basename(sp))[0]
        g_s, r_a = sample_meta_map.get(stem, ("female", 84.0))
        run_single_image_inference(
            sp, g_s, r_a,
            clinical_cfg, models, db_tensors, db_ages, db_ids, db_id_to_row, get_atlas_crop,
            global_ens, fusion_bundle
        )


if __name__ == "__main__":
    main()
