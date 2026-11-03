"""Single-Image / DICOM End-to-End Inference CLI for Global Branch (Whole-Image E-B0 512).

Usage:
    python global_branch/scripts/predict_single_dicom.py --image samples/sample_male_01.png --sex M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PIPE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPE_ROOT.parent
sys.path.insert(0, str(PIPE_ROOT / "src"))

from boneage.dicom_io import load_dicom_file
from boneage.models import BoneAgeModel
from boneage.preprocessing import BoneAgePreprocess


def load_image_512_raw_tensor(image_path: Path, sex_label: str, image_size: int = 512) -> torch.Tensor:
    preprocess = BoneAgePreprocess(mode="P1", target_size=image_size, is_train=False)
    if image_path.suffix.lower() in {".dcm", ".dicom"}:
        pil_img = load_dicom_file(image_path, sex_override=sex_label).image
    else:
        pil_img = Image.open(image_path).convert("L")
    return preprocess(pil_img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-Image Global Bone Age Inference")
    parser.add_argument("--image", type=Path, required=True, help="Path to DICOM (.dcm) or PNG/JPG image")
    parser.add_argument("--sex", type=str, required=True, choices=["F", "M", "f", "m", "0", "1"], help="Biological sex (F/M)")
    parser.add_argument("--real_age", type=float, default=None, help="Optional real age in months to compute MAE")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda)")
    args = parser.parse_args()

    sex_val = 1.0 if args.sex.upper() in {"M", "1"} else 0.0
    sex_lbl = "M" if sex_val == 1.0 else "F"
    x = load_image_512_raw_tensor(args.image, sex_lbl, image_size=512).unsqueeze(0).to(args.device)
    s = torch.tensor([[sex_val]], dtype=torch.float32, device=args.device)

    ckpt_paths = sorted((REPO_ROOT / "weights" / "global_folds").glob("fold_*/best.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {REPO_ROOT / 'weights' / 'global_folds'}")

    fold_preds = []
    for cp in ckpt_paths:
        state = torch.load(cp, map_location=args.device, weights_only=False)
        model = BoneAgeModel(backbone_name=state.get("backbone", "efficientnet_b0"), pretrained=False).to(args.device)
        sd = state.get("model_state") or state.get("model_state_dict") or state
        model.load_state_dict(sd, strict=True)
        model.eval()
        t_mean = float(state.get("target_mean", 0.0))
        t_std = float(state.get("target_std", 1.0))
        with torch.no_grad():
            z = float(model(x, s).cpu().item())
            fold_preds.append(z * t_std + t_mean)

    mean_pred = float(np.mean(fold_preds))
    std_pred = float(np.std(fold_preds))
    q90_radius = 19.24
    result = {
        "predicted_age_months": round(mean_pred, 2),
        "predicted_age_years": round(mean_pred / 12.0, 2),
        "fold_std_months": round(std_pred, 2),
        "conformal_90_interval_months": [round(mean_pred - q90_radius, 2), round(mean_pred + q90_radius, 2)],
    }
    if args.real_age is not None:
        result["real_age_months"] = round(float(args.real_age), 2)
        result["mae_months"] = round(abs(mean_pred - float(args.real_age)), 2)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
