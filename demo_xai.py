import os
import argparse
import pandas as pd
from run_pipeline import (
    SAMPLES_DIR,
    load_anatomical_models_and_atlas,
    load_global_ensemble,
    run_single_image_inference,
    WEIGHTS_DIR,
)
import joblib


def main():
    parser = argparse.ArgumentParser(description="Run 4-Layer XAI Demo on Anonymized Samples in samples/")
    parser.add_argument("--sample", type=str, default="sample_male_01.png", help="Sample filename in samples/ (e.g. sample_male_01.png)")
    args = parser.parse_args()

    clinical_cfg, models, db_tensors, db_ages, db_ids, db_id_to_row, get_atlas_crop = load_anatomical_models_and_atlas()
    global_ens = load_global_ensemble()
    fusion_bundle = joblib.load(os.path.join(WEIGHTS_DIR, "unified_fusion.pkl"))

    meta_df = pd.read_csv(os.path.join(SAMPLES_DIR, "sample_metadata.csv"))
    meta_map = {str(r["sample_file"]): (str(r["gender"]).lower(), float(r["real_age_months"])) for _, r in meta_df.iterrows()}

    fname = args.sample if args.sample.endswith(".png") else f"{args.sample}.png"
    img_path = os.path.join(SAMPLES_DIR, fname)
    gender, real_age = meta_map.get(fname, ("male", 84.0))

    res = run_single_image_inference(
        img_path, gender, real_age,
        clinical_cfg, models, db_tensors, db_ages, db_ids, db_id_to_row, get_atlas_crop,
        global_ens, fusion_bundle
    )
    if res is not None:
        print(f"Real Age: {res['real_age_months']:.1f}m | Predicted Age: {res['predicted_age_months']:.1f}m | MAE: {res['mae_months']:.2f}m -> {res['case_dir']}/")


if __name__ == "__main__":
    main()
