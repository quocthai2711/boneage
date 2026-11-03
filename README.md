# Unified Pediatric Bone Age Assessment Pipeline (Anatomical ROI + Global Ensemble + 4-Layer Clinical XAI)

An end-to-end, privacy-preserving pediatric bone age assessment and **Explainable AI (XAI)** system grounded in **Greulich & Pyle (G&P)**, **Tanner-Whitehouse (TW3)**, and **Gaskin (2011)** radiological standards:

1. **Anatomical Branch (`anatomical_branch/`)**:
   - Uses **interactive manual 17-keypoint annotation (`0 -> 16`)** (or pre-saved coordinates in `samples/sample_keypoints.json` / `outputs/xai_results/manual_annotations.json`), aligns the middle-finger axis (`KP6 -> KP8`) vertically, enhances contrast using `CLAHE` in LAB color space, and standardizes framing to `512×512`.
   - Extracts **16 anatomical regions of interest (ROIs)**: 14 phalangeal joints (`ROI 0..13`: DIP, PIP, MCP, Thumb IP/MCP) and 2 wrist regions (`ROI 15` Carpal Cavity, `ROI 16` Distal Radius & Ulna).
   - Computes **10 automated clinical morphological measurements (Modules 1A–1E)** orthogonal to each individual finger axis: Carpal bone count & occupancy (`1A`), Epiphyseal-to-Metaphyseal width ratio $R_w = W_e / W_m$ (`1B`), Physeal fusion score $F_r$ (`1C`), Adductor Sesamoid ossification (`1D`), and Distal Radius-Ulna fusion (`1E`).
   - Encodes each `224×224` ROI crop with gender- and region-specific **ResNet50 Triplet Encoders** (`encoder_{male,female}_{joints,wrists}_ep5.pt`), performs **Soft Top-K Cosine Similarity Retrieval** against the Vector Atlas Database (`weights/atlas_db/`), and classifies the hand into **Gaskin Biological Maturity Stages 1–5**.
2. **Global Branch (`global_branch/`)**:
   - **5-Fold Whole-Image Ensemble** (`weights/global_folds/fold_0..4/best.pt`) with biological sex conditioning and **90% Split-Conformal Prediction Intervals** (`±19.24 months`).
3. **Unified Meta-Ensembler & 4-Layer Clinical XAI (`run_pipeline.py`)**:
   - Combines global and anatomical predictions via a **Direct + Residual Meta-Ensembler** (`weights/unified_fusion.pkl`), achieving an **Overall MAE of 8.67 months** (Median AE: **6.44 months**).
   - Exports strictly anonymized **4-Layer XAI Dashboards (`xai.png`)** and **2-Panel Clinical UI images (`ui.png`)** showing **only Real Age, Predicted Age, and Error (MAE)** (zero patient IDs or names).

---

## Directory Structure

```text
github/
├── README.md                              # Setup & usage guide (English)
├── context.md                             # System architecture & clinical algorithm specification
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore configuration
├── run_pipeline.py                        # [Main] Single-image XAI inference & 5-Fold OOF evaluation
├── run_anatomical_ui.py                   # Standalone 16-ROI anatomical branch inference & UI
├── manual_keypoint_tool.py                # Interactive GUI for manual 17-keypoint annotation & XAI
├── preprocess_hospital.py                 # Two-stage standardization (512x512 RGB + 16 ROIs) for new scans
├── demo_xai.py                            # Quick 4-Layer XAI demo on anonymized sample radiographs
├── demo_morphology.py                     # Visual demo of orthogonal finger-axis morphology (Modules 1B-1D)
├── anatomical_branch/                     # 16-ROI Anatomical & Morphological XAI Engine
│   ├── __init__.py
│   ├── clinical_algorithm.py              # Modules 1A-1E, Soft Top-K Retrieval, Gaskin 5-Stage Gating
│   └── xai_explainer.py                   # Cosine Grad-CAM & Anonymized 4-Layer XAI Renderer
├── global_branch/                         # 5-Fold Whole-Image Global Branch
│   ├── README.md
│   ├── scripts/
│   │   └── predict_single_dicom.py        # Single DICOM / PNG / JPG global branch CLI
│   └── src/boneage/
│       ├── __init__.py
│       ├── dicom_io.py                    # Privacy-preserving DICOM reader
│       ├── models.py                      # Whole-image backbone + sex-conditioning head
│       └── preprocessing.py               # Percentile normalization & CLAHE preprocessing
├── weights/                               # Pretrained Checkpoints & Compact Atlas Database
│   ├── encoder_{male,female}_{joints,wrists}_ep5.pt
│   ├── {male,female}_rf*.pkl              # Calibrated Random Forest / Huber regressors
│   ├── clinical_stage_config.json         # Stage-specific anatomical group weights
│   ├── unified_fusion.pkl                 # Unified Direct + Residual Meta-Ensembler weights
│   ├── atlas_visual_crops.pt              # Compact JPEG-encoded 224x224 visual Atlas reference crops
│   ├── hospital_features_anonymized.csv   # Anonymized feature table for 5-Fold OOF benchmark
│   ├── atlas_db/{male,female}/roi_0..16.pt# 2048-D Vector Atlas Database
│   └── global_folds/fold_0..4/best.pt     # 5-Fold Global Branch checkpoints
├── samples/                               # 4 Anonymized Sample Hand Radiographs + 17 Keypoints
│   ├── sample_metadata.csv                # Sample metadata (sample_file, gender, real_age_months)
│   ├── sample_keypoints.json              # Pre-annotated 17 keypoints (0 -> 16) for the 4 sample radiographs
│   ├── sample_male_01.png
│   ├── sample_male_02.png
│   ├── sample_female_01.png
│   └── sample_female_02.png
└── outputs/                               # Exported XAI Dashboards & Evaluation Plots
    └── xai_results/
        ├── prediction_results.csv         # Anonymized results (real_age_months, predicted_age_months, mae_months)
        ├── mae_distribution_all.png
        ├── mae_distribution_female.png
        ├── mae_distribution_male.png
        └── sample_*/{xai.png, ui.png}
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Predict Bone Age & Export 4-Layer XAI on a Single X-Ray Image
```bash
# Run on a pre-annotated sample radiograph (loads 17 keypoints from samples/sample_keypoints.json):
python3 run_pipeline.py --image samples/sample_male_01.png --gender male --real_age 84

# Run on a new clinical radiograph (opens the interactive OpenCV GUI to click 17 keypoints 0 -> 16):
python3 run_pipeline.py --image path/to/new_scan.png --gender female

# Force re-annotating the 17 keypoints interactively with --redo:
python3 run_pipeline.py --image samples/sample_male_01.png --gender male --real_age 84 --redo
```
- **Interactive 17-Keypoint Order (`0 -> 16`)**:
  - `0-1-2`: Little finger (`DIP5 -> PIP5 -> MCP5`)
  - `3-4-5`: Ring finger (`DIP4 -> PIP4 -> MCP4`)
  - `6-7-8`: Middle finger (`DIP3 -> PIP3 -> MCP3`)
  - `9-10-11`: Index finger (`DIP2 -> PIP2 -> MCP2`)
  - `12-13`: Thumb (`IP1 -> MCP1`)
  - `14`: Thumb base (`CMC1`) | `15`: Carpal center | `16`: Wrist (`Distal Radius-Ulna`)
  - **Controls**: `Left-Click` = add point | `Right-Click` = undo | `R` = reset | `S`/`ESC` = skip.
- Output files are saved to `outputs/xai_results/<image_name>/`:
  - `xai.png`: Full 4-Layer Explainable AI Dashboard (`2620×1240`) with ROI-only Cosine Grad-CAM, side-by-side Atlas nearest-neighbor crops, and Gaskin Stage decision weights.
  - `ui.png`: Clean 2-Panel Clinical View (`1600×900`) comparing the original radiograph and the ROI-Grad-CAM overlay.

### 3. Run Quick Demos
```bash
# Run 4-Layer XAI demo on a sample radiograph
python3 demo_xai.py --sample sample_male_01.png

# Render orthogonal finger-axis morphological measurements (Modules 1B, 1C, 1D)
python3 demo_morphology.py --image sample_male_01.png
```

### 4. Reproduce 5-Fold OOF Benchmark & Export Sample XAI Dashboards
```bash
python3 run_pipeline.py
```

### 5. Standalone Interactive Manual 17-Keypoint Annotation & XAI Tool
```bash
python3 manual_keypoint_tool.py --image samples/sample_male_01.png --gender male --real_age 84 --redo
```
