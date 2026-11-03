# Global Branch: 5-Fold Whole-Image Bone Age Model (`EfficientNet-B0 512`)

This module provides the inference engine for the **5-Fold Whole-Image Global Branch** (`weights/global_folds/fold_0..4/best.pt`) with biological sex conditioning and **90% Split-Conformal Prediction Intervals (`±19.24 months`)**.

## Standalone Single-Image / DICOM Inference

```bash
python3 global_branch/scripts/predict_single_dicom.py --image samples/sample_male_01.png --sex M --real_age 84
```
