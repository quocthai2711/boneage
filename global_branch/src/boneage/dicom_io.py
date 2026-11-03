"""Privacy-preserving DICOM loading for local hospital evaluation."""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset

MAX_DICOM_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class DicomCase:
    """De-identified model input decoded from one DICOM object."""

    case_key: str
    image: Image.Image
    sex: float
    sex_label: str


def normalize_sex(value: object) -> tuple[float, str]:
    """Map DICOM sex values to the binary contract used during model training."""
    normalized = str(value).strip().upper()
    if normalized in {"F", "FEMALE", "0"}:
        return 0.0, "F"
    if normalized in {"M", "MALE", "1"}:
        return 1.0, "M"
    raise ValueError("DICOM PatientSex must be M or F for this model")


def _safe_case_key(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return f"case_{hashlib.sha256(raw).hexdigest()[:16]}"


def dicom_dataset_to_uint8(dataset: Dataset) -> np.ndarray:
    """Decode one monochrome DICOM frame into the project's 8-bit intensity contract."""
    try:
        pixels = np.asarray(dataset.pixel_array)
    except Exception as exc:
        raise ValueError("Unable to decode DICOM pixel data") from exc

    pixels = np.squeeze(pixels)
    if pixels.ndim != 2:
        raise ValueError(f"Expected one monochrome DICOM frame, got shape={pixels.shape}")
    if min(pixels.shape) < 64:
        raise ValueError(f"DICOM image is too small for inference: shape={pixels.shape}")

    pixels = pixels.astype(np.float32)
    slope = float(dataset.get("RescaleSlope", 1.0))
    intercept = float(dataset.get("RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept
    if not np.isfinite(pixels).all():
        raise ValueError("DICOM pixel data contains non-finite values")

    valid = np.ones(pixels.shape, dtype=bool)
    if "PixelPaddingValue" in dataset:
        padding = float(dataset.PixelPaddingValue) * slope + intercept
        valid &= pixels != padding
    sample = pixels[valid]
    if sample.size < pixels.size * 0.01:
        raise ValueError("DICOM image has insufficient non-padding pixels")
    low, high = np.percentile(sample, [1.0, 99.0])
    if high <= low:
        raise ValueError("DICOM image has no usable intensity range")

    scaled = np.clip((pixels - low) / (high - low), 0.0, 1.0)
    photometric = str(dataset.get("PhotometricInterpretation", "")).upper()
    if photometric == "MONOCHROME1":
        scaled = 1.0 - scaled
    elif photometric != "MONOCHROME2":
        raise ValueError(f"Unsupported DICOM PhotometricInterpretation: {photometric or 'missing'}")
    scaled[~valid] = 0.0
    return np.rint(scaled * 255.0).astype(np.uint8)


def dicom_bytes_to_case(
    payload: bytes,
    *,
    case_key: str,
    sex_override: str | None = None,
) -> DicomCase:
    if not payload or len(payload) > MAX_DICOM_BYTES:
        raise ValueError(f"DICOM payload must be between 1 and {MAX_DICOM_BYTES} bytes")
    try:
        dataset = pydicom.dcmread(io.BytesIO(payload), force=False)
    except Exception as exc:
        raise ValueError("Unable to parse DICOM payload") from exc
    sex, sex_label = normalize_sex(
        sex_override if sex_override is not None else dataset.get("PatientSex", "")
    )
    image = Image.fromarray(dicom_dataset_to_uint8(dataset), mode="L")
    return DicomCase(case_key=case_key, image=image, sex=sex, sex_label=sex_label)


def load_dicom_file(path: Path, *, sex_override: str | None = None) -> DicomCase:
    resolved = path.resolve(strict=True)
    size = resolved.stat().st_size
    if size > MAX_DICOM_BYTES:
        raise ValueError(f"DICOM file exceeds the {MAX_DICOM_BYTES}-byte safety limit")
    payload = resolved.read_bytes()
    return dicom_bytes_to_case(
        payload,
        case_key=_safe_case_key(payload),
        sex_override=sex_override,
    )


def iter_zip_dicom_cases(
    path: Path,
    *,
    max_cases: int,
    sex_override: str | None = None,
) -> Iterator[DicomCase]:
    """Read a deterministic privacy-safe sample without extracting the archive."""
    if max_cases < 1:
        raise ValueError("max_cases must be positive")
    resolved = path.resolve(strict=True)
    try:
        archive = zipfile.ZipFile(resolved)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("Unable to open ZIP archive") from exc

    with archive:
        entries = [
            entry
            for entry in archive.infolist()
            if not entry.is_dir() and entry.filename.lower().endswith(".dcm")
        ]
        if not entries:
            raise ValueError("ZIP archive contains no .dcm entries")
        # Hash ordering avoids putting patient/file names in output while remaining reproducible.
        selected = sorted(entries, key=lambda entry: _safe_case_key(entry.filename))[:max_cases]
        for entry in selected:
            if entry.file_size < 1 or entry.file_size > MAX_DICOM_BYTES:
                raise ValueError("ZIP contains a DICOM entry outside the allowed size range")
            try:
                with archive.open(entry) as handle:
                    payload = handle.read(MAX_DICOM_BYTES + 1)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise ValueError("Unable to read a DICOM entry from ZIP") from exc
            if len(payload) != entry.file_size:
                raise ValueError("DICOM ZIP entry size mismatch")
            yield dicom_bytes_to_case(
                payload,
                case_key=_safe_case_key(entry.filename),
                sex_override=sex_override,
            )
