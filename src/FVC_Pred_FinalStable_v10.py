# =============================================================================
# OSIC FVC PROGRESSION - FINAL STABLE v10 (FROZEN v9 PREDICTOR + ROBUST EVALUATION)
# =============================================================================
# Design principle:
#   Start from the best observed 10-fold ablation configuration:
#       Direct Delta-FVC + lower-lung focus + FSS + trainable attention
#   and ADD robust experts/calibration without perturbing the mean predictor.
#
# Key differences from the failed SOTA-targeted rewrite:
#   - DenseNet169 only (the empirically strong backbone in this project)
#   - Lower-lung feature bank is the primary neural expert
#   - Direct Delta-FVC is the primary target
#   - Huber-only training for the mean neural predictor
#   - NO joint Laplace/quantile/sigma loss on the shared FVC representation
#   - NO patient-balanced sampler for a row-weighted target metric
#   - NO multi-backbone 2945-D slice vectors
#   - NO multi-window 3-channel reinterpretation of ImageNet channels
#   - Optional whole-relevant-lung expert for diversity, not as a replacement
#   - Independent patient-slope neural expert (SOTA-inspired)
#   - Ridge + GradientBoosting + ExtraTrees clinical experts
#   - Inner-validation convex blending; outer fold remains untouched
#   - Mean prediction and uncertainty calibration are fully decoupled
#   - Uncertainty candidates include constant sigma, a paper-inspired
#     Percent/slope confidence, and a ridge residual model; best is selected
#     on INNER validation only.
#   - Automated lung-bearing slice-range detection approximates the paper's
#     manual removal of irrelevant superior/inferior slices.
#   - Reports strict future-visit OOF metrics and a separate patient-trajectory
#     metric that includes the known baseline observation (closer to the
#     published paper/repository evaluation convention).
#
# FINAL-STABLE PRINCIPLE:
#   The v9 mean predictor, balanced patient folds, strict CT-baseline protocol,
#   rapid-decline loss, and stabilized uncertainty calibration are FROZEN.
#   v10 changes evaluation/reporting only:
#       IMPORTANT (enabled): patient-cluster bootstrap CIs + per-patient metrics
#       OPTIONAL (disabled): repeated balanced 5-fold CV
#       OPTIONAL (disabled): leave-one-patient-out CV
#
# This avoids tuning the neural architecture further to a 29-patient strict
# =============================================================================

import os
import math
import json
import pickle
import random
import warnings
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import binary_fill_holes
from scipy.optimize import minimize, minimize_scalar
from skimage import morphology
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# Configuration
# =============================================================================
class Config:
    data_path = "/kaggle/input/datasets/qingtaozhou/osic-pulmonaryfibrosisprogression/"

    # Match the user's latest evaluation first.
    n_folds = 5
    seed = 42
    inner_val_fraction = 0.18

    # CT / slice selection
    max_slices_per_patient = 20
    lower_lung_ratio = 0.60
    # A slice is considered lung-bearing if its segmented lung area is at least
    # this fraction of the maximum area in that patient's stack.
    lung_bearing_area_fraction = 0.12
    lung_bearing_min_pixels = 500

    # Feature extraction
    backbone = "dn169"
    feature_batch_size = 16
    use_feature_cache = True
    feature_cache_path = "/kaggle/working/osic_recovery_v3_features.pkl"

    # Preprocessing: intentionally close to the corrected ablation that gave
    # the best prior result.
    apply_hu_conversion = True
    apply_clahe = True
    lung_window_center = -600
    lung_window_width = 1500
    lung_hu_low = -950
    lung_hu_high = -250
    clahe_clip_limit = 2.0
    clahe_tile_grid_size = (8, 8)

    # FSS / CT severity proxy
    fss_column = "FSS"
    allow_ct_fss_proxy = True
    fss_proxy_hu_low = -700
    fss_proxy_hu_high = -250

    # Direct Delta-FVC neural model: recover the empirically successful regime.
    n_epochs = 100
    min_epochs = 12
    early_stopping_patience = 18
    batch_size = 20
    learning_rate = 1.8e-5
    weight_decay = 5e-5
    grad_clip_norm = 1.0
    num_workers = 2
    direct_seeds = (42, 123)


    # -----------------------------------------------------------------
    # v6: targeted changes based on the v5 catastrophic-error audit
    # -----------------------------------------------------------------
    # 65.8% of >=500 mL errors were FVC over-predictions, i.e. the model
    # underestimated decline. We therefore keep Huber as the dominant loss
    # but add a small MSE component so large residuals are not almost ignored.
    direct_huber_weight = 0.85
    direct_mse_weight = 0.15

    # Training-only rapid-decliner weighting. PatientSlope is computed only
    # for training patients. The most rapidly declining training patients
    # receive at most a modest 1.40x loss weight.
    rapid_decline_quantile = 0.20
    rapid_decline_weight_strength = 0.40
    rapid_decline_weight_cap = 1.40

    # Epoch selection now considers RMSE as well as MAE. This is deliberately
    # modest so the strong v3 MAE behavior remains dominant.
    epoch_select_mae_weight = 0.70
    epoch_select_rmse_weight = 0.30

    # IMPORTANT: disable quantile-derived absolute FVC clipping. The v5 audit
    # found a baseline-FVC=6399 mL patient whose predictions were all forced
    # to 5004.65 mL by the training-target upper clip. Only broad physiological
    # sanity bounds are retained.
    use_dynamic_fvc_clip = False

    # Diagnostics
    catastrophic_error_ml = 500.0
    tail_fraction = 0.15


    # -----------------------------------------------------------------
    # v8 strict CT-baseline protocol
    # -----------------------------------------------------------------
    # OSIC Weeks are relative to the baseline CT scan (CT = Week 0).
    # PRIMARY protocol:
    #   1) use exact Week 0 FVC if available;
    #   2) otherwise use the closest FVC at or BEFORE CT;
    #   3) NEVER use a post-CT FVC as the primary baseline;
    #   4) predict only strictly post-CT targets.
    ct_week_zero = 0.0
    strict_baseline_must_be_pre_or_at_ct = True
    min_target_week_from_ct = 0.0  # targets require target_week > 0

    # Patients without any FVC at/before CT are excluded from the STRICT
    # prognostic cohort before GroupKFold. They can be described separately
    # under the dynamic-follow-up protocol, but must not leak into the primary
    # CT-baseline comparison.
    exclude_without_pre_ct_fvc = True

    # Build a separate dynamic-follow-up RECORD manifest for transparency.
    # This does NOT affect or mix with the primary strict model/evaluation.
    build_dynamic_secondary_manifest = True

    # Keep the successful v6 mean-predictor changes.
    # Uncertainty remains the low-capacity v3 form.
    uncertainty_use_v3_features = True


    # -----------------------------------------------------------------
    # v9 patient-balanced CV
    # -----------------------------------------------------------------
    # The v8 strict cohort was small enough that plain GroupKFold produced
    # extremely different validation difficulty (fold MAE ~76 to ~251 mL).
    # Build folds at PATIENT level and greedily balance clinical/progression
    # characteristics and target counts across folds.
    balanced_outer_folds = True
    balanced_inner_folds = True
    inner_balanced_splits = 5
    balance_rapid_decline_quantile = 0.25

    # Relative importance in the patient-level balancing objective.
    balance_weight_patient_count = 1.5
    balance_weight_target_count = 1.8
    balance_weight_baseline_fvc = 1.0
    balance_weight_percent = 1.0
    balance_weight_baseline_week = 0.8
    balance_weight_slope = 1.6
    balance_weight_mean_horizon = 1.0
    balance_weight_max_horizon = 1.0
    balance_weight_rapid_status = 1.6


    # -----------------------------------------------------------------
    # v10 FINAL-STABLE evaluation/reporting
    # -----------------------------------------------------------------
    output_prefix = "recovery_v10_final_stable"

    # IMPORTANT / enabled by default:
    # Cluster bootstrap resamples PATIENTS, not visits, so repeated FVC rows
    # from the same patient are never treated as independent observations.
    enable_patient_bootstrap = True
    bootstrap_repetitions = 5000
    bootstrap_confidence = 0.95
    bootstrap_seed = 20260818

    # Save one row per patient with strict OOF error metrics. This is important
    # for identifying whether results depend on a few unusually difficult cases.
    save_per_patient_metrics = True

    # OPTIONAL / expensive:
    # Re-run the exact frozen model with different BALANCED-FOLD seeds.
    # Leave OFF for the normal primary run. Turn on for robustness analysis.
    enable_repeated_balanced_cv = False
    repeated_cv_repeats = 10
    repeated_cv_seed_stride = 1009

    # OPTIONAL / very expensive:
    # Strict leave-one-patient-out CV. With ~29 eligible patients this trains
    # roughly 29 outer folds; use only for a dedicated robustness experiment.
    enable_lopo_cv = False

    # Do not bootstrap every optional retraining run; aggregate those runs
    # separately. This keeps optional analyses computationally manageable.
    bootstrap_optional_runs = False

    # -----------------------------------------------------------------
    # v9 stabilized uncertainty
    # -----------------------------------------------------------------
    # Variable sigma models are shrunk toward the fold's robust constant
    # sigma and cannot become strongly overconfident on a small inner set.
    uncertainty_shrink_to_constant = 0.65
    uncertainty_floor_fraction_of_constant = 0.70
    uncertainty_candidate_min_lll_gain = 0.005

    # Narrower scale search than v8's 0.35..5.0.
    sigma_scale_min = 0.75
    sigma_scale_max = 1.50
    sigma_scale_steps = 180
    constant_sigma_min = 70.0
    constant_sigma_max = 500.0
    constant_sigma_steps = 300

    # Optional experts. Lower-lung direct expert is always present.
    use_whole_relevant_lung_expert = True
    use_slope_neural_expert = True
    use_tabular_experts = True

    # Blend objective. A small RMSE term suppresses catastrophic outliers while
    # MAE remains dominant.
    blend_mae_weight = 0.65
    blend_rmse_weight = 0.35
    blend_regularization = 0.002

    # Predictions are clipped only using OUTER-TRAIN target distribution.
    # This prevents rare pathological extrapolation without touching labels in
    # the outer validation fold.
    dynamic_clip_margin_ml = 250.0
    absolute_fvc_min = 500.0
    absolute_fvc_max = 7000.0

    # Laplace / confidence
    laplace_sigma_floor = 70.0
    laplace_delta_cap = 1000.0
    uncertainty_ridge_alpha = 8.0

    # Published reference (Dolatabadi et al., 2026).
    sota_lll = -6.508
    sota_rmse = 168.26


config = Config()


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Metrics
# =============================================================================
def compute_laplace_score(y_true, y_pred, sigma, cfg):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim == 0:
        sigma = np.full_like(y_true, float(sigma))
    delta = np.minimum(np.abs(y_true - y_pred), cfg.laplace_delta_cap)
    sigma = np.maximum(sigma, cfg.laplace_sigma_floor)
    return float(np.mean(-(
        np.sqrt(2.0) * delta / sigma
        + np.log(np.sqrt(2.0) * sigma)
    )))


def metric_bundle(y_true, y_pred, sigma, cfg):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_true - y_pred
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "LLL": compute_laplace_score(y_true, y_pred, sigma, cfg),
        "MeanSigma": float(np.mean(np.maximum(np.asarray(sigma), cfg.laplace_sigma_floor))),
    }


# =============================================================================
# DICOM / CT helpers
# =============================================================================
def _safe_slice_position(path):
    try:
        dcm = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        if hasattr(dcm, "ImagePositionPatient") and len(dcm.ImagePositionPatient) >= 3:
            return (0, float(dcm.ImagePositionPatient[2]))
        if hasattr(dcm, "SliceLocation"):
            return (1, float(dcm.SliceLocation))
        if hasattr(dcm, "InstanceNumber"):
            return (2, float(dcm.InstanceNumber))
    except Exception:
        pass
    return (3, os.path.basename(path))


def _uniform_subsample(indices, max_n):
    indices = list(indices)
    if len(indices) <= max_n:
        return indices
    pts = np.linspace(0, len(indices) - 1, max_n)
    pts = np.unique(np.round(pts).astype(int))
    return [indices[i] for i in pts]


def _read_hu(path, cfg):
    dcm = pydicom.dcmread(path, force=True)
    px = dcm.pixel_array.astype(np.float32)
    if cfg.apply_hu_conversion:
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        return px * slope + intercept
    return px


def _lung_mask_from_hu(hu, cfg):
    mask = (hu > cfg.lung_hu_low) & (hu < cfg.lung_hu_high)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=500)
    mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=64)
    mask = binary_fill_holes(mask).astype(bool)
    return mask


def slice_lung_statistics(path, cfg):
    try:
        hu = _read_hu(path, cfg)
        mask = _lung_mask_from_hu(hu, cfg)
        lung_count = int(mask.sum())
        if lung_count > 0:
            fib_count = int(np.sum(
                (hu >= cfg.fss_proxy_hu_low)
                & (hu <= cfg.fss_proxy_hu_high)
                & mask
            ))
        else:
            fib_count = 0
        return lung_count, fib_count
    except Exception:
        return 0, 0


def preprocess_single_slice(path, cfg):
    try:
        hu = _read_hu(path, cfg)
        mask = _lung_mask_from_hu(hu, cfg)
        lung_count = int(mask.sum())

        lo = cfg.lung_window_center - cfg.lung_window_width / 2.0
        hi = cfg.lung_window_center + cfg.lung_window_width / 2.0
        img = np.clip(hu, lo, hi)
        img = (img - lo) / (hi - lo + 1e-8)

        if lung_count > 0:
            img = img * mask.astype(np.float32)

        if cfg.apply_clahe:
            u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(
                clipLimit=cfg.clahe_clip_limit,
                tileGridSize=cfg.clahe_tile_grid_size,
            )
            img = clahe.apply(u8).astype(np.float32) / 255.0
            if lung_count > 0:
                img *= mask.astype(np.float32)

        return img.astype(np.float32)
    except Exception as exc:
        print(f"Warning: failed to preprocess {path}: {exc}")
        return None


feature_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# =============================================================================
# DenseNet169 frozen feature extractor
# =============================================================================
class FeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if model_name == "dn169":
            self.backbone = models.densenet169(weights="DEFAULT")
            self.backbone.classifier = nn.Identity()
            self.feature_dim = 1664
        elif model_name == "dn121":
            self.backbone = models.densenet121(weights="DEFAULT")
            self.backbone.classifier = nn.Identity()
            self.feature_dim = 1024
        else:
            raise ValueError(f"Unsupported recovery backbone: {model_name}")
        self.backbone = self.backbone.to(device).eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_features(self, batch):
        batch = batch.to(device, non_blocking=True)
        z = self.backbone(batch)
        return z.detach().cpu().numpy().astype(np.float32)


def _images_to_features(images, extractor, cfg):
    if not images:
        return np.zeros((0, extractor.feature_dim), dtype=np.float32)
    tensors = []
    for img in images:
        rgb = np.stack([img, img, img], axis=-1)
        tensors.append(feature_transform(rgb))
    chunks = []
    for i in range(0, len(tensors), cfg.feature_batch_size):
        batch = torch.stack(tensors[i:i + cfg.feature_batch_size], dim=0)
        chunks.append(extractor.extract_features(batch))
    return np.concatenate(chunks, axis=0)


def _cache_signature(cfg):
    return json.dumps({
        "version": 4,
        "backbone": cfg.backbone,
        "max_slices": cfg.max_slices_per_patient,
        "lower_ratio": cfg.lower_lung_ratio,
        "bearing_fraction": cfg.lung_bearing_area_fraction,
        "window": [cfg.lung_window_center, cfg.lung_window_width],
        "lung_hu": [cfg.lung_hu_low, cfg.lung_hu_high],
        "clahe": cfg.apply_clahe,
    }, sort_keys=True)


def build_feature_banks(cfg, extractor):
    """
    Produces:
      lower_bank: inferior 60% of automatically detected lung-bearing range
      full_bank : whole automatically detected lung-bearing range
      fss_proxy : density fraction over lung-bearing slices

    Detecting lung-bearing boundaries avoids wasting the fixed slice budget on
    abdominal/neck slices and is the automatic analogue of manually selecting
    relevant CT bounds.
    """
    signature = _cache_signature(cfg)
    if cfg.use_feature_cache and os.path.exists(cfg.feature_cache_path):
        try:
            with open(cfg.feature_cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("signature") == signature:
                print(f"Loading cached recovery features: {cfg.feature_cache_path}")
                return cache["lower_bank"], cache["full_bank"], cache["fss_proxy"]
        except Exception as exc:
            print(f"Ignoring feature cache: {exc}")

    train_dir = os.path.join(cfg.data_path, "train")
    patient_ids = sorted([
        p for p in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, p))
    ])

    lower_bank, full_bank, fss_proxy = {}, {}, {}

    for pid in tqdm(patient_ids, desc="Recovery CT feature banks"):
        pdir = os.path.join(train_dir, pid)
        paths = [
            os.path.join(pdir, f) for f in os.listdir(pdir)
            if f.lower().endswith(".dcm")
        ]
        if not paths:
            continue
        paths = sorted(paths, key=_safe_slice_position)

        # First lightweight pass: determine actual lung-bearing range.
        stats = [slice_lung_statistics(p, cfg) for p in paths]
        areas = np.asarray([s[0] for s in stats], dtype=np.float64)
        max_area = float(areas.max()) if len(areas) else 0.0

        if max_area > 0:
            threshold = max(
                cfg.lung_bearing_min_pixels,
                cfg.lung_bearing_area_fraction * max_area,
            )
            good = np.where(areas >= threshold)[0]
        else:
            good = np.asarray([], dtype=int)

        if len(good) >= 2:
            # Use first-to-last relevant slice to preserve a contiguous stack.
            relevant = list(range(int(good[0]), int(good[-1]) + 1))
        else:
            relevant = list(range(len(paths)))

        full_idx = _uniform_subsample(relevant, cfg.max_slices_per_patient)
        n_lower = max(1, int(math.ceil(len(relevant) * cfg.lower_lung_ratio)))
        # In DICOM patient coordinates, lower z is generally inferior for axial CT.
        lower_candidates = relevant[:n_lower]
        lower_idx = _uniform_subsample(lower_candidates, cfg.max_slices_per_patient)

        needed = sorted(set(full_idx) | set(lower_idx))
        processed = {}
        for idx in needed:
            img = preprocess_single_slice(paths[idx], cfg)
            if img is not None:
                processed[idx] = img

        full_images = [processed[i] for i in full_idx if i in processed]
        lower_images = [processed[i] for i in lower_idx if i in processed]
        full_bank[pid] = _images_to_features(full_images, extractor, cfg)
        lower_bank[pid] = _images_to_features(lower_images, extractor, cfg)

        # FSS proxy over all detected lung-bearing slices from first-pass stats.
        lung_total = sum(stats[i][0] for i in relevant)
        fib_total = sum(stats[i][1] for i in relevant)
        fss_proxy[pid] = float(fib_total / lung_total) if lung_total > 0 else 0.0

    if cfg.use_feature_cache:
        try:
            os.makedirs(os.path.dirname(cfg.feature_cache_path), exist_ok=True)
            with open(cfg.feature_cache_path, "wb") as f:
                pickle.dump({
                    "signature": signature,
                    "lower_bank": lower_bank,
                    "full_bank": full_bank,
                    "fss_proxy": fss_proxy,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved feature cache: {cfg.feature_cache_path}")
        except Exception as exc:
            print(f"Could not save feature cache: {exc}")

    return lower_bank, full_bank, fss_proxy


# =============================================================================
# Longitudinal records
# =============================================================================
def patient_ols_slope(group):
    x = group["Weeks"].to_numpy(dtype=np.float64)
    y = group["FVC"].to_numpy(dtype=np.float64)
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0
    xc = x - x.mean()
    return float(np.sum(xc * (y - y.mean())) / max(np.sum(xc ** 2), 1e-8))


def _select_strict_pre_ct_baseline(group, cfg):
    """
    PRIMARY v8 baseline rule.

    Return the most recent FVC measurement available at or before the baseline
    CT (Week <= 0). Therefore:
      - exact Week 0 wins when present;
      - otherwise choose the largest negative Week;
      - post-CT FVC is NEVER allowed as the strict baseline.

    Returns None if the patient has no FVC measurement at/before CT.
    """
    g = group.copy()
    g["Weeks"] = pd.to_numeric(g["Weeks"], errors="coerce")
    g = g[np.isfinite(g["Weeks"])].copy()

    eligible = g[g["Weeks"] <= float(cfg.ct_week_zero)].copy()
    if len(eligible) == 0:
        return None

    # The maximum eligible week is the closest available observation to CT
    # without crossing into the future.
    best_week = float(eligible["Weeks"].max())
    tied = eligible[np.isclose(eligible["Weeks"], best_week)].copy()

    # Deterministic handling of duplicate-week rows.
    tied = tied.reset_index(drop=True)
    return tied.iloc[0]


def strict_eligible_patients(df, cfg):
    """
    Identify the exact patient cohort eligible for the primary strict protocol
    BEFORE cross-validation splitting.
    """
    eligible = []
    excluded = []

    for pid, group in df.groupby("Patient"):
        group = group.sort_values("Weeks").reset_index(drop=True)
        baseline = _select_strict_pre_ct_baseline(group, cfg)

        if baseline is None:
            excluded.append((pid, "no_FVC_at_or_before_CT"))
            continue

        baseline_week = float(baseline["Weeks"])
        # Need at least one true future target after CT and after the anchor.
        target_mask = (
            (group["Weeks"].astype(float) > float(cfg.min_target_week_from_ct))
            & (group["Weeks"].astype(float) > baseline_week)
        )

        if not bool(np.any(target_mask)):
            excluded.append((pid, "no_post_CT_target"))
            continue

        eligible.append(pid)

    excluded_df = pd.DataFrame(
        excluded, columns=["Patient", "ExclusionReason"]
    )
    return eligible, excluded_df


def patient_forward_ols_slope(group, baseline_row, cfg):
    """
    Training-only future progression slope.

    Uses only:
      - the strict pre-/at-CT anchor, and
      - post-CT observations after that anchor.

    Pre-CT history older than the chosen anchor is excluded.
    """
    baseline_week = float(baseline_row["Weeks"])

    forward = group[
        (group["Weeks"].astype(float) > float(cfg.min_target_week_from_ct))
        & (group["Weeks"].astype(float) > baseline_week)
    ].copy()

    anchor = pd.DataFrame([baseline_row])
    slope_group = pd.concat([anchor, forward], ignore_index=True)
    slope_group = (
        slope_group.sort_values("Weeks")
        .drop_duplicates(subset=["Weeks"], keep="first")
        .reset_index(drop=True)
    )

    if len(slope_group) < 2:
        return 0.0
    return patient_ols_slope(slope_group)


def build_records(df, fss_proxy, cfg):
    """
    Build the PRIMARY strict CT-baseline prognosis records.

    Every returned row is guaranteed to satisfy:
        BaselineWeek <= 0
        TargetWeek > 0
        TargetWeek > BaselineWeek

    The baseline FVC was therefore available no later than CT acquisition.
    """
    rows = []
    has_true_fss = (
        cfg.fss_column in df.columns
        and pd.to_numeric(
            df[cfg.fss_column], errors="coerce"
        ).notna().any()
    )
    has_smoking = "SmokingStatus" in df.columns

    for pid, group in df.groupby("Patient"):
        group = group.sort_values("Weeks").reset_index(drop=True)
        if len(group) < 2:
            continue

        baseline = _select_strict_pre_ct_baseline(group, cfg)
        if baseline is None:
            continue

        baseline_fvc = float(baseline["FVC"])
        baseline_week = float(baseline["Weeks"])

        # Safety assertion: primary protocol must never use future FVC as anchor.
        if baseline_week > float(cfg.ct_week_zero) + 1e-9:
            raise RuntimeError(
                f"Strict protocol violation for {pid}: "
                f"BaselineWeek={baseline_week} > CT Week 0"
            )

        slope = patient_forward_ols_slope(group, baseline, cfg)

        if has_true_fss:
            fss = pd.to_numeric(
                pd.Series([baseline[cfg.fss_column]]),
                errors="coerce"
            ).iloc[0]
            if pd.isna(fss):
                fss = fss_proxy.get(pid, 0.0)
        else:
            fss = fss_proxy.get(pid, 0.0)

        smoking = (
            str(baseline["SmokingStatus"])
            if has_smoking else "Unknown"
        )

        for _, cur in group.iterrows():
            target_week = float(cur["Weeks"])

            # PRIMARY strict prognosis targets are strictly post-CT.
            if target_week <= float(cfg.min_target_week_from_ct):
                continue

            weeks_from_baseline = target_week - baseline_week
            if weeks_from_baseline <= 0:
                continue

            target = float(cur["FVC"])
            delta = target - baseline_fvc
            target_week_from_ct = (
                target_week - float(cfg.ct_week_zero)
            )

            rows.append({
                "Patient": pid,
                "Protocol": "strict_CT_baseline",
                "Age": float(baseline["Age"]),
                "Sex": str(baseline["Sex"]),
                "SmokingStatus": smoking,
                "Percent": float(baseline["Percent"]),

                "BaselineWeek": baseline_week,
                "BaselineWeekFromCT": (
                    baseline_week - float(cfg.ct_week_zero)
                ),
                "BaselineFVC": baseline_fvc,

                "WeeksFromBaseline": weeks_from_baseline,
                "WeeksDelta": weeks_from_baseline,
                "LogWeeksDelta": math.log1p(
                    max(weeks_from_baseline, 0.0)
                ),

                "TargetWeek": target_week,
                "TargetWeekFromCT": target_week_from_ct,
                "LogTargetWeekFromCT": math.log1p(
                    max(target_week_from_ct, 0.0)
                ),

                "FSS": float(fss),
                "TargetFVC": target,
                "DeltaFVC": delta,
                "PatientSlope": slope,
            })

    records = pd.DataFrame(rows)

    if len(records):
        if (records["BaselineWeek"] > float(cfg.ct_week_zero) + 1e-9).any():
            raise RuntimeError(
                "STRICT protocol error: post-CT baseline found in records."
            )
        if (records["TargetWeek"] <= float(cfg.min_target_week_from_ct)).any():
            raise RuntimeError(
                "STRICT protocol error: non-post-CT target found."
            )

    return records, has_true_fss, has_smoking


def build_dynamic_followup_records(df, fss_proxy, cfg):
    """
    SECONDARY dynamic-follow-up manifest.

    For each post-CT target, use the most recent FVC measurement STRICTLY BEFORE
    that target as the clinical anchor. This may be post-CT and is therefore a
    different task from the primary strict CT-baseline prognosis.

    These records are kept completely separate and are NOT used by the primary
    v8 model, folds, metrics, calibration, or SOTA comparison.
    """
    rows = []
    has_true_fss = (
        cfg.fss_column in df.columns
        and pd.to_numeric(
            df[cfg.fss_column], errors="coerce"
        ).notna().any()
    )
    has_smoking = "SmokingStatus" in df.columns

    for pid, group in df.groupby("Patient"):
        group = group.sort_values("Weeks").reset_index(drop=True)
        if len(group) < 2:
            continue

        for target_idx, cur in group.iterrows():
            target_week = float(cur["Weeks"])
            if target_week <= float(cfg.min_target_week_from_ct):
                continue

            prior = group[group["Weeks"].astype(float) < target_week].copy()
            if len(prior) == 0:
                continue

            # Latest information genuinely available before this target.
            anchor = prior.sort_values("Weeks").iloc[-1]
            anchor_week = float(anchor["Weeks"])
            anchor_fvc = float(anchor["FVC"])
            weeks_from_anchor = target_week - anchor_week
            if weeks_from_anchor <= 0:
                continue

            if has_true_fss:
                fss = pd.to_numeric(
                    pd.Series([anchor[cfg.fss_column]]),
                    errors="coerce"
                ).iloc[0]
                if pd.isna(fss):
                    fss = fss_proxy.get(pid, 0.0)
            else:
                fss = fss_proxy.get(pid, 0.0)

            smoking = (
                str(anchor["SmokingStatus"])
                if has_smoking else "Unknown"
            )

            rows.append({
                "Patient": pid,
                "Protocol": "dynamic_followup",
                "Age": float(anchor["Age"]),
                "Sex": str(anchor["Sex"]),
                "SmokingStatus": smoking,
                "Percent": float(anchor["Percent"]),
                "BaselineWeek": anchor_week,
                "BaselineFVC": anchor_fvc,
                "TargetWeek": target_week,
                "TargetWeekFromCT": (
                    target_week - float(cfg.ct_week_zero)
                ),
                "WeeksFromBaseline": weeks_from_anchor,
                "TargetFVC": float(cur["FVC"]),
                "DeltaFVC": float(cur["FVC"]) - anchor_fvc,
                "FSS": float(fss),
            })

    return pd.DataFrame(rows)



def patient_balance_summary(records, cfg):
    """
    One row per patient for fold balancing.

    PatientSlope is used only to construct comparable evaluation folds; it is
    never supplied from validation patients to the model. Regression
    stratification on outcome severity is analogous to class stratification
    and must be reported transparently in the methods.
    """
    if len(records) == 0:
        return pd.DataFrame()

    p = (
        records.groupby("Patient")
        .agg(
            NTargets=("TargetFVC", "size"),
            BaselineFVC=("BaselineFVC", "first"),
            Percent=("Percent", "first"),
            BaselineWeek=("BaselineWeek", "first"),
            PatientSlope=("PatientSlope", "first"),
            MeanHorizon=("TargetWeekFromCT", "mean"),
            MaxHorizon=("TargetWeekFromCT", "max"),
        )
        .reset_index()
    )

    slope_q = float(
        np.quantile(
            p["PatientSlope"].to_numpy(dtype=np.float64),
            cfg.balance_rapid_decline_quantile,
        )
    )
    p["RapidDecliner"] = (
        p["PatientSlope"].to_numpy(dtype=np.float64) <= slope_q
    ).astype(float)
    return p


def make_balanced_patient_folds(records, n_splits, cfg, seed):
    """
    Deterministic greedy patient-level fold construction.

    The objective balances:
      patient count, target count, baseline FVC, Percent, baseline week,
      forward progression slope, mean/max CT horizon, and rapid-decliner rate.

    Returns a list of dictionaries with patient sets and fold summaries.
    """
    p = patient_balance_summary(records, cfg)
    n_patients = len(p)
    if n_patients < n_splits:
        raise RuntimeError(
            f"Cannot make {n_splits} balanced folds from {n_patients} patients."
        )

    numeric_cols = [
        "NTargets", "BaselineFVC", "Percent", "BaselineWeek",
        "PatientSlope", "MeanHorizon", "MaxHorizon",
    ]
    X = p[numeric_cols].to_numpy(dtype=np.float64)

    # Standardize patient characteristics so units do not dominate.
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd

    weights = np.array([
        cfg.balance_weight_target_count,
        cfg.balance_weight_baseline_fvc,
        cfg.balance_weight_percent,
        cfg.balance_weight_baseline_week,
        cfg.balance_weight_slope,
        cfg.balance_weight_mean_horizon,
        cfg.balance_weight_max_horizon,
    ], dtype=np.float64)

    # Center rapid status; include an explicit patient-count dimension.
    rapid = p["RapidDecliner"].to_numpy(dtype=np.float64)
    rapid_centered = rapid - np.mean(rapid)

    V = np.column_stack([
        np.ones(n_patients) * cfg.balance_weight_patient_count,
        Z * weights.reshape(1, -1),
        rapid_centered * cfg.balance_weight_rapid_status,
    ])

    desired = np.sum(V, axis=0) / float(n_splits)
    fold_sums = np.zeros((n_splits, V.shape[1]), dtype=np.float64)
    fold_members = [[] for _ in range(n_splits)]

    # Hard/eccentric patients are assigned first. Seed is used only to break
    # exact/near ties deterministically.
    rng = np.random.default_rng(seed)
    extremeness = np.linalg.norm(V[:, 1:], axis=1)
    jitter = rng.uniform(0.0, 1e-6, size=n_patients)
    order = np.argsort(-(extremeness + jitter))

    # Seed each fold with one difficult patient.
    for f, idx in enumerate(order[:n_splits]):
        fold_members[f].append(int(idx))
        fold_sums[f] += V[idx]

    for idx in order[n_splits:]:
        costs = []
        for f in range(n_splits):
            trial = fold_sums.copy()
            trial[f] += V[idx]

            # Global imbalance after tentative assignment.
            imbalance = np.sum((trial - desired.reshape(1, -1)) ** 2)

            # Small extra penalty for patient-count spread.
            counts = np.array(
                [len(m) + (1 if k == f else 0)
                 for k, m in enumerate(fold_members)],
                dtype=np.float64,
            )
            count_penalty = 0.25 * np.var(counts)
            costs.append(float(imbalance + count_penalty))

        best_cost = np.min(costs)
        best_candidates = [
            f for f, c in enumerate(costs)
            if np.isclose(c, best_cost, rtol=1e-10, atol=1e-10)
        ]
        # Prefer the fold with fewer patients if tied.
        best = min(best_candidates, key=lambda f: len(fold_members[f]))
        fold_members[best].append(int(idx))
        fold_sums[best] += V[idx]

    result = []
    all_patients = set(p["Patient"].astype(str).tolist())

    for f, member_idx in enumerate(fold_members):
        val_patients = set(
            p.iloc[member_idx]["Patient"].astype(str).tolist()
        )
        train_patients = all_patients - val_patients

        sub = p[p["Patient"].astype(str).isin(val_patients)].copy()
        summary = {
            "Fold": f + 1,
            "Patients": int(len(sub)),
            "Targets": int(sub["NTargets"].sum()),
            "MeanBaselineFVC": float(sub["BaselineFVC"].mean()),
            "MeanPercent": float(sub["Percent"].mean()),
            "MeanBaselineWeek": float(sub["BaselineWeek"].mean()),
            "MeanSlope": float(sub["PatientSlope"].mean()),
            "RapidRate": float(sub["RapidDecliner"].mean()),
            "MeanHorizon": float(
                np.average(
                    sub["MeanHorizon"],
                    weights=np.maximum(sub["NTargets"], 1)
                )
            ),
        }
        result.append({
            "train_patients": train_patients,
            "val_patients": val_patients,
            "summary": summary,
        })

    return result


def balanced_inner_split(train_records, cfg, seed, outer_fold):
    """
    Choose one balanced patient fold as inner validation.
    """
    n_patients = train_records["Patient"].nunique()
    n_splits = min(cfg.inner_balanced_splits, n_patients)
    n_splits = max(2, n_splits)

    folds = make_balanced_patient_folds(
        train_records, n_splits, cfg, seed
    )
    chosen = (outer_fold - 1) % n_splits
    val_patients = folds[chosen]["val_patients"]

    inner_val = train_records[
        train_records["Patient"].astype(str).isin(val_patients)
    ].reset_index(drop=True)
    inner_train = train_records[
        ~train_records["Patient"].astype(str).isin(val_patients)
    ].reset_index(drop=True)

    return inner_train, inner_val, folds[chosen]["summary"]

def make_patient_table(records):
    if len(records) == 0:
        return records.copy()
    cols = [
        "Patient", "Age", "Sex", "SmokingStatus", "Percent",
        "BaselineWeek", "BaselineFVC", "FSS", "PatientSlope",
    ]
    return records.sort_values(["Patient", "WeeksFromBaseline"]).groupby("Patient", as_index=False)[cols].first()


# =============================================================================
# Direct Delta-FVC neural data/model (kept close to best ablation)
# =============================================================================
def fit_direct_tools(train_records, cfg):
    clinical_cols = ["Age", "Percent", "BaselineWeek", "Sex"]
    clinical = ColumnTransformer([
        ("num", StandardScaler(), ["Age", "Percent", "BaselineWeek"]),
        ("cat", OneHotEncoder(
            drop="first", handle_unknown="ignore", sparse_output=False
        ), ["Sex"]),
    ])
    clinical.fit(train_records[clinical_cols])

    baseline_scaler = StandardScaler().fit(train_records[["BaselineFVC"]])
    weeks_scaler = StandardScaler().fit(train_records[["WeeksFromBaseline"]])
    logweeks_scaler = StandardScaler().fit(train_records[["LogWeeksDelta"]])

    # Explicit CT-relative forecast horizon.
    target_ct_scaler = StandardScaler().fit(
        train_records[["TargetWeekFromCT"]]
    )
    log_target_ct_scaler = StandardScaler().fit(
        train_records[["LogTargetWeekFromCT"]]
    )

    fss_scaler = StandardScaler().fit(train_records[["FSS"]])
    target_scaler = StandardScaler().fit(train_records[["DeltaFVC"]])

    # Training-patient forward-slope distribution only.
    patient_slopes = (
        train_records[["Patient", "PatientSlope"]]
        .drop_duplicates("Patient")["PatientSlope"]
        .to_numpy(dtype=np.float64)
    )
    patient_slopes = patient_slopes[np.isfinite(patient_slopes)]

    if len(patient_slopes) >= 5:
        rapid_threshold = float(
            np.quantile(patient_slopes, cfg.rapid_decline_quantile)
        )
        q10 = float(np.quantile(patient_slopes, 0.10))
        q50 = float(np.quantile(patient_slopes, 0.50))
        rapid_scale = max(abs(q50 - q10), 1.0)
    else:
        rapid_threshold = -5.0
        rapid_scale = 5.0

    return {
        "clinical": clinical,
        "baseline_scaler": baseline_scaler,
        "weeks_scaler": weeks_scaler,
        "logweeks_scaler": logweeks_scaler,
        "target_ct_scaler": target_ct_scaler,
        "log_target_ct_scaler": log_target_ct_scaler,
        "fss_scaler": fss_scaler,
        "target_scaler": target_scaler,
        "rapid_threshold": rapid_threshold,
        "rapid_scale": rapid_scale,
        "rapid_weight_strength": float(cfg.rapid_decline_weight_strength),
        "rapid_weight_cap": float(cfg.rapid_decline_weight_cap),
    }

class DirectDataset(Dataset):
    def __init__(self, records, bank, feature_dim, max_slices, tools):
        self.records = records.reset_index(drop=True)
        self.bank = bank
        self.feature_dim = feature_dim
        self.max_slices = max_slices

        clinical = tools["clinical"].transform(
            self.records[["Age", "Percent", "BaselineWeek", "Sex"]]
        ).astype(np.float32)

        baseline = tools["baseline_scaler"].transform(
            self.records[["BaselineFVC"]]
        ).astype(np.float32)

        weeks_from_baseline = tools["weeks_scaler"].transform(
            self.records[["WeeksFromBaseline"]]
        ).astype(np.float32)

        logweeks = tools["logweeks_scaler"].transform(
            self.records[["LogWeeksDelta"]]
        ).astype(np.float32)

        target_week_from_ct = tools["target_ct_scaler"].transform(
            self.records[["TargetWeekFromCT"]]
        ).astype(np.float32)

        log_target_week_from_ct = tools["log_target_ct_scaler"].transform(
            self.records[["LogTargetWeekFromCT"]]
        ).astype(np.float32)

        fss = tools["fss_scaler"].transform(
            self.records[["FSS"]]
        ).astype(np.float32)

        # v7: CT-relative and FVC-anchor-relative time are no longer conflated.
        self.meta = np.concatenate([
            clinical,
            baseline,
            weeks_from_baseline,
            logweeks,
            target_week_from_ct,
            log_target_week_from_ct,
            fss,
        ], axis=1).astype(np.float32)

        self.target = tools["target_scaler"].transform(
            self.records[["DeltaFVC"]]
        ).astype(np.float32).reshape(-1)

        # v6 rapid-decline weighting retained, but PatientSlope is now the
        # CT-aligned forward slope rather than a slope across all pre/post visits.
        slope = self.records["PatientSlope"].to_numpy(dtype=np.float64)
        severity = np.clip(
            (tools["rapid_threshold"] - slope)
            / max(tools["rapid_scale"], 1e-6),
            0.0, 1.0
        )
        w = 1.0 + tools["rapid_weight_strength"] * severity
        self.sample_weight = np.clip(
            w, 1.0, tools["rapid_weight_cap"]
        ).astype(np.float32)

    @property
    def meta_dim(self):
        return int(self.meta.shape[1])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        feats = self.bank.get(row["Patient"])
        if feats is None or len(feats) == 0:
            feats = np.zeros((1, self.feature_dim), dtype=np.float32)

        feats = feats[:self.max_slices]
        n = len(feats)

        padded = np.zeros(
            (self.max_slices, self.feature_dim), dtype=np.float32
        )
        mask = np.zeros(self.max_slices, dtype=np.bool_)
        padded[:n] = feats
        mask[:n] = True

        return (
            torch.from_numpy(padded),
            torch.from_numpy(mask),
            torch.from_numpy(self.meta[idx]),
            torch.tensor(self.target[idx], dtype=torch.float32),
            torch.tensor(
                float(self.sample_weight[idx]), dtype=torch.float32
            ),
            torch.tensor(float(row["TargetFVC"]), dtype=torch.float32),
            torch.tensor(float(row["BaselineFVC"]), dtype=torch.float32),
        )

class AttentionFusionRegressor(nn.Module):
    def __init__(self, img_dim, meta_dim):
        super().__init__()
        self.slice_attention = nn.Sequential(
            nn.Linear(img_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, 128), nn.LayerNorm(128), nn.ReLU()
        )
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, 128), nn.LayerNorm(128), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def pool(self, feats, mask):
        scores = self.slice_attention(feats).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1)
        return torch.sum(w.unsqueeze(-1) * feats, dim=1)

    def forward(self, feats, mask, meta):
        pooled = self.pool(feats, mask)
        return self.fusion(torch.cat([self.img_proj(pooled), self.meta_proj(meta)], dim=1)).squeeze(-1)


def _direct_loader(ds, cfg, train, seed=0):
    if train:
        gen = torch.Generator()
        gen.manual_seed(seed)
        return DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(), generator=gen,
        )
    return DataLoader(
        ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
    )


def train_direct_epoch(model, loader, optimizer, cfg):
    model.train()
    total, n = 0.0, 0

    for feats, mask, meta, target, sample_weight, _y, _b in loader:
        feats = feats.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sample_weight = sample_weight.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(feats, mask, meta)

        err = pred - target
        abs_err = torch.abs(err)

        # Huber(delta=1) per sample.
        huber = torch.where(
            abs_err <= 1.0,
            0.5 * err ** 2,
            abs_err - 0.5,
        )

        # Small quadratic component keeps catastrophic residuals relevant.
        mse_half = 0.5 * err ** 2
        per_sample = (
            cfg.direct_huber_weight * huber
            + cfg.direct_mse_weight * mse_half
        )

        # Normalize by the sum of weights so overall loss scale stays stable.
        loss = torch.sum(per_sample * sample_weight) / torch.clamp(
            torch.sum(sample_weight), min=1e-6
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()

        total += float(loss.item())
        n += 1

    return total / max(n, 1)


@torch.no_grad()
def predict_direct(model, loader, tools):
    model.eval()
    pred_s, y, baseline = [], [], []

    for feats, mask, meta, _target, _weight, true_fvc, base in loader:
        feats = feats.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)

        pred_s.append(model(feats, mask, meta).cpu().numpy())
        y.append(true_fvc.numpy())
        baseline.append(base.numpy())

    pred_s = np.concatenate(pred_s)
    pred_delta = tools["target_scaler"].inverse_transform(
        pred_s.reshape(-1, 1)
    ).reshape(-1)

    y = np.concatenate(y).astype(np.float64)
    baseline = np.concatenate(baseline).astype(np.float64)
    return y, baseline + pred_delta

def select_direct_epoch(train_records, val_records, bank, feature_dim, cfg, seed):
    set_seed(seed)
    tools = fit_direct_tools(train_records, cfg)

    tr_ds = DirectDataset(
        train_records, bank, feature_dim, cfg.max_slices_per_patient, tools
    )
    va_ds = DirectDataset(
        val_records, bank, feature_dim, cfg.max_slices_per_patient, tools
    )

    tr_loader = _direct_loader(tr_ds, cfg, True, seed)
    va_loader = _direct_loader(va_ds, cfg, False)

    model = AttentionFusionRegressor(feature_dim, tr_ds.meta_dim).to(device)
    opt = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    sch = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.n_epochs
    )

    best_score = float("inf")
    best_state = None
    best_epoch = cfg.min_epochs
    stale = 0

    for epoch in range(cfg.n_epochs):
        train_direct_epoch(model, tr_loader, opt, cfg)
        y, p = predict_direct(model, va_loader, tools)

        mae = float(np.mean(np.abs(y - p)))
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        score = (
            cfg.epoch_select_mae_weight * mae
            + cfg.epoch_select_rmse_weight * rmse
        )

        sch.step()

        if epoch + 1 >= cfg.min_epochs and score < best_score - 1e-6:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        elif epoch + 1 >= cfg.min_epochs:
            stale += 1

        if (
            epoch + 1 >= cfg.min_epochs
            and stale >= cfg.early_stopping_patience
        ):
            break

    if best_state is None:
        best_state = deepcopy(model.state_dict())
        best_epoch = epoch + 1

    model.load_state_dict(best_state)
    y, p = predict_direct(model, va_loader, tools)

    del tr_loader, va_loader, tr_ds, va_ds, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_epoch, y, p

def fit_direct_fixed(train_records, val_records, bank, feature_dim, cfg, seed, epochs):
    set_seed(seed)
    tools = fit_direct_tools(train_records, cfg)
    tr_ds = DirectDataset(train_records, bank, feature_dim, cfg.max_slices_per_patient, tools)
    va_ds = DirectDataset(val_records, bank, feature_dim, cfg.max_slices_per_patient, tools)
    tr_loader = _direct_loader(tr_ds, cfg, True, seed)
    va_loader = _direct_loader(va_ds, cfg, False)
    model = AttentionFusionRegressor(feature_dim, tr_ds.meta_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(epochs), 1))
    for _ in range(int(epochs)):
        train_direct_epoch(model, tr_loader, opt, cfg)
        sch.step()
    y, p = predict_direct(model, va_loader, tools)
    del tr_loader, va_loader, tr_ds, va_ds, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return y, p


# =============================================================================
# Independent patient-level slope neural expert
# =============================================================================
def fit_slope_tools(patient_table):
    num_cols = ["Age", "Percent", "BaselineWeek", "BaselineFVC", "FSS"]
    cat_cols = ["Sex", "SmokingStatus"]
    clinical = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    clinical.fit(patient_table[num_cols + cat_cols])
    target_scaler = StandardScaler().fit(patient_table[["PatientSlope"]])
    return {"clinical": clinical, "num_cols": num_cols, "cat_cols": cat_cols, "target_scaler": target_scaler}


class SlopeDataset(Dataset):
    def __init__(self, table, bank, feature_dim, max_slices, tools):
        self.table = table.reset_index(drop=True)
        self.bank = bank
        self.feature_dim = feature_dim
        self.max_slices = max_slices
        cols = tools["num_cols"] + tools["cat_cols"]
        self.meta = tools["clinical"].transform(self.table[cols]).astype(np.float32)
        self.target = tools["target_scaler"].transform(self.table[["PatientSlope"]]).astype(np.float32).reshape(-1)

    @property
    def meta_dim(self):
        return int(self.meta.shape[1])

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        feats = self.bank.get(row["Patient"])
        if feats is None or len(feats) == 0:
            feats = np.zeros((1, self.feature_dim), dtype=np.float32)
        feats = feats[:self.max_slices]
        n = len(feats)
        padded = np.zeros((self.max_slices, self.feature_dim), dtype=np.float32)
        mask = np.zeros(self.max_slices, dtype=np.bool_)
        padded[:n] = feats
        mask[:n] = True
        return (
            torch.from_numpy(padded), torch.from_numpy(mask),
            torch.from_numpy(self.meta[idx]),
            torch.tensor(self.target[idx], dtype=torch.float32),
        )


def _slope_loader(ds, cfg, train, seed=0):
    if train:
        gen = torch.Generator(); gen.manual_seed(seed)
        return DataLoader(ds, batch_size=min(cfg.batch_size, max(4, len(ds))), shuffle=True,
                          num_workers=cfg.num_workers, generator=gen,
                          pin_memory=torch.cuda.is_available())
    return DataLoader(ds, batch_size=max(8, cfg.batch_size * 2), shuffle=False,
                      num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())


def train_slope_epoch(model, loader, optimizer, cfg):
    model.train(); total = 0.0; n = 0
    for feats, mask, meta, target in loader:
        feats = feats.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(feats, mask, meta)
        loss = torch.mean(torch.abs(pred - target))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        total += float(loss.item()); n += 1
    return total / max(n, 1)


@torch.no_grad()
def predict_slopes(model, loader, tools, table):
    model.eval(); out = []
    for feats, mask, meta, _target in loader:
        feats = feats.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        out.append(model(feats, mask, meta).cpu().numpy())
    z = np.concatenate(out)
    slope = tools["target_scaler"].inverse_transform(z.reshape(-1, 1)).reshape(-1)
    return dict(zip(table["Patient"].tolist(), slope.tolist()))


def slope_map_to_fvc(records, slope_map):
    s = records["Patient"].map(slope_map).fillna(0.0).to_numpy(dtype=np.float64)
    return records["BaselineFVC"].to_numpy(dtype=np.float64) + s * records["WeeksFromBaseline"].to_numpy(dtype=np.float64), s


def select_slope_epoch(train_records, val_records, bank, feature_dim, cfg, seed):
    set_seed(seed)
    tr_tab = make_patient_table(train_records)
    va_tab = make_patient_table(val_records)
    tools = fit_slope_tools(tr_tab)
    tr_ds = SlopeDataset(tr_tab, bank, feature_dim, cfg.max_slices_per_patient, tools)
    va_ds = SlopeDataset(va_tab, bank, feature_dim, cfg.max_slices_per_patient, tools)
    tr_loader = _slope_loader(tr_ds, cfg, True, seed)
    va_loader = _slope_loader(va_ds, cfg, False)
    model = AttentionFusionRegressor(feature_dim, tr_ds.meta_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=cfg.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)

    best_mae = float("inf"); best_state = None; best_epoch = cfg.min_epochs; stale = 0
    for epoch in range(cfg.n_epochs):
        train_slope_epoch(model, tr_loader, opt, cfg)
        slope_map = predict_slopes(model, va_loader, tools, va_tab)
        p, _ = slope_map_to_fvc(val_records, slope_map)
        y = val_records["TargetFVC"].to_numpy(dtype=np.float64)
        mae = float(np.mean(np.abs(y - p)))
        sch.step()
        if epoch + 1 >= cfg.min_epochs and mae < best_mae - 1e-6:
            best_mae = mae; best_state = deepcopy(model.state_dict()); best_epoch = epoch + 1; stale = 0
        elif epoch + 1 >= cfg.min_epochs:
            stale += 1
        if epoch + 1 >= cfg.min_epochs and stale >= cfg.early_stopping_patience:
            break

    if best_state is None:
        best_state = deepcopy(model.state_dict()); best_epoch = epoch + 1
    model.load_state_dict(best_state)
    slope_map = predict_slopes(model, va_loader, tools, va_tab)
    p, s = slope_map_to_fvc(val_records, slope_map)

    del tr_loader, va_loader, tr_ds, va_ds, model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return best_epoch, p, s


def fit_slope_fixed(train_records, val_records, bank, feature_dim, cfg, seed, epochs):
    set_seed(seed)
    tr_tab = make_patient_table(train_records)
    va_tab = make_patient_table(val_records)
    tools = fit_slope_tools(tr_tab)
    tr_ds = SlopeDataset(tr_tab, bank, feature_dim, cfg.max_slices_per_patient, tools)
    va_ds = SlopeDataset(va_tab, bank, feature_dim, cfg.max_slices_per_patient, tools)
    tr_loader = _slope_loader(tr_ds, cfg, True, seed)
    va_loader = _slope_loader(va_ds, cfg, False)
    model = AttentionFusionRegressor(feature_dim, tr_ds.meta_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=cfg.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(epochs), 1))
    for _ in range(int(epochs)):
        train_slope_epoch(model, tr_loader, opt, cfg); sch.step()
    slope_map = predict_slopes(model, va_loader, tools, va_tab)
    p, s = slope_map_to_fvc(val_records, slope_map)
    del tr_loader, va_loader, tr_ds, va_ds, model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return p, s


# =============================================================================
# Tabular experts
# =============================================================================
def tabular_frame(records):
    x = records[[
        "Age", "Percent", "BaselineWeek", "BaselineFVC",
        "WeeksFromBaseline", "LogWeeksDelta",
        "TargetWeekFromCT", "LogTargetWeekFromCT",
        "FSS", "Sex", "SmokingStatus",
    ]].copy()

    # Explicitly distinguish time from FVC anchor vs time from CT.
    x["Weeks_x_Percent"] = x["WeeksFromBaseline"] * x["Percent"]
    x["Weeks_x_FSS"] = x["WeeksFromBaseline"] * x["FSS"]
    x["CTWeek_x_Percent"] = x["TargetWeekFromCT"] * x["Percent"]
    x["CTWeek_x_FSS"] = x["TargetWeekFromCT"] * x["FSS"]
    x["BaselineFVC_per_Percent"] = (
        x["BaselineFVC"] / np.maximum(x["Percent"], 1.0)
    )
    return x

def fit_tabular_experts(train_records, seed):
    X = tabular_frame(train_records)
    y = train_records["DeltaFVC"].to_numpy(dtype=np.float64)
    num_cols = [c for c in X.columns if c not in ["Sex", "SmokingStatus"]]
    cat_cols = ["Sex", "SmokingStatus"]
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    Z = prep.fit_transform(X)

    ridge = Ridge(alpha=18.0)
    ridge.fit(Z, y)

    gbr = GradientBoostingRegressor(
        loss="huber", n_estimators=220, learning_rate=0.025,
        max_depth=2, min_samples_leaf=5, random_state=seed,
    )
    gbr.fit(Z, y)

    extra = ExtraTreesRegressor(
        n_estimators=450, max_depth=10, min_samples_leaf=3,
        max_features=0.85, random_state=seed, n_jobs=-1,
    )
    extra.fit(Z, y)

    return prep, {"Ridge": ridge, "GBR": gbr, "ExtraTrees": extra}


def predict_tabular_experts(prep, models_dict, records):
    Z = prep.transform(tabular_frame(records))
    baseline = records["BaselineFVC"].to_numpy(dtype=np.float64)
    out = {}
    for name, model in models_dict.items():
        out[name] = baseline + np.asarray(model.predict(Z), dtype=np.float64)
    return out


# =============================================================================
# Blending / clipping
# =============================================================================
def training_prediction_bounds(train_records, cfg):
    """
    Broad physiological sanity bounds only.

    v5 diagnostics found that a training-quantile upper clip of ~5004.65 mL
    forced every prediction for a patient with baseline FVC=6399 mL down to
    the clipping boundary. That created 8 catastrophic errors by construction.

    Never derive the final absolute FVC range from fold target quantiles.
    """
    return float(cfg.absolute_fvc_min), float(cfg.absolute_fvc_max)


def clip_predictions(p, bounds):
    return np.clip(np.asarray(p, dtype=np.float64), bounds[0], bounds[1])


def blend_objective_from_pred(y, p, cfg):
    e = y - p
    mae = np.mean(np.abs(e))
    rmse = np.sqrt(np.mean(e ** 2))
    return cfg.blend_mae_weight * (mae / 160.0) + cfg.blend_rmse_weight * (rmse / 210.0)


def fit_convex_blend(y, pred_matrix, cfg):
    y = np.asarray(y, dtype=np.float64)
    P = np.asarray(pred_matrix, dtype=np.float64)
    k = P.shape[1]
    x0 = np.ones(k, dtype=np.float64) / k

    def obj(w):
        p = P @ w
        reg = cfg.blend_regularization * float(np.sum((w - x0) ** 2))
        return blend_objective_from_pred(y, p, cfg) + reg

    result = minimize(
        obj, x0, method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = result.x if result.success else x0
    w = np.maximum(w, 0.0)
    w = w / max(w.sum(), 1e-12)

    base_pred = P @ w
    # A single inner-only bias term can correct systematic fold shift.
    res = minimize_scalar(
        lambda b: blend_objective_from_pred(y, base_pred + b, cfg),
        bounds=(-150.0, 150.0), method="bounded",
    )
    bias = float(res.x) if res.success else 0.0
    return w, bias


# =============================================================================
# Decoupled uncertainty calibration
# =============================================================================
def optimize_sigma_scale(y, pred, sigma_base, cfg):
    scales = np.linspace(cfg.sigma_scale_min, cfg.sigma_scale_max, cfg.sigma_scale_steps)
    scores = [compute_laplace_score(y, pred, np.asarray(sigma_base) * s, cfg) for s in scales]
    j = int(np.argmax(scores))
    return float(scales[j]), float(scores[j])


def uncertainty_features(records, blend_pred, expert_matrix):
    """
    v8 retains the low-capacity v3 uncertainty feature set.

    Mean prediction and uncertainty are deliberately separated. The calibrator
    receives only:
      1) forecast interval from the selected FVC anchor,
      2) expert disagreement,
      3) predicted absolute DeltaFVC magnitude,
      4) baseline Percent.
    """
    disagreement = np.std(expert_matrix, axis=1)
    delta_mag = np.abs(
        blend_pred
        - records["BaselineFVC"].to_numpy(dtype=np.float64)
    )

    return np.column_stack([
        np.log1p(
            np.abs(
                records["WeeksFromBaseline"].to_numpy(dtype=np.float64)
            )
        ),
        np.log1p(disagreement),
        np.log1p(delta_mag),
        records["Percent"].to_numpy(dtype=np.float64) / 100.0,
    ])

def _shrink_sigma_to_constant(variable_sigma, constant_sigma, cfg):
    variable_sigma = np.maximum(
        np.asarray(variable_sigma, dtype=np.float64),
        cfg.laplace_sigma_floor,
    )
    c = max(float(constant_sigma), cfg.laplace_sigma_floor)
    alpha = float(cfg.uncertainty_shrink_to_constant)

    # Log-space shrinkage preserves positivity and is less dominated by large
    # raw sigma values than arithmetic averaging.
    shrunk = np.exp(
        (1.0 - alpha) * np.log(variable_sigma)
        + alpha * np.log(c)
    )

    # A fold-specific robust floor prevents a small inner set from producing
    # pathological overconfidence such as v8 fold-3 mean sigma ~95 mL.
    robust_floor = max(
        cfg.laplace_sigma_floor,
        cfg.uncertainty_floor_fraction_of_constant * c,
    )
    return np.maximum(shrunk, robust_floor)


def fit_uncertainty_calibrator(records, y, pred, expert_matrix, slope_pred, cfg):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = np.abs(y - pred)

    # -----------------------------------------------------------------
    # 1) Robust constant sigma = default / fallback.
    # -----------------------------------------------------------------
    const_grid = np.linspace(
        cfg.constant_sigma_min,
        cfg.constant_sigma_max,
        cfg.constant_sigma_steps,
    )
    const_scores = [
        compute_laplace_score(y, pred, s, cfg)
        for s in const_grid
    ]
    j = int(np.argmax(const_scores))
    const_sigma = float(const_grid[j])
    const_score = float(const_scores[j])

    candidates = [(
        const_score,
        {
            "type": "constant",
            "sigma": const_sigma,
            "constant_sigma": const_sigma,
            "raw_inner_lll": const_score,
        }
    )]

    # -----------------------------------------------------------------
    # 2) Paper-inspired slope sigma, stabilized toward constant.
    # -----------------------------------------------------------------
    paper_base = (
        records["Percent"].to_numpy(dtype=np.float64)
        - np.asarray(slope_pred, dtype=np.float64)
          * np.abs(
              records["WeeksFromBaseline"].to_numpy(dtype=np.float64)
          )
    )
    paper_base = np.maximum(
        paper_base, cfg.laplace_sigma_floor
    )
    paper_scale, _ = optimize_sigma_scale(
        y, pred, paper_base, cfg
    )
    paper_raw = np.maximum(
        paper_base * paper_scale,
        cfg.laplace_sigma_floor,
    )
    paper_stable = _shrink_sigma_to_constant(
        paper_raw, const_sigma, cfg
    )
    paper_score = compute_laplace_score(
        y, pred, paper_stable, cfg
    )
    candidates.append((
        float(paper_score),
        {
            "type": "paper_slope_stable",
            "scale": float(paper_scale),
            "constant_sigma": const_sigma,
            "raw_inner_lll": float(paper_score),
        }
    ))

    # -----------------------------------------------------------------
    # 3) Low-capacity ridge residual model, stabilized toward constant.
    # -----------------------------------------------------------------
    X = uncertainty_features(records, pred, expert_matrix)
    sx = StandardScaler().fit(X)
    Xs = sx.transform(X)

    sigma_target = np.maximum(
        np.sqrt(2.0) * err,
        cfg.laplace_sigma_floor,
    )
    ridge = Ridge(alpha=cfg.uncertainty_ridge_alpha)
    ridge.fit(Xs, np.log(sigma_target))

    ridge_base = np.exp(ridge.predict(Xs))
    ridge_base = np.maximum(
        ridge_base, cfg.laplace_sigma_floor
    )
    ridge_scale, _ = optimize_sigma_scale(
        y, pred, ridge_base, cfg
    )
    ridge_raw = np.maximum(
        ridge_base * ridge_scale,
        cfg.laplace_sigma_floor,
    )
    ridge_stable = _shrink_sigma_to_constant(
        ridge_raw, const_sigma, cfg
    )
    ridge_score = compute_laplace_score(
        y, pred, ridge_stable, cfg
    )

    candidates.append((
        float(ridge_score),
        {
            "type": "ridge_residual_stable",
            "scale": float(ridge_scale),
            "constant_sigma": const_sigma,
            "scaler": sx,
            "ridge": ridge,
            "raw_inner_lll": float(ridge_score),
        }
    ))

    # Variable sigma is accepted only if its inner LLL advantage over the
    # constant model is non-trivial. Otherwise prefer the much more stable
    # constant uncertainty model.
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best = candidates[0]

    if best["type"] != "constant":
        gain = float(best_score - const_score)
        if gain < cfg.uncertainty_candidate_min_lll_gain:
            best_score = const_score
            best = candidates[
                next(i for i, c in enumerate(candidates)
                     if c[1]["type"] == "constant")
            ][1].copy()
            best["selection_note"] = (
                f"fallback_constant_gain_{gain:.5f}"
            )
        else:
            best["selection_note"] = (
                f"variable_gain_{gain:.5f}"
            )
    else:
        best = best.copy()
        best["selection_note"] = "constant_best"

    best["inner_lll"] = float(best_score)
    best["candidate_scores"] = [
        (c[1]["type"], float(c[0]))
        for c in candidates
    ]
    best["constant_sigma"] = const_sigma
    return best


def apply_uncertainty_calibrator(calib, records, pred, expert_matrix, slope_pred, cfg):
    t = calib["type"]

    if t == "constant":
        return np.full(
            len(records),
            calib["sigma"],
            dtype=np.float64,
        )

    if t == "paper_slope_stable":
        base = (
            records["Percent"].to_numpy(dtype=np.float64)
            - np.asarray(slope_pred, dtype=np.float64)
              * np.abs(
                  records["WeeksFromBaseline"].to_numpy(dtype=np.float64)
              )
        )
        raw = np.maximum(
            base * calib["scale"],
            cfg.laplace_sigma_floor,
        )
        return _shrink_sigma_to_constant(
            raw, calib["constant_sigma"], cfg
        )

    if t == "ridge_residual_stable":
        X = uncertainty_features(
            records, pred, expert_matrix
        )
        base = np.exp(
            calib["ridge"].predict(
                calib["scaler"].transform(X)
            )
        )
        raw = np.maximum(
            base * calib["scale"],
            cfg.laplace_sigma_floor,
        )
        return _shrink_sigma_to_constant(
            raw, calib["constant_sigma"], cfg
        )

    raise ValueError(
        f"Unknown uncertainty calibrator: {t}"
    )


# =============================================================================
# Evaluation conventions
# =============================================================================

def per_patient_oof_metrics(oof, cfg):
    """
    One row per strict OOF patient.

    These are descriptive patient-level metrics; they do not alter the pooled
    scoring. They are particularly important for the small strict cohort.
    """
    rows = []
    for pid, g in oof.groupby("Patient"):
        y = g["TargetFVC"].to_numpy(dtype=np.float64)
        p = g["PredFVC"].to_numpy(dtype=np.float64)
        s = g["PredSigma"].to_numpy(dtype=np.float64)
        ae = np.abs(y - p)

        rows.append({
            "Patient": pid,
            "NTargets": int(len(g)),
            "BaselineWeek": float(g["BaselineWeek"].iloc[0]),
            "BaselineFVC": float(g["BaselineFVC"].iloc[0]),
            "Percent": float(g["Percent"].iloc[0]),
            "FSS": float(g["FSS"].iloc[0]),
            "MeanTargetWeekFromCT": float(g["TargetWeekFromCT"].mean()),
            "MaxTargetWeekFromCT": float(g["TargetWeekFromCT"].max()),
            "MAE": float(np.mean(ae)),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "LLL": float(compute_laplace_score(y, p, s, cfg)),
            "P95AE": float(np.quantile(ae, 0.95)),
            "MaxAE": float(np.max(ae)),
            "N500": int(np.sum(ae >= cfg.catastrophic_error_ml)),
            "MeanSigma": float(np.mean(s)),
            "SignedBias": float(np.mean(p - y)),
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["MAE", "MaxAE"], ascending=False
        ).reset_index(drop=True)
    return out


def _bootstrap_duplicate_patients(oof, sampled_patients):
    """
    Build one patient-cluster bootstrap sample.

    Duplicate sampled patients receive unique temporary IDs so patient-level
    aggregation and trajectory metrics correctly count repeated bootstrap draws.
    """
    pieces = []
    for draw_idx, pid in enumerate(sampled_patients):
        g = oof[oof["Patient"] == pid].copy()
        if len(g) == 0:
            continue
        g["Patient"] = (
            g["Patient"].astype(str)
            + f"__bootdraw_{draw_idx:03d}"
        )
        pieces.append(g)

    if not pieces:
        return pd.DataFrame(columns=oof.columns)
    return pd.concat(pieces, ignore_index=True)


def patient_cluster_bootstrap(oof, cfg):
    """
    Patient-cluster bootstrap confidence intervals for:
      - strict pooled future-visit MAE/RMSE/LLL
      - patient-mean future-only MAE/RMSE/LLL
      - trajectory-with-observed-anchor MAE/RMSE/LLL

    Resampling unit = patient, never individual visit.
    """
    patient_ids = np.array(
        sorted(oof["Patient"].astype(str).unique())
    )
    n_patients = len(patient_ids)
    if n_patients < 2:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(cfg.bootstrap_seed)
    rows = []

    for b in range(int(cfg.bootstrap_repetitions)):
        sampled = rng.choice(
            patient_ids, size=n_patients, replace=True
        )
        boot = _bootstrap_duplicate_patients(oof, sampled)
        if len(boot) == 0:
            continue

        strict = metric_bundle(
            boot["TargetFVC"],
            boot["PredFVC"],
            boot["PredSigma"],
            cfg,
        )
        pf = patient_mean_future_metrics(boot, cfg)
        pt = patient_trajectory_metrics_with_baseline(boot, cfg)

        rows.append({
            "Bootstrap": b + 1,
            "Strict_MAE": strict["MAE"],
            "Strict_RMSE": strict["RMSE"],
            "Strict_LLL": strict["LLL"],
            "PatientFuture_MAE": pf["MAE"],
            "PatientFuture_RMSE": pf["RMSE"],
            "PatientFuture_LLL": pf["LLL"],
            "Trajectory_MAE": pt["MAE"],
            "Trajectory_RMSE": pt["RMSE"],
            "Trajectory_LLL": pt["LLL"],
        })

    dist = pd.DataFrame(rows)
    if len(dist) == 0:
        return dist, pd.DataFrame()

    alpha = 1.0 - float(cfg.bootstrap_confidence)
    qlo = alpha / 2.0
    qhi = 1.0 - alpha / 2.0

    summary_rows = []
    for col in [
        "Strict_MAE", "Strict_RMSE", "Strict_LLL",
        "PatientFuture_MAE", "PatientFuture_RMSE", "PatientFuture_LLL",
        "Trajectory_MAE", "Trajectory_RMSE", "Trajectory_LLL",
    ]:
        x = dist[col].to_numpy(dtype=np.float64)
        summary_rows.append({
            "Metric": col,
            "BootstrapMean": float(np.mean(x)),
            "BootstrapSD": float(np.std(x, ddof=1)),
            "CI_Lower": float(np.quantile(x, qlo)),
            "CI_Upper": float(np.quantile(x, qhi)),
            "Confidence": float(cfg.bootstrap_confidence),
            "NBootstrap": int(len(x)),
            "NPatients": int(n_patients),
        })

    return dist, pd.DataFrame(summary_rows)


def print_bootstrap_summary(bootstrap_summary):
    if bootstrap_summary is None or len(bootstrap_summary) == 0:
        return

    confidence = 100.0 * float(
        bootstrap_summary["Confidence"].iloc[0]
    )
    print(
        f"\nPATIENT-CLUSTER BOOTSTRAP {confidence:.0f}% CONFIDENCE INTERVALS"
    )
    print(
        bootstrap_summary[
            ["Metric", "BootstrapMean", "BootstrapSD", "CI_Lower", "CI_Upper"]
        ].round({
            "BootstrapMean": 5,
            "BootstrapSD": 5,
            "CI_Lower": 5,
            "CI_Upper": 5,
        }).to_string(index=False)
    )

def patient_mean_future_metrics(oof, cfg):
    rows = []
    for pid, g in oof.groupby("Patient"):
        m = metric_bundle(g["TargetFVC"], g["PredFVC"], g["PredSigma"], cfg)
        rows.append(m)
    d = pd.DataFrame(rows)
    return {k: float(d[k].mean()) for k in ["MAE", "RMSE", "LLL", "MeanSigma"]}


def patient_trajectory_metrics_with_baseline(oof, cfg):
    """
    Closer to the released SOTA repository's trajectory evaluation: each patient's
    known baseline FVC is included as an exactly reconstructed time point, then
    patient-level metrics are averaged across patients.

    This is reported separately and MUST NOT replace the strict future-only OOF
    metric in the manuscript unless the comparison protocol is explicitly stated.
    """
    rows = []
    for pid, g in oof.groupby("Patient"):
        g = g.sort_values("TargetWeek")
        base = float(g["BaselineFVC"].iloc[0])
        y = np.concatenate([[base], g["TargetFVC"].to_numpy(dtype=np.float64)])
        p = np.concatenate([[base], g["PredFVC"].to_numpy(dtype=np.float64)])
        # Baseline is observed, so use the competition floor for its confidence.
        s = np.concatenate([[cfg.laplace_sigma_floor], g["PredSigma"].to_numpy(dtype=np.float64)])
        rows.append(metric_bundle(y, p, s, cfg))
    d = pd.DataFrame(rows)
    return {k: float(d[k].mean()) for k in ["MAE", "RMSE", "LLL", "MeanSigma"]}


# =============================================================================
# Fold runner
# =============================================================================

def tail_diagnostics(y, pred, cfg):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ae = np.abs(y - pred)
    n_tail = max(1, int(np.ceil(cfg.tail_fraction * len(ae))))
    tail = np.sort(ae)[-n_tail:]
    return {
        "TailMAE": float(np.mean(tail)),
        "P95AE": float(np.quantile(ae, 0.95)),
        "MaxAE": float(np.max(ae)),
        "N500": int(np.sum(ae >= cfg.catastrophic_error_ml)),
    }


def build_error_diagnostics(oof, cfg):
    d = oof.copy()
    d["SignedError"] = d["PredFVC"] - d["TargetFVC"]
    d["TrueDeltaFVC"] = d["TargetFVC"] - d["BaselineFVC"]
    d["PredDeltaFVC"] = d["PredFVC"] - d["BaselineFVC"]

    d["TrueSlope"] = (
        d["TrueDeltaFVC"]
        / np.maximum(np.abs(d["WeeksFromBaseline"]), 1.0)
    )
    d["PredSlope"] = (
        d["PredDeltaFVC"]
        / np.maximum(np.abs(d["WeeksFromBaseline"]), 1.0)
    )

    catastrophic = d.loc[
        d["AbsError"] >= cfg.catastrophic_error_ml
    ].copy().sort_values("AbsError", ascending=False)

    # Primary horizon diagnostic is relative to the CT acquisition (Week 0).
    horizon_labels = ["<=12", "13-26", "27-52", "53-78", ">78"]
    horizon_bins = [-np.inf, 12, 26, 52, 78, np.inf]
    d["CTHorizonBin"] = pd.cut(
        d["TargetWeekFromCT"],
        bins=horizon_bins,
        labels=horizon_labels,
    )

    horizon = (
        d.groupby("CTHorizonBin", observed=True)
        .agg(
            N=("AbsError", "size"),
            MAE=("AbsError", "mean"),
            P95AE=("AbsError", lambda x: float(
                np.quantile(x.to_numpy(dtype=np.float64), 0.95)
            )),
            MaxAE=("AbsError", "max"),
            N500=("AbsError", lambda x: int(np.sum(
                x.to_numpy(dtype=np.float64)
                >= cfg.catastrophic_error_ml
            ))),
            MeanWeeksFromBaseline=("WeeksFromBaseline", "mean"),
            MeanBaselineWeek=("BaselineWeek", "mean"),
        )
        .reindex(horizon_labels)
        .reset_index()
        .rename(columns={"CTHorizonBin": "HorizonBin"})
    )

    horizon["N"] = horizon["N"].fillna(0).astype(int)
    horizon["N500"] = horizon["N500"].fillna(0).astype(int)
    horizon["N500Rate"] = np.where(
        horizon["N"] > 0,
        horizon["N500"] / horizon["N"],
        np.nan,
    )

    patient = (
        catastrophic.groupby("Patient")
        .agg(
            N500=("AbsError", "size"),
            MeanAE=("AbsError", "mean"),
            MaxAE=("AbsError", "max"),
            MedianTargetWeekFromCT=("TargetWeekFromCT", "median"),
            MedianWeeksFromBaseline=("WeeksFromBaseline", "median"),
            BaselineWeek=("BaselineWeek", "first"),
            BaselineFVC=("BaselineFVC", "first"),
            Percent=("Percent", "first"),
            FSS=("FSS", "first"),
        )
        .sort_values(["N500", "MaxAE"], ascending=False)
        .reset_index()
    )

    return catastrophic, horizon, patient


def run_recovery(cfg, df):
    set_seed(cfg.seed)
    print("\nPreparing DenseNet169 feature extractor...")
    extractor = FeatureExtractor(cfg.backbone)
    print("Building/loading automatically cropped lower/full lung feature banks...")
    lower_bank, full_bank, fss_proxy = build_feature_banks(cfg, extractor)

    eligible_patients, excluded_patients = strict_eligible_patients(
        df, cfg
    )
    strict_df = df[df["Patient"].isin(eligible_patients)].copy()

    if strict_df["Patient"].nunique() < cfg.n_folds:
        raise RuntimeError(
            f"Only {strict_df['Patient'].nunique()} strict-protocol patients "
            f"for {cfg.n_folds}-fold CV."
        )

    all_records, has_true_fss, has_smoking = build_records(
        strict_df, fss_proxy, cfg
    )

    print("\nSTRICT CT-BASELINE COHORT")
    print(
        f"Original patients: {df['Patient'].nunique()} | "
        f"eligible: {strict_df['Patient'].nunique()} | "
        f"excluded: {excluded_patients['Patient'].nunique() if len(excluded_patients) else 0}"
    )

    if len(excluded_patients):
        print("Exclusion reasons:")
        print(
            excluded_patients["ExclusionReason"]
            .value_counts()
            .to_string()
        )

    print(
        f"Strict future targets: {len(all_records)} | "
        f"patients={all_records['Patient'].nunique()}"
    )

    if len(all_records):
        patient_baselines = (
            all_records[["Patient", "BaselineWeek"]]
            .drop_duplicates("Patient")
        )
        n_b0 = int(np.sum(
            np.isclose(patient_baselines["BaselineWeek"], 0.0)
        ))
        n_pre = int(np.sum(
            patient_baselines["BaselineWeek"] < 0.0
        ))
        n_post = int(np.sum(
            patient_baselines["BaselineWeek"] > 0.0
        ))

        print(
            f"STRICT baselines: Week 0={n_b0} | "
            f"pre-CT={n_pre} | post-CT={n_post}"
        )
        if n_post != 0:
            raise RuntimeError(
                "v8 protocol violation: post-CT strict baseline detected."
            )

        print(
            "STRICT targets: "
            f"TargetWeekFromCT range "
            f"{all_records['TargetWeekFromCT'].min():.0f} .. "
            f"{all_records['TargetWeekFromCT'].max():.0f} weeks"
        )

    print(
        f"True FSS column available: {has_true_fss} | "
        f"SmokingStatus available: {has_smoking}"
    )
    if not has_true_fss:
        print(
            "NOTE: FSS is a CT-density proxy, "
            "not a validated clinical FSS."
        )

    # Secondary dynamic records are descriptive only and are never mixed into
    # strict training, model selection, calibration, or OOF metrics.
    dynamic_manifest = pd.DataFrame()
    if cfg.build_dynamic_secondary_manifest:
        dynamic_manifest = build_dynamic_followup_records(
            df, fss_proxy, cfg
        )
        print(
            "\nSECONDARY dynamic-follow-up manifest "
            "(NOT used in strict model):"
        )
        print(
            f"rows={len(dynamic_manifest)} | "
            f"patients={dynamic_manifest['Patient'].nunique() if len(dynamic_manifest) else 0}"
        )
        if len(dynamic_manifest):
            n_dynamic_post_anchor = int(np.sum(
                dynamic_manifest["BaselineWeek"] > 0
            ))
            print(
                f"dynamic anchors after CT={n_dynamic_post_anchor} "
                f"(allowed only in secondary protocol)"
            )

    # -----------------------------------------------------------------
    # Patient-balanced outer folds.
    # -----------------------------------------------------------------
    outer_folds = make_balanced_patient_folds(
        all_records, cfg.n_folds, cfg, cfg.seed
    )

    print("\nBALANCED OUTER-FOLD PATIENT PROFILES")
    outer_profile = pd.DataFrame(
        [f["summary"] for f in outer_folds]
    )
    print(
        outer_profile.round({
            "MeanBaselineFVC": 1,
            "MeanPercent": 1,
            "MeanBaselineWeek": 2,
            "MeanSlope": 2,
            "RapidRate": 3,
            "MeanHorizon": 1,
        }).to_string(index=False)
    )

    fold_rows = []
    oof_parts = []

    for fold, fold_def in enumerate(outer_folds, start=1):
        print("\n" + "=" * 110)
        print(f"RECOVERY OUTER FOLD {fold}/{cfg.n_folds}")
        print("=" * 110)

        train_patients = fold_def["train_patients"]
        val_patients = fold_def["val_patients"]

        train_df = strict_df[
            strict_df["Patient"].astype(str).isin(train_patients)
        ].copy()
        val_df = strict_df[
            strict_df["Patient"].astype(str).isin(val_patients)
        ].copy()
        train_records, _, _ = build_records(train_df, fss_proxy, cfg)
        val_records, _, _ = build_records(val_df, fss_proxy, cfg)
        if len(train_records) == 0 or len(val_records) == 0:
            raise RuntimeError(f"Fold {fold}: empty train/validation records")

        bounds = training_prediction_bounds(train_records, cfg)
        print(f"Broad physiological FVC sanity bounds: {bounds[0]:.0f} .. {bounds[1]:.0f} mL")

        # Balanced inner patient split for model selection, blending and
        # uncertainty calibration.
        inner_train, inner_val, inner_profile = balanced_inner_split(
            train_records,
            cfg,
            seed=cfg.seed + 100 * fold,
            outer_fold=fold,
        )
        print(
            f"Balanced inner split: "
            f"{inner_train['Patient'].nunique()} train patients / "
            f"{inner_val['Patient'].nunique()} validation patients | "
            f"val targets={inner_profile['Targets']} | "
            f"mean slope={inner_profile['MeanSlope']:.2f} | "
            f"rapid rate={inner_profile['RapidRate']:.3f}"
        )

        inner_experts = {}
        outer_experts = {}
        expert_epochs = {}

        # ---------------------------------------------------------------------
        # Primary lower-lung Direct Delta-FVC expert, multi-seed average.
        # ---------------------------------------------------------------------
        inner_seed_preds, outer_seed_preds = [], []
        for base_seed in cfg.direct_seeds:
            seed = int(base_seed + 1000 * fold)
            ep, y_inner, p_inner = select_direct_epoch(
                inner_train, inner_val, lower_bank, extractor.feature_dim, cfg, seed
            )
            y_outer, p_outer = fit_direct_fixed(
                train_records, val_records, lower_bank, extractor.feature_dim, cfg, seed, ep
            )
            inner_seed_preds.append(p_inner)
            outer_seed_preds.append(p_outer)
            expert_epochs[f"LowerDirect_seed{base_seed}"] = ep
            print(f"  Lower direct seed {base_seed}: epoch={ep}, inner MAE={np.mean(np.abs(y_inner-p_inner)):.2f}")
        inner_experts["LowerDirect"] = np.mean(np.stack(inner_seed_preds), axis=0)
        outer_experts["LowerDirect"] = np.mean(np.stack(outer_seed_preds), axis=0)

        # ---------------------------------------------------------------------
        # Whole relevant-lung Direct Delta-FVC expert for diversity.
        # ---------------------------------------------------------------------
        if cfg.use_whole_relevant_lung_expert:
            inner_seed_preds, outer_seed_preds = [], []
            for base_seed in cfg.direct_seeds:
                seed = int(base_seed + 2000 * fold + 17)
                ep, _yi, pi = select_direct_epoch(
                    inner_train, inner_val, full_bank, extractor.feature_dim, cfg, seed
                )
                _yo, po = fit_direct_fixed(
                    train_records, val_records, full_bank, extractor.feature_dim, cfg, seed, ep
                )
                inner_seed_preds.append(pi); outer_seed_preds.append(po)
                expert_epochs[f"FullDirect_seed{base_seed}"] = ep
            inner_experts["FullDirect"] = np.mean(np.stack(inner_seed_preds), axis=0)
            outer_experts["FullDirect"] = np.mean(np.stack(outer_seed_preds), axis=0)

        # ---------------------------------------------------------------------
        # Independent patient-level slope expert, trained with L1.
        # ---------------------------------------------------------------------
        inner_slope_pred = None
        outer_slope_pred = None
        if cfg.use_slope_neural_expert:
            inner_slope_fvcs, outer_slope_fvcs = [], []
            inner_slopes, outer_slopes = [], []
            for base_seed in cfg.direct_seeds:
                seed = int(base_seed + 3000 * fold + 31)
                ep, pi, si = select_slope_epoch(
                    inner_train, inner_val, lower_bank, extractor.feature_dim, cfg, seed
                )
                po, so = fit_slope_fixed(
                    train_records, val_records, lower_bank, extractor.feature_dim, cfg, seed, ep
                )
                inner_slope_fvcs.append(pi); outer_slope_fvcs.append(po)
                inner_slopes.append(si); outer_slopes.append(so)
                expert_epochs[f"Slope_seed{base_seed}"] = ep
            inner_experts["SlopeNeural"] = np.mean(np.stack(inner_slope_fvcs), axis=0)
            outer_experts["SlopeNeural"] = np.mean(np.stack(outer_slope_fvcs), axis=0)
            inner_slope_pred = np.mean(np.stack(inner_slopes), axis=0)
            outer_slope_pred = np.mean(np.stack(outer_slopes), axis=0)

        # ---------------------------------------------------------------------
        # Tabular experts trained independently of image model.
        # ---------------------------------------------------------------------
        if cfg.use_tabular_experts:
            prep_i, tab_i = fit_tabular_experts(inner_train, cfg.seed + fold)
            inner_experts.update(predict_tabular_experts(prep_i, tab_i, inner_val))
            prep_o, tab_o = fit_tabular_experts(train_records, cfg.seed + 100 + fold)
            outer_experts.update(predict_tabular_experts(prep_o, tab_o, val_records))

        # Persistence is a useful shrinkage expert for short horizons.
        inner_experts["Persistence"] = inner_val["BaselineFVC"].to_numpy(dtype=np.float64)
        outer_experts["Persistence"] = val_records["BaselineFVC"].to_numpy(dtype=np.float64)

        names = list(inner_experts.keys())
        if set(names) != set(outer_experts.keys()):
            raise RuntimeError("Inner/outer expert sets do not match")

        # Ensure all experts are in identical order and safely bounded.
        P_inner = np.column_stack([clip_predictions(inner_experts[n], bounds) for n in names])
        P_outer = np.column_stack([clip_predictions(outer_experts[n], bounds) for n in names])
        y_inner = inner_val["TargetFVC"].to_numpy(dtype=np.float64)
        y_outer = val_records["TargetFVC"].to_numpy(dtype=np.float64)

        # If slope expert was disabled, derive a conservative slope proxy from
        # the strongest lower-direct expert for uncertainty only.
        if inner_slope_pred is None:
            dt = np.maximum(inner_val["WeeksFromBaseline"].to_numpy(dtype=np.float64), 1.0)
            inner_slope_pred = (P_inner[:, names.index("LowerDirect")] - inner_val["BaselineFVC"].to_numpy(dtype=np.float64)) / dt
            dto = np.maximum(val_records["WeeksFromBaseline"].to_numpy(dtype=np.float64), 1.0)
            outer_slope_pred = (P_outer[:, names.index("LowerDirect")] - val_records["BaselineFVC"].to_numpy(dtype=np.float64)) / dto

        weights, bias = fit_convex_blend(y_inner, P_inner, cfg)
        pred_inner = clip_predictions(P_inner @ weights + bias, bounds)
        pred_outer = clip_predictions(P_outer @ weights + bias, bounds)

        weight_text = ", ".join(f"{n}={w:.3f}" for n, w in zip(names, weights))
        print(f"Blend weights: {weight_text} | bias={bias:+.1f} mL")
        print(
            f"Inner blend: MAE={np.mean(np.abs(y_inner-pred_inner)):.2f} | "
            f"RMSE={np.sqrt(np.mean((y_inner-pred_inner)**2)):.2f}"
        )

        # Decoupled uncertainty: cannot change the FVC mean prediction.
        calibrator = fit_uncertainty_calibrator(
            inner_val, y_inner, pred_inner, P_inner, inner_slope_pred, cfg
        )
        sigma_outer = apply_uncertainty_calibrator(
            calibrator, val_records, pred_outer, P_outer, outer_slope_pred, cfg
        )
        print(
            f"Uncertainty selected: {calibrator['type']} | "
            f"inner candidate LLLs={calibrator['candidate_scores']}"
        )

        fm = metric_bundle(y_outer, pred_outer, sigma_outer, cfg)
        td = tail_diagnostics(y_outer, pred_outer, cfg)
        fold_rows.append({
            "Fold": fold,
            "Patients": int(val_records["Patient"].nunique()),
            "Targets": int(len(val_records)),
            "MAE": fm["MAE"], "RMSE": fm["RMSE"], "LLL": fm["LLL"],
            "MeanSigma": fm["MeanSigma"],
            "ConstantSigma": float(calibrator.get("constant_sigma", np.nan)),
            "TailMAE": td["TailMAE"],
            "P95AE": td["P95AE"],
            "MaxAE": td["MaxAE"],
            "N500": td["N500"],
            "MeanSlope": float(
                val_records[["Patient", "PatientSlope"]]
                .drop_duplicates("Patient")["PatientSlope"].mean()
            ),
            "RapidRate": float(
                (
                    val_records[["Patient", "PatientSlope"]]
                    .drop_duplicates("Patient")["PatientSlope"]
                    <= np.quantile(
                        all_records[["Patient", "PatientSlope"]]
                        .drop_duplicates("Patient")["PatientSlope"],
                        cfg.balance_rapid_decline_quantile
                    )
                ).mean()
            ),
            "Uncertainty": calibrator["type"],
            "UncertaintySelection": calibrator.get("selection_note", ""),
            "BlendBias": bias,
            "BlendWeights": weight_text,
            "MeanSelectedEpoch": float(np.mean(list(expert_epochs.values()))) if expert_epochs else np.nan,
        })

        out = val_records[[
            "Patient",
            "Protocol",
            "BaselineWeek",
            "BaselineFVC",
            "TargetWeek",
            "TargetWeekFromCT",
            "WeeksFromBaseline",
            "WeeksDelta",
            "TargetFVC",
            "Percent",
            "FSS",
        ]].copy()
        out["PredFVC"] = pred_outer
        out["PredSigma"] = sigma_outer
        out["AbsError"] = np.abs(y_outer - pred_outer)
        out["Fold"] = fold
        for j, n in enumerate(names):
            out[f"Expert_{n}"] = P_outer[:, j]
        oof_parts.append(out)

        print(
            f"FOLD {fold}: MAE={fm['MAE']:.2f} mL | RMSE={fm['RMSE']:.2f} mL | "
            f"LLL={fm['LLL']:.5f} | sigma={fm['MeanSigma']:.1f}"
        )

    fold_table = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_parts, ignore_index=True)
    pooled = metric_bundle(oof["TargetFVC"], oof["PredFVC"], oof["PredSigma"], cfg)
    pooled_tail = tail_diagnostics(oof["TargetFVC"], oof["PredFVC"], cfg)
    patient_future = patient_mean_future_metrics(oof, cfg)
    patient_traj = patient_trajectory_metrics_with_baseline(oof, cfg)
    catastrophic, horizon_summary, patient_error_summary = build_error_diagnostics(oof, cfg)

    # v10 IMPORTANT evaluation additions. These do not alter predictions.
    per_patient_metrics = per_patient_oof_metrics(oof, cfg)

    bootstrap_dist = pd.DataFrame()
    bootstrap_summary = pd.DataFrame()
    if cfg.enable_patient_bootstrap:
        print(
            f"\nRunning patient-cluster bootstrap "
            f"({cfg.bootstrap_repetitions} resamples)..."
        )
        bootstrap_dist, bootstrap_summary = patient_cluster_bootstrap(
            oof, cfg
        )

    print("\n" + "=" * 125)
    print(f"FINAL STABLE v10 (FROZEN v9) - {cfg.n_folds}-FOLD RESULTS")
    print("=" * 125)
    print(fold_table[[
        "Fold", "Patients", "Targets",
        "MAE", "RMSE", "LLL",
        "MeanSigma", "ConstantSigma",
        "TailMAE", "P95AE", "MaxAE", "N500",
        "MeanSlope", "RapidRate",
        "Uncertainty", "MeanSelectedEpoch"
    ]].round({
        "MAE": 2, "RMSE": 2, "LLL": 5,
        "MeanSigma": 1, "ConstantSigma": 1,
        "TailMAE": 1, "P95AE": 1, "MaxAE": 1,
        "MeanSlope": 2, "RapidRate": 3,
        "MeanSelectedEpoch": 1,
    }).to_string(index=False))

    print("\nFold mean ± SD")
    print(f"MAE : {fold_table['MAE'].mean():.2f} ± {fold_table['MAE'].std(ddof=1):.2f} mL")
    print(f"RMSE: {fold_table['RMSE'].mean():.2f} ± {fold_table['RMSE'].std(ddof=1):.2f} mL")
    print(f"LLL : {fold_table['LLL'].mean():.5f} ± {fold_table['LLL'].std(ddof=1):.5f}")
    print(
        f"Targets/fold: {fold_table['Targets'].min()} .. "
        f"{fold_table['Targets'].max()} | "
        f"Patients/fold: {fold_table['Patients'].min()} .. "
        f"{fold_table['Patients'].max()}"
    )

    print("\nPRIMARY STRICT CT-BASELINE pooled post-CT OOF metrics")
    print(f"OOF MAE  = {pooled['MAE']:.2f} mL")
    print(f"OOF RMSE = {pooled['RMSE']:.2f} mL")
    print(f"OOF LLL  = {pooled['LLL']:.5f}")
    print(f"Mean sigma = {pooled['MeanSigma']:.1f} mL")
    print(f"Tail {int(cfg.tail_fraction*100)}% MAE = {pooled_tail['TailMAE']:.2f} mL")
    print(f"95th percentile absolute error = {pooled_tail['P95AE']:.2f} mL")
    print(f"Maximum absolute error = {pooled_tail['MaxAE']:.2f} mL")
    print(f"Predictions with |error| >= {cfg.catastrophic_error_ml:.0f} mL = {pooled_tail['N500']}")

    print("\nPRIMARY patient-mean post-CT FUTURE-ONLY metrics")
    print(f"MAE  = {patient_future['MAE']:.2f} mL")
    print(f"RMSE = {patient_future['RMSE']:.2f} mL")
    print(f"LLL  = {patient_future['LLL']:.5f}")

    print("\nPRIMARY patient-trajectory metrics INCLUDING observed pre/at-CT anchor")
    print(f"MAE  = {patient_traj['MAE']:.2f} mL")
    print(f"RMSE = {patient_traj['RMSE']:.2f} mL")
    print(f"LLL  = {patient_traj['LLL']:.5f}")

    if cfg.enable_patient_bootstrap:
        print_bootstrap_summary(bootstrap_summary)

    if len(per_patient_metrics):
        print("\nWORST PATIENTS BY STRICT OOF MAE")
        print(
            per_patient_metrics.head(10).round({
                "MAE": 2, "RMSE": 2, "LLL": 5,
                "P95AE": 2, "MaxAE": 2,
                "MeanSigma": 1, "SignedBias": 2,
            }).to_string(index=False)
        )

    print("\nError by prediction horizon FROM CT (Week 0)")
    print(horizon_summary.round(3).to_string(index=False))

    print("\nPatients contributing the most >=500 mL errors")
    if len(patient_error_summary):
        print(patient_error_summary.head(20).round(2).to_string(index=False))
    else:
        print("None")

    if len(catastrophic):
        too_high = int(np.sum(catastrophic["SignedError"] > 0))
        too_low = int(np.sum(catastrophic["SignedError"] < 0))
        print(
            f"\nCatastrophic direction: predicted FVC too high={too_high}, "
            f"too low={too_low}"
        )

    print("\nPublished reference (Dolatabadi et al.)")
    print(f"LLL target : {cfg.sota_lll:.3f}")
    print(f"RMSE target: {cfg.sota_rmse:.2f} mL")
    print(
        "PRIMARY strict protocol uses only FVC available at/before CT and only post-CT targets. "
        "The trajectory block includes the observed pre/at-CT anchor and must be labeled "
        "explicitly when compared with published trajectory-style evaluations."
    )

    os.makedirs("/kaggle/working", exist_ok=True)
    prefix = str(cfg.output_prefix)
    fold_path = f"/kaggle/working/{prefix}_fold_results.csv"
    oof_path = f"/kaggle/working/{prefix}_oof_predictions.csv"
    summary_path = f"/kaggle/working/{prefix}_summary.csv"
    catastrophic_path = f"/kaggle/working/{prefix}_catastrophic_errors.csv"
    horizon_path = f"/kaggle/working/{prefix}_horizon_error_summary.csv"
    patient_path = f"/kaggle/working/{prefix}_patient_error_summary.csv"
    excluded_path = f"/kaggle/working/{prefix}_strict_excluded_patients.csv"
    dynamic_manifest_path = f"/kaggle/working/{prefix}_dynamic_followup_manifest.csv"
    outer_profile_path = f"/kaggle/working/{prefix}_outer_fold_profiles.csv"
    per_patient_path = f"/kaggle/working/{prefix}_per_patient_oof_metrics.csv"
    bootstrap_dist_path = f"/kaggle/working/{prefix}_patient_bootstrap_distribution.csv"
    bootstrap_summary_path = f"/kaggle/working/{prefix}_patient_bootstrap_ci.csv"

    fold_table.to_csv(fold_path, index=False)
    oof.to_csv(oof_path, index=False)
    catastrophic.to_csv(catastrophic_path, index=False)
    horizon_summary.to_csv(horizon_path, index=False)
    patient_error_summary.to_csv(patient_path, index=False)
    excluded_patients.to_csv(excluded_path, index=False)
    if cfg.build_dynamic_secondary_manifest:
        dynamic_manifest.to_csv(dynamic_manifest_path, index=False)
    outer_profile.to_csv(outer_profile_path, index=False)

    if cfg.save_per_patient_metrics:
        per_patient_metrics.to_csv(per_patient_path, index=False)

    if cfg.enable_patient_bootstrap:
        bootstrap_dist.to_csv(bootstrap_dist_path, index=False)
        bootstrap_summary.to_csv(bootstrap_summary_path, index=False)

    pd.DataFrame([{
        "Strict_Eligible_Patients": strict_df["Patient"].nunique(),
        "Strict_Excluded_Patients": (
            excluded_patients["Patient"].nunique()
            if len(excluded_patients) else 0
        ),
        "Strict_BaselineWeek_Max": (
            float(all_records["BaselineWeek"].max())
            if len(all_records) else np.nan
        ),
        "Strict_OOF_MAE": pooled["MAE"],
        "Strict_OOF_RMSE": pooled["RMSE"],
        "Strict_OOF_LLL": pooled["LLL"],
        "Strict_TailMAE": pooled_tail["TailMAE"],
        "Strict_P95AE": pooled_tail["P95AE"],
        "Strict_MaxAE": pooled_tail["MaxAE"],
        "Strict_N500": pooled_tail["N500"],
        "PatientFuture_MAE": patient_future["MAE"],
        "PatientFuture_RMSE": patient_future["RMSE"],
        "PatientFuture_LLL": patient_future["LLL"],
        "TrajectoryWithBaseline_MAE": patient_traj["MAE"],
        "TrajectoryWithBaseline_RMSE": patient_traj["RMSE"],
        "TrajectoryWithBaseline_LLL": patient_traj["LLL"],
        "SOTA_LLL_reference": cfg.sota_lll,
        "SOTA_RMSE_reference": cfg.sota_rmse,
        "BootstrapEnabled": bool(cfg.enable_patient_bootstrap),
        "BootstrapRepetitions": (
            int(cfg.bootstrap_repetitions)
            if cfg.enable_patient_bootstrap else 0
        ),
        "Strict_MAE_CI_Lower": (
            float(bootstrap_summary.loc[
                bootstrap_summary["Metric"] == "Strict_MAE", "CI_Lower"
            ].iloc[0]) if len(bootstrap_summary) else np.nan
        ),
        "Strict_MAE_CI_Upper": (
            float(bootstrap_summary.loc[
                bootstrap_summary["Metric"] == "Strict_MAE", "CI_Upper"
            ].iloc[0]) if len(bootstrap_summary) else np.nan
        ),
        "Strict_RMSE_CI_Lower": (
            float(bootstrap_summary.loc[
                bootstrap_summary["Metric"] == "Strict_RMSE", "CI_Lower"
            ].iloc[0]) if len(bootstrap_summary) else np.nan
        ),
        "Strict_RMSE_CI_Upper": (
            float(bootstrap_summary.loc[
                bootstrap_summary["Metric"] == "Strict_RMSE", "CI_Upper"
            ].iloc[0]) if len(bootstrap_summary) else np.nan
        ),
        "Strict_LLL_CI_Lower": (
            float(bootstrap_summary.loc[
                bootstrap_summary["Metric"] == "Strict_LLL", "CI_Lower"
            ].iloc[0]) if len(bootstrap_summary) else np.nan
        ),
        "Strict_LLL_CI_Upper": (
            float(bootstrap_summary.loc[
                bootstrap_summary["Metric"] == "Strict_LLL", "CI_Upper"
            ].iloc[0]) if len(bootstrap_summary) else np.nan
        ),
        "Trajectory_LLL_CI_Lower": (
            float(bootstrap_summary.loc[
                bootstrap_summary["Metric"] == "Trajectory_LLL", "CI_Lower"
            ].iloc[0]) if len(bootstrap_summary) else np.nan
        ),
        "Trajectory_LLL_CI_Upper": (
            float(bootstrap_summary.loc[
                bootstrap_summary["Metric"] == "Trajectory_LLL", "CI_Upper"
            ].iloc[0]) if len(bootstrap_summary) else np.nan
        ),
    }]).to_csv(summary_path, index=False)

    print(
        f"\nSaved:\n"
        f"  {fold_path}\n"
        f"  {oof_path}\n"
        f"  {summary_path}\n"
        f"  {catastrophic_path}\n"
        f"  {horizon_path}\n"
        f"  {patient_path}\n"
        f"  {excluded_path}\n"
        f"  {dynamic_manifest_path}\n"
        f"  {outer_profile_path}\n"
        f"  {per_patient_path}\n"
        f"  {bootstrap_summary_path}\n"
        f"  {bootstrap_dist_path}"    )

    return (
        fold_table, oof, pooled, patient_future, patient_traj,
        catastrophic, horizon_summary, patient_error_summary,
        excluded_patients, dynamic_manifest,
        per_patient_metrics, bootstrap_summary,
    )



# =============================================================================
# OPTIONAL ROBUSTNESS ANALYSES
# =============================================================================
def run_optional_repeated_balanced_cv(base_cfg, df):
    """
    OPTIONAL and expensive.

    Repeats the exact frozen v9/v10 model with different balanced-fold seeds.
    The architecture/loss/hyperparameters do not change. This measures
    sensitivity to patient-fold assignment.

    Each repeat writes its own files. A compact across-repeat summary is saved.
    """
    if not base_cfg.enable_repeated_balanced_cv:
        return pd.DataFrame()

    rows = []
    print(
        f"\nOPTIONAL: repeated balanced {base_cfg.n_folds}-fold CV "
        f"for {base_cfg.repeated_cv_repeats} repeats"
    )

    for r in range(int(base_cfg.repeated_cv_repeats)):
        cfg_r = deepcopy(base_cfg)
        cfg_r.enable_repeated_balanced_cv = False
        cfg_r.enable_lopo_cv = False
        cfg_r.enable_patient_bootstrap = bool(
            base_cfg.bootstrap_optional_runs
        )
        cfg_r.seed = (
            int(base_cfg.seed)
            + (r + 1) * int(base_cfg.repeated_cv_seed_stride)
        )
        cfg_r.output_prefix = (
            f"{base_cfg.output_prefix}_repeat_{r+1:02d}"
        )

        print(
            f"\n----- REPEATED CV {r+1}/{base_cfg.repeated_cv_repeats} "
            f"| fold seed={cfg_r.seed} -----"
        )
        result = run_recovery(cfg_r, df)

        pooled = result[2]
        patient_future = result[3]
        patient_traj = result[4]

        rows.append({
            "Repeat": r + 1,
            "Seed": cfg_r.seed,
            "Strict_MAE": pooled["MAE"],
            "Strict_RMSE": pooled["RMSE"],
            "Strict_LLL": pooled["LLL"],
            "PatientFuture_MAE": patient_future["MAE"],
            "PatientFuture_RMSE": patient_future["RMSE"],
            "PatientFuture_LLL": patient_future["LLL"],
            "Trajectory_MAE": patient_traj["MAE"],
            "Trajectory_RMSE": patient_traj["RMSE"],
            "Trajectory_LLL": patient_traj["LLL"],
        })

    out = pd.DataFrame(rows)
    path = (
        f"/kaggle/working/"
        f"{base_cfg.output_prefix}_repeated_cv_summary.csv"
    )
    out.to_csv(path, index=False)

    print("\nREPEATED BALANCED CV SUMMARY")
    for c in [
        "Strict_MAE", "Strict_RMSE", "Strict_LLL",
        "Trajectory_MAE", "Trajectory_RMSE", "Trajectory_LLL",
    ]:
        print(
            f"{c}: {out[c].mean():.5f} ± "
            f"{out[c].std(ddof=1):.5f}"
        )
    print(f"Saved: {path}")
    return out


def run_optional_lopo(base_cfg, df):
    """
    OPTIONAL and VERY expensive.

    Leave-one-patient-out evaluation of the same frozen strict model.
    No architecture or loss changes are made.
    """
    if not base_cfg.enable_lopo_cv:
        return None

    eligible, _ = strict_eligible_patients(df, base_cfg)
    n_patients = len(eligible)
    if n_patients < 3:
        raise RuntimeError("Too few strict patients for LOPO.")

    cfg_l = deepcopy(base_cfg)
    cfg_l.enable_repeated_balanced_cv = False
    cfg_l.enable_lopo_cv = False
    cfg_l.enable_patient_bootstrap = bool(
        base_cfg.bootstrap_optional_runs
    )
    cfg_l.n_folds = n_patients
    cfg_l.output_prefix = (
        f"{base_cfg.output_prefix}_LOPO_{n_patients}patients"
    )

    print(
        f"\nOPTIONAL: LEAVE-ONE-PATIENT-OUT CV "
        f"({n_patients} outer folds)."
    )
    return run_recovery(cfg_l, df)

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    set_seed(config.seed)

    df = pd.read_csv(
        os.path.join(config.data_path, "train.csv")
    )
    required = {
        "Patient", "Weeks", "FVC",
        "Percent", "Age", "Sex",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"train.csv missing required columns: {sorted(missing)}"
        )

    print("\n" + "=" * 110)
    print("FINAL STABLE PRIMARY RUN")
    print("Frozen v9 predictor + strict CT protocol + balanced CV")
    print("=" * 110)

    results = run_recovery(config, df)

    # OPTIONAL analyses are deliberately after the primary result.
    if config.enable_repeated_balanced_cv:
        repeated_results = run_optional_repeated_balanced_cv(
            config, df
        )

    if config.enable_lopo_cv:
        lopo_results = run_optional_lopo(config, df)
