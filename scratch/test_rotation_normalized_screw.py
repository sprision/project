"""Experimental Rotation Normalization evaluation on MVTec AD screw.

Tests whether rotating screws to canonical vertical orientation before feature extraction
improves PaDiM and PatchCore on the identical calibration and held-out splits.
"""
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def align_screw_image(img: Image.Image) -> Image.Image:
    """Detect screw orientation and rotate to canonical vertical orientation with head at top."""
    arr = np.array(img.convert("L"))
    mask = arr < 155
    y_idx, x_idx = np.where(mask)
    if len(y_idx) < 100:
        return img
    cy, cx = np.mean(y_idx), np.mean(x_idx)
    coords = np.stack([y_idx - cy, x_idx - cx], axis=1)
    evals, evecs = np.linalg.eigh(np.cov(coords, rowvar=False))
    major_evec = evecs[:, np.argmax(evals)]
    angle_deg = np.degrees(np.arctan2(major_evec[0], major_evec[1]))
    
    rot_angle = - (angle_deg - 90.0)
    bg_color = int(np.median(arr[~mask]))
    rotated = img.rotate(rot_angle, resample=Image.Resampling.BILINEAR, center=(cx, cy), fillcolor=bg_color)
    
    rot_arr = np.array(rotated.convert("L"))
    rot_mask = rot_arr < 155
    ry, rx = np.where(rot_mask)
    if len(ry) > 0:
        ymin, ymax = np.min(ry), np.max(ry)
        h = ymax - ymin
        top_slice = rot_mask[ymin:ymin + int(h * 0.35), :]
        bot_slice = rot_mask[ymax - int(h * 0.35):ymax, :]
        top_mass = np.sum(top_slice)
        bot_mass = np.sum(bot_slice)
        if bot_mass > top_mass * 1.15:
            rotated = rotated.rotate(180, resample=Image.Resampling.BILINEAR, fillcolor=bg_color)
            
    return rotated

def main():
    print("=== TESTING ROTATION NORMALIZATION ON SCREW ===")
    
    aligned_dir = Path("scratch/aligned_screw")
    aligned_train_dir = aligned_dir / "train" / "good"
    aligned_train_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Align training images and save to disk
    train_paths = sorted(Path("data/raw/mvtec_ad/screw/train/good").glob("*.png"))
    print(f"Aligning {len(train_paths)} training images...")
    t0 = time.time()
    aligned_train_paths = []
    for p in train_paths:
        dst = aligned_train_dir / p.name
        if not dst.exists():
            im_aligned = align_screw_image(Image.open(p).convert("RGB"))
            im_aligned.save(dst)
        aligned_train_paths.append(dst)
    print(f"Aligned training set in {time.time()-t0:.1f}s")

    # 2. Fit a PaDiM model on rotation-normalized training images
    from src.app.vision.model import PaDiM
    aligned_padim = PaDiM(device=device, n_features=100)
    print("Fitting PaDiM on aligned training images...")
    t0 = time.time()
    aligned_padim.fit(aligned_train_paths)
    print(f"Fitted aligned PaDiM in {time.time()-t0:.1f}s")

    # 3. Collect test paths identically
    raw_root = Path("data/raw/mvtec_ad/screw/test")
    paths = []
    for dd in sorted(raw_root.iterdir()):
        if not dd.is_dir(): continue
        is_good = (dd.name == "good")
        for p in sorted(dd.glob("*.png")):
            paths.append((is_good, p))

    rng = np.random.default_rng(42)
    cal_paths, ev_paths = [], []
    for is_good in (True, False):
        idx = [i for i, (g, _) in enumerate(paths) if g == is_good]
        rng.shuffle(idx)
        k = int(round(len(idx) * 0.5))
        cal_paths += [paths[i] for i in idx[:k]]
        ev_paths += [paths[i] for i in idx[k:]]

    # Evaluate on Calibration Split (aligned)
    print("\n--- Evaluating Calibration Split (Aligned PaDiM) ---")
    cal_scores, cal_labels = [], []
    for is_good, p in cal_paths:
        im_aligned = align_screw_image(Image.open(p).convert("RGB"))
        am = aligned_padim.anomaly_map(im_aligned)
        cal_scores.append(float(am.max()))
        cal_labels.append(0 if is_good else 1)

    cal_scores = np.array(cal_scores)
    cal_labels = np.array(cal_labels)

    # Pick threshold on cal
    order = np.argsort(cal_scores)
    sc, yc = cal_scores[order], cal_labels[order]
    mids = (sc[:-1] + sc[1:]) / 2.0
    candidates = np.concatenate([[float(sc[0]) - 1.0], mids, [float(sc[-1]) + 1.0]])
    best_t, best_bal = None, -1.0
    for t in candidates:
        pred = (cal_scores >= t).astype(int)
        tp = int(((pred == 1) & (cal_labels == 1)).sum())
        fn = int(((pred == 0) & (cal_labels == 1)).sum())
        fp = int(((pred == 1) & (cal_labels == 0)).sum())
        tn = int(((pred == 0) & (cal_labels == 0)).sum())
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        bal = 0.5 * (sens + spec)
        if bal > best_bal:
            best_bal, best_t = bal, float(t)

    print(f"Selected Threshold on Cal: {best_t:.4f} (Cal Balanced Accuracy: {best_bal:.4f})")
    cal_auroc = roc_auc_score(cal_labels, cal_scores)
    print(f"Cal Image AUROC: {cal_auroc:.4f}")

    # Evaluate on Strict Held-Out Split (aligned)
    print("\n--- Evaluating Strict Held-Out Split (Aligned PaDiM) ---")
    ev_scores, ev_labels = [], []
    for is_good, p in ev_paths:
        im_aligned = align_screw_image(Image.open(p).convert("RGB"))
        am = aligned_padim.anomaly_map(im_aligned)
        ev_scores.append(float(am.max()))
        ev_labels.append(0 if is_good else 1)

    ev_scores = np.array(ev_scores)
    ev_labels = np.array(ev_labels)

    pred = (ev_scores >= best_t).astype(int)
    tp = int(((pred == 1) & (ev_labels == 1)).sum())
    fn = int(((pred == 0) & (ev_labels == 1)).sum())
    fp = int(((pred == 1) & (ev_labels == 0)).sum())
    tn = int(((pred == 0) & (ev_labels == 0)).sum())
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    fpr = fp / max(1, fp + tn)
    bal_acc = 0.5 * (sens + spec)
    ev_auroc = roc_auc_score(ev_labels, ev_scores)

    print(f"Held-Out Results for Rotation-Normalized PaDiM (Screw):")
    print(f"  Images: {len(ev_labels)} (Good: {(ev_labels==0).sum()}, Defect: {(ev_labels==1).sum()})")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Sensitivity:       {sens:.4f}")
    print(f"  Specificity:       {spec:.4f}")
    print(f"  False Positive Rate: {fpr:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Image AUROC:       {ev_auroc:.4f}")

if __name__ == "__main__":
    main()
