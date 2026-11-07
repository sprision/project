"""Experimental PatchCore implementation and benchmark on MVTec AD screw.

Evaluates PatchCore on the exact identical calibration and held-out splits
defined in scripts/calibrate_thresholds.py (seed 42, 50/50).
"""
import json
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from sklearn.metrics import roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def extract_features(backbone, img_tensor: torch.Tensor) -> torch.Tensor:
    """Extract neighborhood-aware patch features from layer2 and layer3."""
    with torch.no_grad():
        feats = backbone(img_tensor)
        f1, f2 = feats[0], feats[1] # layer2: (B, 128, 28, 28), layer3: (B, 256, 14, 14)
        f2_up = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        f_cat = torch.cat([f1, f2_up], dim=1) # (B, 384, 28, 28)
        f_pooled = F.avg_pool2d(f_cat, kernel_size=3, stride=1, padding=1) # (B, 384, 28, 28)
    return f_pooled

def build_memory_bank(backbone, train_paths: list[Path], coreset_size: int = 5000) -> torch.Tensor:
    """Extract patch features from all normal training images and subsample memory bank."""
    print(f"Building PatchCore memory bank from {len(train_paths)} training images...")
    t0 = time.time()
    patch_list = []
    
    batch_tensors = []
    for i, p in enumerate(train_paths):
        im = Image.open(p).convert("RGB")
        batch_tensors.append(transform(im))
        if len(batch_tensors) == 32 or i == len(train_paths) - 1:
            batch = torch.stack(batch_tensors, dim=0).to(device)
            f_pooled = extract_features(backbone, batch) # (B, 384, 28, 28)
            B, C, H, W = f_pooled.shape
            patches = f_pooled.permute(0, 2, 3, 1).reshape(-1, C) # (B*H*W, 384)
            patch_list.append(patches.cpu())
            batch_tensors = []

    all_patches = torch.cat(patch_list, dim=0) # (~250,880, 384)
    print(f"Total extracted normal patches: {all_patches.shape[0]} in {time.time()-t0:.1f}s")
    
    # Subsample memory bank to coreset_size using random or k-center greedy selection
    # In practice, random uniform sampling over 250k patches yields virtually identical AUROC to greedy coreset
    # while taking 0.1 seconds instead of minutes.
    rng = np.random.default_rng(42)
    indices = rng.choice(all_patches.shape[0], size=min(coreset_size, all_patches.shape[0]), replace=False)
    memory_bank = all_patches[indices].to(device)
    print(f"Subsampled memory bank shape: {memory_bank.shape}")
    return memory_bank

def compute_patchcore_score_map(backbone, memory_bank: torch.Tensor, img_tensor: torch.Tensor) -> tuple[float, np.ndarray]:
    """Compute nearest neighbor patch distance and image anomaly score."""
    with torch.no_grad():
        f_pooled = extract_features(backbone, img_tensor) # (1, 384, 28, 28)
        B, C, H, W = f_pooled.shape
        test_patches = f_pooled.permute(0, 2, 3, 1).reshape(-1, C) # (784, 384)
        
        # Compute squared Euclidean distance to memory bank: (test_patches - memory_bank)^2
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x * y^T
        x_norm = (test_patches ** 2).sum(dim=1, keepdim=True) # (784, 1)
        y_norm = (memory_bank ** 2).sum(dim=1, keepdim=True).T # (1, M)
        dists = torch.clamp(x_norm + y_norm - 2.0 * torch.matmul(test_patches, memory_bank.T), min=0.0) # (784, M)
        min_dists, _ = torch.min(dists, dim=1) # (784,)
        patch_scores = torch.sqrt(min_dists) # (784,)
        
        score_map = patch_scores.reshape(1, 1, H, W)
        score_map_up = F.interpolate(score_map, size=(224, 224), mode="bilinear", align_corners=False)
        am = score_map_up.squeeze().cpu().numpy()
        image_score = float(patch_scores.max().item())
        
    return image_score, am

def main():
    backbone = timm.create_model("resnet18", pretrained=True, features_only=True, out_indices=(2, 3)).to(device)
    backbone.eval()

    train_paths = sorted(Path("data/raw/mvtec_ad/screw/train/good").glob("*.png"))
    memory_bank = build_memory_bank(backbone, train_paths, coreset_size=10000)

    # Collect test paths identically
    raw_root = Path("data/raw/mvtec_ad/screw/test")
    paths = []
    for dd in sorted(raw_root.iterdir()):
        if not dd.is_dir(): continue
        is_good = (dd.name == "good")
        for p in sorted(dd.glob("*.png")):
            paths.append((is_good, p))

    # Split identically to calibration protocol (seed 42, frac 0.5)
    rng = np.random.default_rng(42)
    cal_paths, ev_paths = [], []
    for is_good in (True, False):
        idx = [i for i, (g, _) in enumerate(paths) if g == is_good]
        rng.shuffle(idx)
        k = int(round(len(idx) * 0.5))
        cal_paths += [paths[i] for i in idx[:k]]
        ev_paths += [paths[i] for i in idx[k:]]

    print(f"\nCalibration split: {len(cal_paths)} images | Held-out evaluation split: {len(ev_paths)} images")

    # 1. Evaluate Calibration split
    print("\n--- Evaluating Calibration Split ---")
    cal_scores, cal_labels = [], []
    for is_good, p in cal_paths:
        im = Image.open(p).convert("RGB")
        s, _ = compute_patchcore_score_map(backbone, memory_bank, transform(im).unsqueeze(0).to(device))
        cal_scores.append(s)
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

    # 2. Evaluate Strict Held-Out Evaluation split
    print("\n--- Evaluating Strict Held-Out Split ---")
    ev_scores, ev_labels, ev_maps = [], [], []
    for is_good, p in ev_paths:
        im = Image.open(p).convert("RGB")
        s, am = compute_patchcore_score_map(backbone, memory_bank, transform(im).unsqueeze(0).to(device))
        ev_scores.append(s)
        ev_labels.append(0 if is_good else 1)
        ev_maps.append((p, am))

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

    print(f"Held-Out Results for PatchCore (Screw):")
    print(f"  Images: {len(ev_labels)} (Good: {(ev_labels==0).sum()}, Defect: {(ev_labels==1).sum()})")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Sensitivity:       {sens:.4f}")
    print(f"  Specificity:       {spec:.4f}")
    print(f"  False Positive Rate: {fpr:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Image AUROC:       {ev_auroc:.4f}")

    # Compute Pixel AUROC on held-out evaluation
    print("\nComputing Pixel AUROC on Held-Out Split...")
    all_pixel_preds, all_pixel_labels = [], []
    gt_root = Path("data/raw/mvtec_ad/screw/ground_truth")
    for (p, am), y in zip(ev_maps, ev_labels):
        if y == 0:
            gt = np.zeros((224, 224), dtype=np.uint8)
        else:
            dname = p.parent.name
            gt_p = gt_root / dname / (p.stem + "_mask.png")
            if gt_p.exists():
                gt_im = Image.open(gt_p).convert("L").resize((224, 224), Image.Resampling.NEAREST)
                gt = (np.array(gt_im) > 0).astype(np.uint8)
            else:
                continue
        all_pixel_preds.append(am.flatten())
        all_pixel_labels.append(gt.flatten())

    if all_pixel_preds:
        all_pixel_preds = np.concatenate(all_pixel_preds)
        all_pixel_labels = np.concatenate(all_pixel_labels)
        pix_auroc = roc_auc_score(all_pixel_labels, all_pixel_preds)
        print(f"  Pixel AUROC:       {pix_auroc:.4f}")

if __name__ == "__main__":
    main()

