import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app.vision.model import PaDiM

p = "data/index/vision/cable/padim.pkl"
model = PaDiM.load(p, device="cuda")

def get_multiscale_scores(p_img):
    im = Image.open(p_img)
    x = model._tf(im.convert("RGB")).unsqueeze(0).to(model.device)
    emb = model._embed(x).cpu()
    c, h, w = emb.shape[1], emb.shape[2], emb.shape[3]
    emb = emb.reshape(1, c, h * w)
    delta = (emb - model.means_.unsqueeze(0))
    d = delta[0]
    if model._inv_t is None:
        model._inv_t = model.inv_cov_.permute(2, 0, 1).contiguous()
    dt = d.permute(1, 0).contiguous()
    m2 = torch.bmm(dt.unsqueeze(1), model._inv_t).squeeze(1) * dt
    m = torch.sqrt(torch.clamp(m2.sum(dim=1), min=0))
    raw_map = m.reshape(h, w).numpy()
    return float(raw_map.max()), float(gaussian_filter(raw_map, sigma=1).max()), float(gaussian_filter(raw_map, sigma=4).max())

print("=== POKE INSULATION (10 images) ===")
poke_paths = sorted(Path("data/raw/mvtec_ad/cable/test/poke_insulation").glob("*.png"))
for p in poke_paths:
    raw, s1, s4 = get_multiscale_scores(p)
    print(f"  {p.name}: raw={raw:5.2f} | sigma1={s1:5.2f} | sigma4={s4:5.2f}")

print("\n=== GOOD SAMPLES (sample 10 images) ===")
good_paths = sorted(Path("data/raw/mvtec_ad/cable/test/good").glob("*.png"))[:10]
for p in good_paths:
    raw, s1, s4 = get_multiscale_scores(p)
    print(f"  {p.name}: raw={raw:5.2f} | sigma1={s1:5.2f} | sigma4={s4:5.2f}")
