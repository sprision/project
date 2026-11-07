from pathlib import Path
from PIL import Image
import numpy as np
import cv2

def analyze_wires(img_path):
    arr = np.array(Image.open(img_path))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = ((H < 10) | (H > 170)) & (S > 70) & (V > 50)
    yellow = (H >= 15) & (H <= 35) & (S > 70) & (V > 70)
    green = (H >= 35) & (H <= 85) & (S > 50) & (V > 50)
    blue = (H >= 85) & (H <= 130) & (S > 50) & (V > 50)
    return {'red': int(np.sum(red)), 'yellow': int(np.sum(yellow)), 'green': int(np.sum(green)), 'blue': int(np.sum(blue))}

for folder in ['good', 'missing_wire', 'missing_cable', 'poke_insulation']:
    paths = sorted((Path('data/raw/mvtec_ad/cable/test') / folder).glob('*.png'))[:5]
    print(f'=== {folder} ===')
    for p in paths:
        w = analyze_wires(p)
        print(f"  {p.name}: red={w['red']:6d}, yellow={w['yellow']:6d}, green={w['green']:6d}, blue={w['blue']:6d}")

