# =============================================================================
# prepare_local_testdata.py — Yerel CPU duman testi için sentetik veri üretici
# -----------------------------------------------------------------------------
# Pipeline'ı Kaggle olmadan, veri olmadan da uçtan uca çalıştırabilmek için
# 40 adet yapay "X-ray benzeri" görüntü + ikili maske üretir:
#   data_local/images/img_XXX.png   (0-255 gri tonlamalı, akciğer kütlesi var)
#   data_local/masks/img_XXX.png    (0/255 ikili maske, görüntüyle aynı ad)
# CSV yok — build_file_list klasör taramasıyla eşleşen maskeleri bulur.
# Gerçek veri geldiğinde bu klasörü silin; bilimsel kullanım amaçlı DEĞİLDİR.
# =============================================================================

from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent / "data_local"
IMAGE_DIR = BASE_DIR / "images"
MASK_DIR = BASE_DIR / "masks"
N_SAMPLES = 40
SIZE = 320
SEED = 42


def make_pair(rng: np.random.Generator, idx: int) -> tuple[Image.Image, Image.Image]:
    """Tek yapay göğüs grafisi + akciğer maskesi üretir."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    cx, cy = SIZE / 2, SIZE / 2

    # Gövde: ortalanmış yumuşak elips, kenarlara doğru koyulaşan göğüs duvarı
    body = np.exp(-(((xx - cx) / (SIZE * 0.38)) ** 2 + ((yy - cy) / (SIZE * 0.46)) ** 2))

    # Akciğerler: gövde içinde iki yumuşak blob (hasta bazlı boyut/konum varyasyonu)
    spread = rng.uniform(0.10, 0.14)
    left = np.exp(-(((xx - cx + SIZE * 0.13) / (SIZE * spread)) ** 2 + ((yy - cy + 10) / (SIZE * 0.20)) ** 2))
    right = np.exp(-(((xx - cx - SIZE * 0.13) / (SIZE * spread)) ** 2 + ((yy - cy + 10) / (SIZE * 0.20)) ** 2))

    # Kaburga gölgeleri: yatayda periyodik çizgiler
    ribs = 0.08 * np.sin(yy / rng.uniform(9, 13) + rng.uniform(0, 3)) * body

    img = 40 * body + 150 * (left + right) * 0.35 + 40 * ribs
    img += rng.normal(0, 6, img.shape)                      # film grain
    img = np.clip(img, 0, 255).astype(np.uint8)

    mask = ((left + right) > 0.35).astype(np.uint8) * 255   # ikili akciğer alanı

    return Image.fromarray(img, mode="L"), Image.fromarray(mask, mode="L")


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    for i in range(N_SAMPLES):
        img, mask = make_pair(rng, i)
        img.save(IMAGE_DIR / f"img_{i:03d}.png")
        mask.save(MASK_DIR / f"img_{i:03d}.png")

    print(f"{N_SAMPLES} sentetik görüntü/maske çifti üretildi → {BASE_DIR}")


if __name__ == "__main__":
    main()
