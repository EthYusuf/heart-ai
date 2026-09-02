# =============================================================================
# predict.py — Eğitilmiş modelle çıkarım (inference)
# -----------------------------------------------------------------------------
# En yeni (veya --run ile seçilen) koşunun best_model.pth dosyasını yükler,
# girdi görüntü(ler)ini eğitimdeki doğrulama ön işlemesiyle birebir aynı
# zincirden geçirir ve tahmini kaydeder.
#
# Kullanım:
#   python predict.py data_local/images/img_000.png   # tek görüntü
#   python predict.py data_local/images               # klasördeki tüm görüntüler
#   python predict.py                                 # yolu etkileşimli sorar
#   python predict.py <yol> --run local_debug_v1_20260901_232858
#
# Çıktı (segmentasyon): predictions/<ad>_pred.png — gri görüntü üzerine
# kırmızı tahmin maskesi + maske alanı yüzdesi.
# Çıktı (sınıflandırma): sınıf adı + olasılıklar konsola yazılır.
# =============================================================================

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")  # pencere açmadan dosyaya çiz
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import Dataset as MonaiDataset

from config import Config
from datasets import get_transforms
from models import model_factory

EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".nii", ".gz", ".tif", ".tiff"}


def latest_run(outputs_dir: Path) -> Path:
    """outputs_local altında best_model.pth içeren en yeni koşu klasörü."""
    runs = sorted(
        (d for d in outputs_dir.glob("*") if (d / "best_model.pth").exists()),
        key=lambda d: d.stat().st_mtime,
    )
    if not runs:
        sys.exit("Hata: outputs_local altında eğitilmiş model yok. Önce: python run_local.py")
    return runs[-1]


def build_image_transform(cfg) -> object:
    """Çıkarım ön işlemesi: eğitimdeki val zincirinin image kolonu.

    get_transforms'a TASK='classification' verilirse yalnız 'image' anahtarını
    içeren zincir üretilir; görüntüye uygulanan dönüşümler (yoğunluk ölçekleme,
    normalize, resize) segmentasyon eğitimiyle birebir aynı kalır. Çıkarımda
    maske olmadığı için label dönüşümlerine gerek yoktur.
    """
    tfm_cfg = SimpleNamespace(
        TASK="classification",  # image-only zincir için
        IN_CHANNELS=cfg.IN_CHANNELS,
        INTENSITY_MIN=cfg.INTENSITY_MIN,
        INTENSITY_MAX=cfg.INTENSITY_MAX,
        IMAGE_SIZE=cfg.IMAGE_SIZE,
    )
    return get_transforms(tfm_cfg, mode="val")


def predict_one(model, tfm, cfg, image_path: Path, out_dir: Path) -> None:
    ds = MonaiDataset(data=[{"image": str(image_path)}], transform=tfm)
    tensor = ds[0]["image"]
    if tensor.dim() == 3:  # [C,H,W] -> [1,C,H,W]
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(cfg.DEVICE)

    with torch.no_grad():
        logits = model(tensor)

    if cfg.TASK == "segmentation":
        prob = torch.softmax(logits, dim=1)[:, 1]          # riskli sınıf olasılığı
        mask = (prob > 0.5)[0].cpu().numpy()
        coverage = float(mask.mean() * 100)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(tensor[0, 0].cpu().numpy(), cmap="gray")
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.45)
        ax.set_title(f"Tahmin | maske alani: %{coverage:.1f}")
        ax.axis("off")
        out = out_dir / f"{image_path.stem}_pred.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {image_path.name}: maske alani %{coverage:.1f} -> {out}")
    else:
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        names = cfg.CLASS_NAMES or [f"Sinif {i}" for i in range(cfg.NUM_CLASSES)]
        top = ", ".join(f"{names[i]}={probs[i]:.3f}" for i in range(min(3, len(probs))))
        print(f"  {image_path.name}: {names[idx]} | {top}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CardioPulse degil, chest pipeline cikarim araci")
    parser.add_argument("input", nargs="?", help="goruntu dosyasi veya klasor (bos ise sorar)")
    parser.add_argument("--run", help="outputs_local altindaki kosu klasoru (varsayilan: en yenisi)")
    args = parser.parse_args()

    outputs_dir = Path(__file__).resolve().parent / "outputs_local"
    run_dir = Path(args.run) if args.run else latest_run(outputs_dir)
    cfg = Config.load(run_dir / "config.json")

    model = model_factory(
        cfg.MODEL_NAME, cfg.IN_CHANNELS, cfg.NUM_CLASSES,
        cfg.DEVICE, cfg.IMAGE_SIZE, False,  # cikarimda pretrained indirmeye gerek yok
    )
    state = torch.load(run_dir / "best_model.pth", map_location=cfg.DEVICE)
    model.load_state_dict(state["model_state"])
    model.to(cfg.DEVICE).eval()
    print(f"Model yuklendi: {run_dir.name} | task={cfg.TASK} model={cfg.MODEL_NAME} "
          f"cihaz={cfg.DEVICE} | egitim metriği={state.get('metric', 0.0):.4f}")

    inp = args.input or input("Goruntu yolu (dosya veya klasor): ").strip().strip('"')
    path = Path(inp)
    if not path.exists():
        sys.exit(f"Hata: '{inp}' bulunamadi")

    if path.is_dir():
        paths = sorted(p for p in path.rglob("*") if p.suffix.lower() in EXTENSIONS)
    else:
        paths = [path]
    if not paths:
        sys.exit(f"Hata: '{inp}' icinde goruntu bulunamadi")

    out_dir = run_dir / "predictions"
    out_dir.mkdir(exist_ok=True)
    tfm = build_image_transform(cfg)

    print(f"{len(paths)} goruntu isleniyor -> cikti klasoru: {out_dir}")
    for p in paths:
        predict_one(model, tfm, cfg, p, out_dir)


if __name__ == "__main__":
    main()
