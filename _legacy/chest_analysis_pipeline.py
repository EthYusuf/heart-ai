# =============================================================================
# Göğüs Röntgeni Analizi - Derin Öğrenme Pipeline'ı
# Medical Chest X-Ray / CT Deep Learning Pipeline
# -----------------------------------------------------------------------------
# Yazar       : [Adınız]
# Kaggle Ortamı için Optimize Edilmiştir (40 GB Veri Seti)
# Kullanılan Kütüphaneler: PyTorch, MONAI, torchvision, scikit-learn
# Görev       : Göğüs hastalıklarının segmentasyonu / sınıflandırılması
# =============================================================================

# ─── Standart Kütüphaneler ───────────────────────────────────────────────────
import os
import gc
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

# ─── Sayısal / Görüntü İşleme ────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ─── PyTorch ─────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast   # Otomatik karma hassasiyet (AMP)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ─── MONAI — Medikal Görüntüleme Kütüphanesi ─────────────────────────────────
# Kaggle'da yüklü değilse: !pip install monai
import monai
from monai.transforms import (
    Compose,
    LoadImaged,          # Görüntüyü disk'ten tembel (lazy) yükler
    EnsureChannelFirstd, # Kanal boyutunu öne taşır
    ScaleIntensityRanged,# Yoğunluk ölçekleme / normalizasyon
    Resized,             # Yeniden boyutlandırma
    RandFlipd,           # Rastgele yatay / dikey çevirme
    RandRotate90d,       # 90° katları rastgele döndürme
    RandAffined,         # Afin dönüşüm (zoom, kaydırma)
    RandGaussianNoised,  # Gaussian gürültü ekleme
    RandAdjustContrastd, # Kontrast ayarı
    ToTensord,           # NumPy → PyTorch Tensor
    CenterSpatialCropd,  # Merkezi kırpma
    NormalizeIntensityd, # İstatistiksel normalizasyon
)
from monai.networks.nets import UNet, DenseNet121
from monai.losses import DiceCELoss, FocalLoss
from monai.metrics import DiceMetric, ROCAUCMetric
from monai.data import CacheDataset, PersistentDataset, DataLoader as MonaiDataLoader
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# ─── Uyarıları Kapat ─────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# 1. YAPILANDIRMA (Configuration)
# =============================================================================

@dataclass
class Config:
    """
    Tüm hiperparametreler ve yol tanımlamaları tek merkezde toplandı.
    Deney parametrelerini değiştirmek için yalnızca bu sınıfı düzenleyin.
    """

    # ── Veri Yolları (Kaggle /kaggle/input/ altında güncelleyin) ─────────────
    DATA_ROOT: Path = Path("/kaggle/input/<DATASET-KLASORU>")
    IMAGE_DIR: Path = DATA_ROOT / "images"          # Görüntü klasörü
    MASK_DIR:  Path = DATA_ROOT / "masks"           # Segmentasyon maskeleri (varsa)
    CSV_PATH:  Path = DATA_ROOT / "labels.csv"      # Etiket dosyası
    OUTPUT_DIR: Path = Path("/kaggle/working/outputs")

    # ── Görev Tipi ───────────────────────────────────────────────────────────
    # "segmentation" → U-Net  |  "classification" → DenseNet121
    TASK: str = "segmentation"

    # ── Model Parametreleri ───────────────────────────────────────────────────
    IMAGE_SIZE: Tuple[int, int] = (512, 512)   # Görüntü boyutu (H x W)
    IN_CHANNELS: int = 1                        # Gri tonlamalı = 1, RGB = 3
    NUM_CLASSES: int = 2                        # Sınıf sayısı (arka plan dahil)

    # ── Eğitim Hiperparametreleri ─────────────────────────────────────────────
    EPOCHS: int = 50
    BATCH_SIZE: int = 4                         # GPU belleğine göre ayarlayın
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    VAL_RATIO: float = 0.15                     # %15 validasyon ayrımı
    TEST_RATIO: float = 0.10                    # %10 test ayrımı

    # ── Bellek Optimizasyonu ──────────────────────────────────────────────────
    NUM_WORKERS: int = 2                        # DataLoader iş parçacığı sayısı
    PIN_MEMORY: bool = True                     # CPU→GPU veri transferini hızlandırır
    USE_AMP: bool = True                        # FP16 karma hassasiyet (VRAM tasarrufu)
    CACHE_RATE: float = 0.0                     # 0.0 = disk'ten tembel okuma (RAM dostu)
    PREFETCH_FACTOR: int = 2                    # DataLoader ön yükleme faktörü

    # ── Kaydetme & Log ─────────────────────────────────────────────────────────
    SAVE_BEST_ONLY: bool = True
    LOG_INTERVAL: int = 10                      # Her N iterasyonda bir log yaz
    SEED: int = 42

    # ── Çalışma Zamanında Türetilecek Değerler ────────────────────────────────
    DEVICE: torch.device = field(init=False)
    CHECKPOINT_PATH: Path = field(init=False)

    def __post_init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CHECKPOINT_PATH = self.OUTPUT_DIR / "best_model.pth"
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        set_determinism(seed=self.SEED)
        logger.info(f"Cihaz: {self.DEVICE} | AMP: {self.USE_AMP} | Görev: {self.TASK}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                        f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


CFG = Config()


# =============================================================================
# 2. VERİ SETİ — Tembel Yükleme (Lazy Loading) ile OOM Önleme
# =============================================================================

class ChestDataset(Dataset):
    """
    Kaggle'daki 40 GB görüntü veri setini belleğe TAMAMEN YÜKLEMEDEN işler.
    Her görüntü yalnızca ihtiyaç duyulduğunda diskten okunur.

    Desteklenen formatlar: DICOM (.dcm), PNG, JPG, NIfTI (.nii.gz)
    """

    def __init__(
        self,
        file_list: List[Dict[str, str]],
        transforms: Compose,
        task: str = "segmentation",
    ) -> None:
        """
        Args:
            file_list  : [{"image": "/yol/img.dcm", "label": "/yol/mask.png"}, ...]
            transforms : MONAI dönüşüm zinciri
            task       : "segmentation" veya "classification"
        """
        self.file_list  = file_list
        self.transforms = transforms
        self.task       = task
        logger.info(f"Veri seti oluşturuldu: {len(self.file_list)} örnek")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Tek bir örneği diskten okur ve dönüşümleri uygular.
        Bellek birikmesini önlemek için Python'ın çöp toplayıcısı
        her adımda kullanılmayan tensörleri serbest bırakır.
        """
        sample = self.file_list[idx]

        try:
            # MONAI dönüşüm zinciri: okuma + ön işleme + artırma
            data = self.transforms(sample)
        except Exception as e:
            logger.warning(f"[idx={idx}] Görüntü işlenemedi: {sample.get('image', '?')} → {e}")
            # Hatalı örnekte komşuya git (eğitimi durdurmaz)
            return self.__getitem__((idx + 1) % len(self.file_list))

        return data


def build_file_list(cfg: Config) -> Tuple[List, List, List]:
    """
    CSV veya klasör taramasından (image, label) çiftleri oluşturur.
    Eğitim / validasyon / test bölmesi burada yapılır.

    Returns:
        train_files, val_files, test_files
    """
    # ── Seçenek A: CSV'den yükle ──────────────────────────────────────────────
    if cfg.CSV_PATH.exists():
        df = pd.read_csv(cfg.CSV_PATH)
        # Beklenen sütunlar: "image_path", "label" (veya "mask_path")
        # Gerekirse sütun isimlerini güncelleyin:
        if cfg.TASK == "classification":
            file_list = [
                {"image": str(cfg.IMAGE_DIR / row["image_path"]),
                 "label": int(row["label"])}
                for _, row in df.iterrows()
            ]
        else:
            file_list = [
                {"image": str(cfg.IMAGE_DIR / row["image_path"]),
                 "label": str(cfg.MASK_DIR  / row["mask_path"])}
                for _, row in df.iterrows()
            ]

    # ── Seçenek B: Klasör taraması ────────────────────────────────────────────
    else:
        logger.warning("CSV bulunamadı — klasör taraması yapılıyor.")
        extensions = {".dcm", ".png", ".jpg", ".jpeg", ".nii", ".nii.gz"}
        image_paths = sorted(
            p for p in cfg.IMAGE_DIR.rglob("*") if p.suffix.lower() in extensions
        )
        if cfg.TASK == "segmentation":
            file_list = [
                {"image": str(p),
                 "label": str(cfg.MASK_DIR / p.name)}  # Maske adı aynı varsayımı
                for p in image_paths
                if (cfg.MASK_DIR / p.name).exists()
            ]
        else:
            # Klasör adı = etiket (örn. /images/normal/img.png → label=0)
            class_dirs = sorted(d for d in cfg.IMAGE_DIR.iterdir() if d.is_dir())
            class_map  = {d.name: i for i, d in enumerate(class_dirs)}
            file_list  = [
                {"image": str(p), "label": class_map[p.parent.name]}
                for d in class_dirs
                for p in d.rglob("*")
                if p.suffix.lower() in extensions
            ]

    # ── Karıştır & Böl ────────────────────────────────────────────────────────
    np.random.seed(cfg.SEED)
    indices   = np.random.permutation(len(file_list))
    n         = len(indices)
    n_test    = max(1, int(n * cfg.TEST_RATIO))
    n_val     = max(1, int(n * cfg.VAL_RATIO))

    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    train_files = [file_list[i] for i in train_idx]
    val_files   = [file_list[i] for i in val_idx]
    test_files  = [file_list[i] for i in test_idx]

    logger.info(f"Toplam: {n} | Eğitim: {len(train_files)} | "
                f"Validasyon: {len(val_files)} | Test: {len(test_files)}")
    return train_files, val_files, test_files


# =============================================================================
# 3. DÖNÜŞÜM ZİNCİRLERİ (Transforms)
# =============================================================================

def get_transforms(cfg: Config, mode: str = "train") -> Compose:
    """
    MONAI Compose dönüşüm zinciri oluşturur.

    Args:
        mode : "train" → artırma aktif | "val" / "test" → yalnızca temel işleme
    """
    h, w = cfg.IMAGE_SIZE

    # ── Tüm modlarda ortak ön işleme ─────────────────────────────────────────
    base = [
        LoadImaged(
            keys=["image", "label"],
            image_only=False,
            ensure_channel_first=True,
            # DICOM için: reader="PydicomReader"
        ),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000, a_max=400,   # BT için HU penceresi; X-ray için 0-255
            b_min=0.0,   b_max=1.0,
            clip=True,
        ),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,             # Sıfır olmayan piksellerle normalize et
            channel_wise=True,
        ),
        Resized(keys=["image", "label"], spatial_size=(h, w)),
    ]

    if mode == "train":
        # ── Veri Artırma (Data Augmentation) — Yalnızca Eğitim ───────────────
        augmentations = [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
            RandAffined(
                keys=["image", "label"],
                prob=0.4,
                rotate_range=(0.15, 0.15),
                scale_range=(0.10, 0.10),
                translate_range=(20, 20),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
        ]
        pipeline = base + augmentations
    else:
        pipeline = base

    pipeline.append(ToTensord(keys=["image", "label"]))
    return Compose(pipeline)


# =============================================================================
# 4. MODEL MİMARİLERİ
# =============================================================================

def build_model(cfg: Config) -> nn.Module:
    """
    Görev tipine göre uygun modeli oluşturur ve cihaza taşır.
    """
    if cfg.TASK == "segmentation":
        # ── MONAI U-Net: 3 encoder seviyesi, residual birimler ────────────────
        model = UNet(
            spatial_dims=2,
            in_channels=cfg.IN_CHANNELS,
            out_channels=cfg.NUM_CLASSES,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=3,
            dropout=0.2,
            act="PRELU",
            norm="BATCH",
        )
        logger.info("Model: MONAI U-Net (2D Segmentasyon)")

    elif cfg.TASK == "classification":
        # ── MONAI DenseNet121: göğüs X-ray sınıflandırma için yaygın ─────────
        model = DenseNet121(
            spatial_dims=2,
            in_channels=cfg.IN_CHANNELS,
            out_channels=cfg.NUM_CLASSES,
            dropout_prob=0.3,
            pretrained=True,   # ImageNet ağırlıkları (transfer öğrenme)
        )
        logger.info("Model: MONAI DenseNet121 (2D Sınıflandırma)")

    else:
        raise ValueError(f"Bilinmeyen görev tipi: {cfg.TASK}. 'segmentation' veya 'classification' girin.")

    # Çoklu GPU desteği (Kaggle P100 / T4 çifti için)
    if torch.cuda.device_count() > 1:
        logger.info(f"DataParallel: {torch.cuda.device_count()} GPU kullanılıyor")
        model = nn.DataParallel(model)

    return model.to(cfg.DEVICE)


# =============================================================================
# 5. KAYIP FONKSİYONU & METRİKLER
# =============================================================================

def build_criterion(cfg: Config):
    """
    Göreve uygun kayıp fonksiyonu seçer.
    """
    if cfg.TASK == "segmentation":
        # Dice + Cross-Entropy karışımı: medikal segmentasyon standardı
        criterion = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,  # Arka plan sınıfını hariç tut
            lambda_dice=0.6,
            lambda_ce=0.4,
        )
    else:
        # Sınıf dengesizliği yaygın olduğu için Focal Loss
        criterion = FocalLoss(
            gamma=2.0,
            weight=None,  # İsteğe bağlı: torch.tensor([w0, w1, ...])
            reduction="mean",
        )
    return criterion


class MetricTracker:
    """
    Eğitim ve validasyon metriklerini epoch bazında izler ve kaydeder.
    """

    def __init__(self, task: str) -> None:
        self.task    = task
        self.history = {"train_loss": [], "val_loss": [], "val_metric": []}

        if task == "segmentation":
            # Her sınıf için ayrı Dice puanı
            self.dice_metric = DiceMetric(
                include_background=False,
                reduction="mean_batch",
                get_not_nans=False,
            )
        else:
            self.auc_metric = ROCAUCMetric()

    def update_segmentation(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """Tahminleri aktivasyon + argmax ile ikili maskeye dönüştürür."""
        preds_softmax = torch.softmax(preds, dim=1)
        preds_onehot  = torch.argmax(preds_softmax, dim=1, keepdim=True)
        labels_onehot = labels.long()
        self.dice_metric(y_pred=preds_onehot, y=labels_onehot)

    def compute(self) -> float:
        if self.task == "segmentation":
            score = self.dice_metric.aggregate().item()
            self.dice_metric.reset()
        else:
            score = self.auc_metric.aggregate()
            self.auc_metric.reset()
        return score

    def log_epoch(self, train_loss: float, val_loss: float, val_metric: float) -> None:
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_metric"].append(val_metric)

    def save_history(self, path: Path) -> None:
        with open(path / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)


# =============================================================================
# 6. EĞİTİM & VALİDASYON DÖNGÜSÜ
# =============================================================================

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler:    GradScaler,
    cfg:       Config,
    epoch:     int,
) -> float:
    """
    Tek bir eğitim epoch'u çalıştırır.

    Bellek Optimizasyonu:
    - AMP (FP16) ile GPU belleği ~%40 azaltılır.
    - Her batch sonrası gradient temizlenir.
    - gc.collect() ile Python çöp toplayıcısı tetiklenir.

    Returns:
        Ortalama epoch kaybı (float)
    """
    model.train()
    running_loss = 0.0
    n_batches    = len(loader)

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(cfg.DEVICE, non_blocking=True)
        labels = batch["label"].to(cfg.DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # None atama bellek verimliliği sağlar

        # ── Otomatik Karma Hassasiyet (AMP) ───────────────────────────────────
        with autocast(enabled=cfg.USE_AMP):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        # ── Gradient Ölçekleme (underflow önleme) ─────────────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient patlama önlemi
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # ── İlerleme Logu ─────────────────────────────────────────────────────
        if (batch_idx + 1) % cfg.LOG_INTERVAL == 0:
            logger.info(
                f"Epoch [{epoch+1:03d}] "
                f"Batch [{batch_idx+1:04d}/{n_batches}] "
                f"Kayıp: {loss.item():.4f} "
                f"VRAM: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
            )

        # ── GPU Bellek Temizleme ───────────────────────────────────────────────
        del images, labels, outputs, loss
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    return running_loss / n_batches


@torch.no_grad()
def validate(
    model:   nn.Module,
    loader:  DataLoader,
    criterion: nn.Module,
    tracker: MetricTracker,
    cfg:     Config,
) -> Tuple[float, float]:
    """
    Validasyon döngüsü — gradient hesaplaması yapılmaz (bellek tasarrufu).

    Segmentasyon görevi için sliding window inference kullanılır:
    büyük görüntüler tile'lara bölünerek tahmin edilir.

    Returns:
        (val_loss, val_metric) çifti
    """
    model.eval()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(cfg.DEVICE, non_blocking=True)
        labels = batch["label"].to(cfg.DEVICE, non_blocking=True)

        with autocast(enabled=cfg.USE_AMP):
            if cfg.TASK == "segmentation":
                # Büyük görüntülerde kayan pencere tahmini (OOM önleyici)
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=cfg.IMAGE_SIZE,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.25,
                )
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)

        running_loss += loss.item()

        # Metrik güncelleme
        if cfg.TASK == "segmentation":
            tracker.update_segmentation(outputs.cpu(), labels.cpu())

        del images, labels, outputs, loss
        torch.cuda.empty_cache()

    val_loss   = running_loss / len(loader)
    val_metric = tracker.compute()
    return val_loss, val_metric


# =============================================================================
# 7. GÖRSELLEŞTİRME
# =============================================================================

def plot_training_history(history: Dict[str, List], save_dir: Path) -> None:
    """
    Eğitim kaybı ve validasyon metrik eğrilerini çizer ve kaydeder.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Eğitim Geçmişi", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Eğitim Kaybı", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   "r-s", label="Validasyon Kaybı", linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Kayıp")
    axes[0].set_title("Kayıp Eğrileri"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_metric"], "g-^", linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Metrik (Dice / AUC)")
    axes[1].set_title("Validasyon Metriği"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Eğitim grafiği kaydedildi: {save_dir / 'training_history.png'}")


def visualize_predictions(
    model:  nn.Module,
    loader: DataLoader,
    cfg:    Config,
    n_samples: int = 4,
) -> None:
    """
    Rastgele örnekler için girdi / tahmin / gerçek maske görselleştirmesi.
    """
    model.eval()
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
    cols = ["Görüntü", "Tahmin", "Gerçek Maske"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontweight="bold")

    batch = next(iter(loader))
    images = batch["image"][:n_samples].to(cfg.DEVICE)
    labels = batch["label"][:n_samples]

    with torch.no_grad(), autocast(enabled=cfg.USE_AMP):
        preds = model(images)
        preds = torch.softmax(preds, dim=1).argmax(dim=1).cpu().numpy()

    images = images.cpu().numpy()
    labels = labels.numpy()

    for i in range(min(n_samples, len(images))):
        img = images[i, 0]   # İlk kanal
        axes[i, 0].imshow(img, cmap="gray"); axes[i, 0].axis("off")
        axes[i, 1].imshow(preds[i], cmap="jet", vmin=0, vmax=CFG.NUM_CLASSES - 1); axes[i, 1].axis("off")
        axes[i, 2].imshow(labels[i, 0] if labels.ndim == 4 else labels[i], cmap="jet"); axes[i, 2].axis("off")

    plt.tight_layout()
    fig.savefig(cfg.OUTPUT_DIR / "sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Örnek tahminler kaydedildi.")


# =============================================================================
# 8. CHECKPOINT & MODEL KAYDETME / YÜKLEME
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    cfg: Config,
) -> None:
    """Model ağırlıklarını ve meta bilgilerini kaydeder."""
    state = {
        "epoch":      epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "metric":     metric,
        "task":       cfg.TASK,
        "image_size": cfg.IMAGE_SIZE,
        "num_classes": cfg.NUM_CLASSES,
        "timestamp":  datetime.now().isoformat(),
    }
    torch.save(state, cfg.CHECKPOINT_PATH)
    logger.info(f"✔ Checkpoint kaydedildi → {cfg.CHECKPOINT_PATH} (Metrik: {metric:.4f})")


def load_checkpoint(model: nn.Module, cfg: Config) -> Tuple[nn.Module, int, float]:
    """
    Kaydedilmiş checkpoint'ten modeli yükler.
    Inference veya eğitimi kaldığı yerden devam ettirmek için kullanın.
    """
    if not cfg.CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {cfg.CHECKPOINT_PATH}")

    state = torch.load(cfg.CHECKPOINT_PATH, map_location=cfg.DEVICE)
    model.load_state_dict(state["model_state"])
    logger.info(f"Checkpoint yüklendi: Epoch {state['epoch']} | Metrik: {state['metric']:.4f}")
    return model, state["epoch"], state["metric"]


# =============================================================================
# 9. ANA FONKSİYON
# =============================================================================

def main():
    """
    Pipeline akışı:
      1. Dosya listesi oluştur (CSV/klasör taraması)
      2. Dönüşüm zincirlerini kur
      3. DataLoader'ları oluştur (tembel yükleme)
      4. Model, kayıp, optimizer, scheduler kur
      5. Eğitim döngüsünü çalıştır
      6. Sonuçları görselleştir ve kaydet
    """
    logger.info("=" * 60)
    logger.info("Pipeline başlıyor...")
    logger.info(f"Çıktı dizini: {CFG.OUTPUT_DIR}")
    logger.info("=" * 60)

    # ── 1. Veri Listesi ───────────────────────────────────────────────────────
    train_files, val_files, test_files = build_file_list(CFG)

    # ── 2. Dönüşümler ─────────────────────────────────────────────────────────
    train_transforms = get_transforms(CFG, mode="train")
    val_transforms   = get_transforms(CFG, mode="val")
    test_transforms  = get_transforms(CFG, mode="test")

    # ── 3. Dataset & DataLoader ───────────────────────────────────────────────
    # CACHE_RATE=0.0 → hiç veri belleğe yüklenmiyor (40 GB güvenliği)
    # Hızlı iterasyon istiyorsanız CACHE_RATE=0.1 ile %10 RAM'de tutabilirsiniz
    train_ds = ChestDataset(train_files, train_transforms, cfg.TASK if False else CFG.TASK)
    val_ds   = ChestDataset(val_files,   val_transforms,   CFG.TASK)
    test_ds  = ChestDataset(test_files,  test_transforms,  CFG.TASK)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
        prefetch_factor=CFG.PREFETCH_FACTOR if CFG.NUM_WORKERS > 0 else None,
        persistent_workers=CFG.NUM_WORKERS > 0,   # Worker süreçlerini canlı tut
        drop_last=True,                            # Küçük son batch'i at (BN kararlılığı)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,                              # Validasyonda batch=1 daha güvenli
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
    )

    # ── 4. Model & Eğitim Bileşenleri ─────────────────────────────────────────
    model     = build_model(CFG)
    criterion = build_criterion(CFG)
    tracker   = MetricTracker(CFG.TASK)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.LEARNING_RATE,
        weight_decay=CFG.WEIGHT_DECAY,
    )
    # Kosinüs tavlama + warm restart: plateau'dan çıkmaya yardımcı olur
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,      # İlk restart döngüsü (epoch)
        T_mult=2,    # Her restart'ta döngü 2x uzar
        eta_min=1e-7,
    )
    # FP16 gradient ölçekleyici
    scaler = GradScaler(enabled=CFG.USE_AMP)

    # ── 5. Eğitim Döngüsü ────────────────────────────────────────────────────
    best_metric = -float("inf")
    start_time  = time.time()

    logger.info(f"Eğitim başlıyor: {CFG.EPOCHS} epoch × {len(train_loader)} batch")

    for epoch in range(CFG.EPOCHS):
        epoch_start = time.time()

        # Eğitim
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, CFG, epoch
        )

        # Validasyon
        val_loss, val_metric = validate(model, val_loader, criterion, tracker, CFG)

        # Öğrenme hızı güncelleme
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Metrikleri kaydet
        tracker.log_epoch(train_loss, val_loss, val_metric)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch [{epoch+1:03d}/{CFG.EPOCHS}] "
            f"Eğitim Kaybı: {train_loss:.4f} | "
            f"Val Kaybı: {val_loss:.4f} | "
            f"Val Metrik: {val_metric:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Süre: {epoch_time:.1f}s"
        )

        # En iyi modeli kaydet
        if val_metric > best_metric:
            best_metric = val_metric
            save_checkpoint(model, optimizer, epoch, val_metric, CFG)
            logger.info(f"★ Yeni en iyi model! Metrik: {best_metric:.4f}")

    total_time = time.time() - start_time
    logger.info(f"Eğitim tamamlandı. Toplam süre: {total_time / 60:.1f} dakika")
    logger.info(f"En iyi validasyon metriği: {best_metric:.4f}")

    # ── 6. Test & Görselleştirme ──────────────────────────────────────────────
    # En iyi modeli yükle
    model, _, _ = load_checkpoint(model, CFG)

    # Test seti değerlendirmesi
    test_loss, test_metric = validate(model, test_loader, criterion, tracker, CFG)
    logger.info(f"TEST — Kayıp: {test_loss:.4f} | Metrik: {test_metric:.4f}")

    # Grafikleri kaydet
    tracker.save_history(CFG.OUTPUT_DIR)
    plot_training_history(tracker.history, CFG.OUTPUT_DIR)
    visualize_predictions(model, test_loader, CFG)

    logger.info(f"Tüm çıktılar kaydedildi → {CFG.OUTPUT_DIR}")
    return model, tracker


# =============================================================================
# BAŞLANGIÇ NOKTASI
# =============================================================================

if __name__ == "__main__":
    # Kaggle Notebook'ta doğrudan çalıştırın:
    # > python chest_analysis_pipeline.py
    # veya notebook hücresinde:
    # > exec(open("chest_analysis_pipeline.py").read())
    trained_model, metrics = main()
