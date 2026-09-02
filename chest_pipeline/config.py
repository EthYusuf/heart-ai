# =============================================================================
# config.py — Merkezi Yapılandırma Modülü
# Tüm hiperparametreler, yollar ve deneysel ayarlar tek noktada yönetilir.
# Farklı deneyler için Config alt sınıfları oluşturun.
# =============================================================================

import os
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from monai.utils import set_determinism


# ─────────────────────────────────────────────────────────────────────────────
# Temel Yapılandırma
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """
    Deneysel yapılandırma. Kaggle Notebook'ta en üstte tanımlayıp
    tüm modüllere tek nesne olarak geçirin.
    """

    # ── Deney Kimliği ─────────────────────────────────────────────────────────
    EXP_NAME:   str  = "chest_unet_baseline"
    EXP_VERSION: int = 1
    DESCRIPTION: str = "MONAI U-Net ile göğüs X-ray segmentasyonu — baseline"

    # ── Görev ─────────────────────────────────────────────────────────────────
    # "segmentation"  → U-Net / U-Net++ / Attention U-Net
    # "classification" → DenseNet121 / EfficientNet / Swin-T
    TASK: str = "segmentation"

    # Çok-etiketli sınıflandırma: bir görüntüde eş zamanlı birden çok bulgu
    # olabilir (ör. NIH ChestX-ray14). Sigmoid + BCE kullanılır, softmax değil.
    MULTI_LABEL: bool = False
    CLASS_NAMES: List[str] = field(default_factory=list)

    # ── Veri Yolları (Kaggle) ─────────────────────────────────────────────────
    DATA_ROOT:  Path = Path("/kaggle/input/<DATASET-ADI>")
    IMAGE_DIR:  Path = DATA_ROOT / "images"
    MASK_DIR:   Path = DATA_ROOT / "masks"          # Segmentasyon maskesi klasörü
    CSV_PATH:   Path = DATA_ROOT / "metadata.csv"   # Sütunlar: image_path, mask_path, label
    OUTPUT_DIR: Path = Path("/kaggle/working/outputs")
    CACHE_DIR:  Path = Path("/kaggle/working/cache") # PersistentDataset önbelleği

    # ── Görüntü Özellikleri ───────────────────────────────────────────────────
    IMAGE_SIZE:   Tuple[int, int] = (512, 512)
    IN_CHANNELS:  int = 1           # Gri tonlamalı CT/X-ray=1, RGB=3
    NUM_CLASSES:  int = 2           # Arka plan dahil sınıf sayısı
    IMAGE_FORMAT: str = "png"       # "png" | "dcm" | "nii" | "jpg"

    # Yoğunluk penceresi: CT (HU) → -1000/400 | X-ray (8-bit PNG) → 0/255
    INTENSITY_MIN: float = 0.0
    INTENSITY_MAX: float = 255.0

    # ── Model Seçimi ──────────────────────────────────────────────────────────
    # "unet" | "unet_pp" | "attention_unet" | "densenet121" | "efficientnet" | "swin_unet"
    MODEL_NAME: str = "unet"
    PRETRAINED: bool = True         # Transfer öğrenme (sınıflandırma için)

    # ── Kayıp Fonksiyonu ──────────────────────────────────────────────────────
    # "dice_ce" | "focal" | "tversky" | "combo" | "boundary" | "lovasz"
    LOSS_NAME: str = "dice_ce"
    LOSS_ALPHA: float = 0.6         # Combo loss: Dice ağırlığı
    TVERSKY_ALPHA: float = 0.3      # Tversky FP ağırlığı
    TVERSKY_BETA:  float = 0.7      # Tversky FN ağırlığı

    # ── Eğitim Hiperparametreleri ─────────────────────────────────────────────
    EPOCHS:         int   = 100
    BATCH_SIZE:     int   = 4
    LEARNING_RATE:  float = 1e-4
    WEIGHT_DECAY:   float = 1e-5
    WARMUP_EPOCHS:  int   = 5       # Lineer LR ısınma süresi
    MIN_LR:         float = 1e-7
    GRAD_CLIP:      float = 1.0     # Gradient norm clipping

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # "adamw" | "sam" | "lion" | "sgd"
    OPTIMIZER:    str   = "adamw"
    MOMENTUM:     float = 0.9       # SGD için
    SAM_RHO:      float = 0.05      # SAM perturbation radius

    # ── Scheduler ─────────────────────────────────────────────────────────────
    # "cosine_warm" | "one_cycle" | "poly" | "step" | "plateau"
    SCHEDULER:     str   = "cosine_warm"
    T0:            int   = 10       # CosineWarm: ilk restart epoch
    T_MULT:        int   = 2        # CosineWarm: restart büyüme faktörü

    # ── Doğrulama Stratejisi ──────────────────────────────────────────────────
    KFOLD:         int   = 5        # K-Fold CV katman sayısı (1=standart bölme)
    VAL_RATIO:     float = 0.15
    TEST_RATIO:    float = 0.10
    STRATIFIED:    bool  = True     # Sınıflara göre dengeli bölme

    # ── Test Time Augmentation ────────────────────────────────────────────────
    USE_TTA:        bool = True
    TTA_STEPS:      int  = 8        # TTA örnekleme sayısı

    # ── Artırma Yoğunluğu ────────────────────────────────────────────────────
    AUG_PROB:      float = 0.5      # Genel augmentation olasılığı
    ELASTIC_PROB:  float = 0.3
    MIXUP_PROB:    float = 0.2      # MixUp (sınıflandırma için)
    MIXUP_ALPHA:   float = 0.4

    # ── Bellek Optimizasyonu ──────────────────────────────────────────────────
    USE_AMP:           bool  = True
    NUM_WORKERS:       int   = 2
    PIN_MEMORY:        bool  = True
    PREFETCH_FACTOR:   int   = 2
    CACHE_RATE:        float = 0.0  # 0.0 = tam tembel yükleme (40 GB güvenli)
    SLIDING_OVERLAP:   float = 0.25 # Sliding window örtüşme oranı

    # ── Keşifsel Veri Analizi (EDA) ───────────────────────────────────────────
    # DÜZELTME: Önceden main.py içinde `RUN_EDA = False` olarak sabit kodlanmıştı
    # ve yalnızca kaynak kodu düzenleyerek değiştirilebiliyordu. Artık diğer
    # tüm ayarlar gibi Config üzerinden yönetiliyor (cfg.RUN_EDA = False ile
    # kapatılabilir). Görüntü kalitesi denetimi / bozuk dosya tespiti bu
    # adımda üretilir — büyük veri setlerinde 10-30 dakika sürebilir.
    RUN_EDA:        bool = True
    EDA_SAMPLE_CAP: int  = 2000   # İstatistik için örneklenecek maksimum görüntü

    # ── Kayıt ve İzleme ───────────────────────────────────────────────────────
    LOG_INTERVAL:   int  = 10
    SAVE_BEST_ONLY: bool = True
    USE_WANDB:      bool = False    # Weights & Biases entegrasyonu
    WANDB_PROJECT:  str  = "chest-xray-analysis"

    # ── Çıkarım / Export ─────────────────────────────────────────────────────
    EXPORT_ONNX:    bool = True
    ONNX_OPSET:     int  = 17
    EXPORT_TORCHSCRIPT: bool = False

    # ── Erken Durdurma ────────────────────────────────────────────────────────
    EARLY_STOPPING:         bool = True
    EARLY_STOPPING_PATIENCE: int = 15
    EARLY_STOPPING_MIN_DELTA: float = 1e-4

    # ── Yeniden Üretilebilirlik ───────────────────────────────────────────────
    SEED: int = 42

    # ── Çalışma Zamanında Türetilecek Alanlar ────────────────────────────────
    DEVICE:          torch.device = field(init=False)
    CHECKPOINT_PATH: Path         = field(init=False)
    LOG_PATH:        Path         = field(init=False)
    RUN_ID:          str          = field(init=False)

    def __post_init__(self):
        self.DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.RUN_ID    = f"{self.EXP_NAME}_v{self.EXP_VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir        = self.OUTPUT_DIR / self.RUN_ID
        run_dir.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_PATH = run_dir / "best_model.pth"
        self.LOG_PATH        = run_dir / "run.log"
        set_determinism(seed=self.SEED)
        self._log_env()

    def _log_env(self):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Deney: {self.RUN_ID}")
        logger.info(f"Cihaz: {self.DEVICE}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU  : {props.name} — {props.total_memory/1e9:.1f} GB VRAM")
        logger.info(f"PyTorch: {torch.__version__} | MONAI: {__import__('monai').__version__}")

    def save(self, path: Optional[Path] = None) -> None:
        """Yapılandırmayı JSON'a kaydeder (deney izlenebilirliği)."""
        path = path or self.CHECKPOINT_PATH.parent / "config.json"
        d = {k: str(v) if isinstance(v, (Path, torch.device)) else v
             for k, v in asdict(self).items()}
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """JSON'dan yapılandırmayı geri yükler."""
        with open(path) as f:
            d = json.load(f)
        # Path ve device alanlarını dönüştür
        for k in ["DATA_ROOT", "IMAGE_DIR", "MASK_DIR", "CSV_PATH",
                  "OUTPUT_DIR", "CACHE_DIR", "CHECKPOINT_PATH", "LOG_PATH"]:
            if k in d:
                d[k] = Path(d[k])
        d.pop("DEVICE", None)
        d.pop("CHECKPOINT_PATH", None)
        d.pop("LOG_PATH", None)
        d.pop("RUN_ID", None)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────────────────────
# Hazır Deney Profilleri
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuickDebugConfig(Config):
    """
    Sistem entegrasyon ve birim testi için minimal ölçekli yapılandırma.
    Pipeline bütünlüğünü doğrulamak amacıyla küçük epoch ve görüntü boyutu kullanır;
    bilimsel sonuç üretimi için kullanılmamalıdır.

    DÜZELTME: Alt sınıf @dataclass ile dekorlanmalı; aksi halde buradaki
    alanlar saha (field) olarak işlenmez ve taban Config'in __init__'i
    tüm profili taban değerleriyle EZİYOR (ör. EPOCHS=3 sessizce 100 oluyor).
    """
    EPOCHS:     int            = 3
    BATCH_SIZE: int            = 2
    IMAGE_SIZE: Tuple[int,int] = (256, 256)
    KFOLD:      int            = 1
    USE_TTA:    bool           = False
    CACHE_RATE: float          = 0.0
    EXP_NAME:   str            = "debug"
    RUN_EDA:    bool           = False   # Hızlı entegrasyon testinde EDA atlanır


@dataclass
class HighPerformanceConfig(Config):
    """
    Yüksek doğruluk hedefli tam ölçekli deneysel yapılandırma.
    Akademik yayın ve kapsamlı klinik değerlendirme çalışmaları için tasarlanmıştır.
    K-Fold CV, TTA ve gelişmiş optimizasyon teknikleri etkinleştirilmiştir.
    """
    EPOCHS:       int   = 200
    BATCH_SIZE:   int   = 8
    IMAGE_SIZE:   Tuple = (768, 768)
    MODEL_NAME:   str   = "attention_unet"
    LOSS_NAME:    str   = "combo"
    OPTIMIZER:    str   = "sam"
    KFOLD:        int   = 5
    USE_TTA:      bool  = True
    TTA_STEPS:    int   = 16
    EXPORT_ONNX:  bool  = True
    EXP_NAME:     str   = "high_perf"


@dataclass
class LocalDebugConfig(Config):
    """
    Yerel Windows makinesi (CPU, GPU'suz) için uçtan uca duman testi profili.

    prepare_local_testdata.py ile üretilen sentetik küçük veri setiyle
    (data_local/) pipeline'ın TÜM aşamalarını çalıştırır:
    veri yükleme → EDA → eğitim → değerlendirme → ONNX export.
    Bilimsel sonuç üretimi için KULLANILMAMALIDIR — gerçek eğitim
    Kaggle GPU üzerinde Config/HighPerformanceConfig ile yapılır.

    Not: IMAGE_DIR/MASK_DIR/CSV_PATH/OUTPUT_DIR/CACHE_DIR alanları taban
    sınıfta DATA_ROOT'tan türetilir; dataclass kalıtımında bu türetme
    yeniden hesaplanmadığı için burada açıkça set edilmelidir.
    """
    _LOCAL = Path(__file__).resolve().parent / "data_local"

    # ── Yerel veri yolları ──
    DATA_ROOT:  Path = _LOCAL
    IMAGE_DIR:  Path = _LOCAL / "images"
    MASK_DIR:   Path = _LOCAL / "masks"
    CSV_PATH:   Path = _LOCAL / "metadata.csv"   # dosya yoksa klasör taraması devreye girer
    OUTPUT_DIR: Path = Path(__file__).resolve().parent / "outputs_local"
    CACHE_DIR:  Path = Path(__file__).resolve().parent / "outputs_local" / "cache"

    # ── CPU'ya göre küçültülmüş ölçek ──
    EPOCHS:     int            = 2
    BATCH_SIZE: int            = 2
    IMAGE_SIZE: Tuple[int,int] = (256, 256)
    KFOLD:      int            = 1
    USE_TTA:    bool           = False
    CACHE_RATE: float          = 0.0
    USE_AMP:    bool           = False   # CPU'da hız kazancı yok, hata riski gereksiz
    NUM_WORKERS: int          = 2       # EDA thread havuzu > 0 ister; DataLoader bunu paylaşır
    PIN_MEMORY: bool          = False   # GPU yoksa anlamsız
    RUN_EDA:    bool           = True
    EDA_SAMPLE_CAP: int       = 40
    EARLY_STOPPING: bool      = False
    EXP_NAME:   str            = "local_debug"
