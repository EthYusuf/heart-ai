# =============================================================================
# datasets.py — Bellek Güvenli Veri Seti & Dönüşüm Modülü
# -----------------------------------------------------------------------------
# 40 GB veri için sıfır kopya, tembel yükleme stratejisi.
# DICOM / NIfTI / PNG / JPEG formatlarını tek API'den yönetir.
# Albumentations entegrasyonu ile zengin artırma desteği.
# =============================================================================

import gc
import logging
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
# DÜZELTME: train_test_split hiçbir yerde kullanılmıyordu (ölü import) —
# build_file_list kendi np.random.permutation tabanlı bölmesini veya CSV'deki
# "split" sütununu kullanıyor. Bu import, Kaggle'da scipy/numpy sürüm
# çakışması olduğunda sklearn'ün tamamını (gereksiz yere) tetikleyip
# datasets.py'nin içe aktarılmasını engelliyordu. Kaldırıldı.

import monai
from monai.data import PersistentDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, NormalizeIntensityd, Resized,
    RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised,
    RandAdjustContrastd, RandCoarseDropoutd, RandGaussianSmoothd,
    RandHistogramShiftd, RandZoomd, RandGridDistortiond,
    ToTensord, CenterSpatialCropd, DivisiblePadd,
    CastToTyped, ThresholdIntensityd, Lambdad,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =============================================================================
# 1. Veri Listesi Oluşturma
# =============================================================================

def build_file_list(
    cfg:       Any,
    fold_idx:  Optional[int] = None,
    kfold_df:  Optional[pd.DataFrame] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    CSV veya klasör taramasından (image, label/mask) çiftleri oluşturur.
    fold_idx verilirse K-Fold bölme kullanılır.

    Returns:
        (train_files, val_files, test_files)
    """
    EXTENSIONS = {".dcm", ".png", ".jpg", ".jpeg", ".nii", ".gz", ".tif", ".tiff"}

    # CSV'de "split" sütunu varsa rastgele bölme yerine o kullanılır.
    # Hasta bazlı / resmî bölmeleri korumak için gereklidir: aynı hastanın
    # farklı çekimleri eğitim ve teste dağılırsa model hastayı ezberler ve
    # test skoru yapay olarak yükselir.
    split_labels: Optional[np.ndarray] = None

    # ── Seçenek A: CSV'den yükle ──────────────────────────────────────────────
    if cfg.CSV_PATH.exists():
        df = pd.read_csv(cfg.CSV_PATH)
        logger.info(f"CSV yüklendi: {cfg.CSV_PATH} — {len(df)} satır")
        logger.info(f"Sütunlar: {list(df.columns)}")

        if cfg.TASK == "segmentation":
            # Beklenen sütunlar: image_path, mask_path
            # Yoksa aşağıdaki sütun adlarını güncelleyin:
            img_col  = next((c for c in ["image_path", "image", "img_path"] if c in df.columns), None)
            msk_col  = next((c for c in ["mask_path", "mask", "seg_path"]   if c in df.columns), None)
            if img_col is None or msk_col is None:
                raise ValueError(f"CSV'de görüntü/maske sütunu bulunamadı. Mevcut: {list(df.columns)}")
            file_list = [
                {"image": str(cfg.IMAGE_DIR / row[img_col]),
                 "label": str(cfg.MASK_DIR  / row[msk_col])}
                for _, row in df.iterrows()
            ]
        else:
            img_col = next((c for c in ["image_path", "image", "img_path"] if c in df.columns), None)
            if img_col is None:
                raise ValueError(f"CSV'de görüntü sütunu bulunamadı. Mevcut: {list(df.columns)}")

            if "split" in df.columns:
                split_labels = df["split"].to_numpy()

            if getattr(cfg, "MULTI_LABEL", False):
                # Çok-etiketli: image dışındaki her sütun bir sınıfın 0/1 göstergesi.
                # Hedef (C,) boyutlu float vektör — BCEWithLogitsLoss bunu bekler.
                lbl_cols = [c for c in df.columns if c not in (img_col, "split")]
                if len(lbl_cols) != cfg.NUM_CLASSES:
                    raise ValueError(
                        f"MULTI_LABEL için {cfg.NUM_CLASSES} etiket sütunu bekleniyor, "
                        f"{len(lbl_cols)} bulundu: {lbl_cols}"
                    )
                targets = df[lbl_cols].to_numpy(dtype="float32")
                file_list = [
                    {"image": str(cfg.IMAGE_DIR / img), "label": torch.from_numpy(t)}
                    for img, t in zip(df[img_col].to_numpy(), targets)
                ]
            else:
                lbl_col = next((c for c in ["label", "class", "target"] if c in df.columns), None)
                if lbl_col is None:
                    raise ValueError(f"CSV'de etiket sütunu bulunamadı. Mevcut: {list(df.columns)}")
                file_list = [
                    {"image": str(cfg.IMAGE_DIR / row[img_col]),
                     "label": torch.tensor(int(row[lbl_col]), dtype=torch.long)}
                    for _, row in df.iterrows()
                ]

    # ── Seçenek B: Klasör taraması ────────────────────────────────────────────
    else:
        logger.warning("CSV bulunamadı — klasör yapısı taranıyor.")
        if cfg.TASK == "segmentation":
            image_paths = sorted(
                p for p in cfg.IMAGE_DIR.rglob("*") if p.suffix.lower() in EXTENSIONS
            )
            file_list = [
                {"image": str(p),
                 "label": str(cfg.MASK_DIR / p.relative_to(cfg.IMAGE_DIR))}
                for p in image_paths
                if (cfg.MASK_DIR / p.relative_to(cfg.IMAGE_DIR)).exists()
            ]
        else:
            # Klasör adı = sınıf: /images/normal/img.png → label=0
            class_dirs = sorted(d for d in cfg.IMAGE_DIR.iterdir() if d.is_dir())
            class_map  = {d.name: i for i, d in enumerate(class_dirs)}
            file_list  = [
                {"image": str(p), "label": class_map[p.parent.name]}
                for d in class_dirs
                for p in d.rglob("*")
                if p.suffix.lower() in EXTENSIONS
            ]
            logger.info(f"Sınıf haritası: {class_map}")

    if not file_list:
        raise RuntimeError("Veri listesi boş! Dosya yollarını kontrol edin.")

    logger.info(f"Toplam örnek: {len(file_list)}")

    # ── Önceden tanımlı bölme (hasta bazlı / resmî) ──────────────────────────
    if split_labels is not None:
        train_files = [f for f, s in zip(file_list, split_labels) if s == "train"]
        val_files   = [f for f, s in zip(file_list, split_labels) if s == "val"]
        test_files  = [f for f, s in zip(file_list, split_labels) if s == "test"]
        if not (train_files and val_files and test_files):
            raise ValueError(
                "'split' sütunu train/val/test değerlerinin üçünü de içermeli. "
                f"Bulunan: {sorted(set(split_labels))}"
            )
        logger.info(
            f"Önceden tanımlı bölme kullanıldı — "
            f"Eğitim: {len(train_files)} | Validasyon: {len(val_files)} | Test: {len(test_files)}"
        )
        return train_files, val_files, test_files

    # ── Karıştır & Böl ────────────────────────────────────────────────────────
    np.random.seed(cfg.SEED)
    indices = np.random.permutation(len(file_list))
    n       = len(indices)
    n_test  = max(1, int(n * cfg.TEST_RATIO))
    n_val   = max(1, int(n * cfg.VAL_RATIO))

    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    train_files = [file_list[i] for i in train_idx]
    val_files   = [file_list[i] for i in val_idx]
    test_files  = [file_list[i] for i in test_idx]

    logger.info(f"Eğitim: {len(train_files)} | Validasyon: {len(val_files)} | Test: {len(test_files)}")
    return train_files, val_files, test_files


# =============================================================================
# 2. MONAI Dönüşüm Zincirleri
# =============================================================================

def fit_channels(x, n_ch: int = 1):
    """
    Kanal sayısını n_ch'e sabitler.
    Veri setinde gri tonlamalı (1), RGB (3) ve RGBA (4) PNG'ler karışık
    bulunabilir; batch'lenebilmesi için tek biçime indirgenir.

    DÜZELTME: Önceden get_transforms içinde İÇ İÇE (closure) tanımlıydı.
    Windows'ta DataLoader worker'ları spawn yöntemiyle başlatılır ve
    dataset'i pickle eder; iç içe fonksiyonlar pickle EDİLEMEZ, bu yüzden
    num_workers>0 iken PicklingError ile pipeline çöküyordu
    (Kaggle/Linux fork'ta pickle gerekmediğinden orada görünmüyordu).
    Modül seviyesine taşındı; n_ch functools.partial ile bağlanır —
    partial + modül fonksiyonu pickle edilebilir.
    """
    c = x.shape[0]
    if c > n_ch:
        return x[:n_ch]
    if c < n_ch:
        # DÜZELTME: `n_ch // c` tam sayı bölmesi n_ch, c'ye tam
        # bölünmediğinde (ör. c=2 → n_ch=3) yetersiz kanal üretip
        # sonraki batch stack aşamasında boyut hatasına yol açıyordu.
        # İlk kanalı tekrarlayarak istenen kanal sayısına tamamlanır.
        return x[:1].repeat(n_ch, 1, 1) if c == 1 else \
               x.repeat((n_ch + c - 1) // c, 1, 1)[:n_ch]
    return x


def binarize_mask(x, thresh: float = 127.5):
    """
    Ham 0/255 ikili maskeleri 0/1'e indirger.

    DÜZELTME: Maske hiç binarize edilmiyordu ve geometrik dönüşümlerin
    interpolasyonu 0-255 arasında ara değerler (ör. 139) üretiyordu;
    DiceCELoss'un one_hot adımı "index 139 is out of bounds" hatasıyla
    çöküyordu. Maske yükleme sonrası hemen 0/1'e indirgenir; resize/zoom
    için maskede 'nearest' interpolasyon kullanılır. Modül seviyesinde
    tanımlıdır çünkü Windows spawn worker'ları pickle edebilirlik ister.
    """
    return (x > thresh).to(x.dtype)


def get_transforms(cfg: Any, mode: str = "train") -> Compose:
    """
    Mode'a göre uygun MONAI dönüşüm zinciri oluşturur.

    mode: "train" | "val" | "test"

    CT penceresi:  a_min=-1000, a_max=400   (akciğer penceresi)
    X-ray:         a_min=0,     a_max=255
    """
    h, w = cfg.IMAGE_SIZE

    # ── Ortak Ön İşleme ───────────────────────────────────────────────────────
    load_keys = ["image", "label"] if cfg.TASK == "segmentation" else ["image"]

    n_ch = cfg.IN_CHANNELS

    base = [
        LoadImaged(
            keys=load_keys,
            ensure_channel_first=True,
            image_only=True,   # meta dict döndürme — collate hatası önlenir
        ),
        Lambdad(keys=["image"], func=partial(fit_channels, n_ch=n_ch)),
        # CT için HU penceresi, X-ray için ham değer aralığı
        ScaleIntensityRanged(
            keys=["image"],
            a_min=cfg.INTENSITY_MIN, a_max=cfg.INTENSITY_MAX,
            b_min=0.0,   b_max=1.0,
            clip=True,
        ),
        ThresholdIntensityd(keys=["image"], threshold=0.0, above=True,  cval=0.0),
        ThresholdIntensityd(keys=["image"], threshold=1.0, above=False, cval=1.0),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        # DÜZELTME: maske nearest ile resize edilir — bilinear 0/1 maskede
        # ara değerler üretip one_hot'ı bozuyordu
        Resized(
            keys=load_keys, spatial_size=(h, w),
            mode=("bilinear", "nearest") if cfg.TASK == "segmentation" else ("bilinear",),
        ),
        DivisiblePadd(keys=load_keys, k=32),  # U-Net encoder için 32'nin katı
    ]

    if cfg.TASK == "segmentation":
        # Maske yükleme sonrası hemen 0/1'e indirgenir (bkz. binarize_mask)
        base.insert(2, Lambdad(keys=["label"], func=partial(binarize_mask, thresh=127.5)))

    if mode == "train":
        # ── Geometrik Artırma ─────────────────────────────────────────────────
        aug = [
            RandFlipd(keys=load_keys, prob=cfg.AUG_PROB, spatial_axis=0),
            RandFlipd(keys=load_keys, prob=cfg.AUG_PROB, spatial_axis=1),
            RandRotate90d(keys=load_keys, prob=0.3, max_k=3),
            RandAffined(
                keys=load_keys,
                prob=cfg.AUG_PROB,
                rotate_range=(0.2, 0.2),
                scale_range=(0.15, 0.15),
                translate_range=(25, 25),
                shear_range=(0.05, 0.05),
                mode=("bilinear", "nearest") if cfg.TASK == "segmentation" else ("bilinear",),
                padding_mode="zeros",
            ),
            RandZoomd(
                keys=load_keys, prob=0.3, min_zoom=0.85, max_zoom=1.15,
                # DÜZELTME: RandZoomd varsayılanı 'area' interpolasyonu —
                # maskeyi bozar; maskede nearest kullanılır
                mode=("bilinear", "nearest") if cfg.TASK == "segmentation" else ("bilinear",),
            ),
            # ── Grid Distortion — Elastik deformasyon ─────────────────────────
            RandGridDistortiond(
                keys=load_keys, prob=cfg.ELASTIC_PROB, distort_limit=0.2,
                mode=("bilinear", "nearest") if cfg.TASK == "segmentation" else ("bilinear",),
            ),
            # ── Yoğunluk Artırma ─────────────────────────────────────────────
            RandGaussianNoised(keys=["image"], prob=0.25, mean=0.0, std=0.05),
            RandGaussianSmoothd(keys=["image"], prob=0.15, sigma_x=(0.5, 1.5)),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.75, 1.5)),
            RandHistogramShiftd(keys=["image"], prob=0.2, num_control_points=10),
            # ── Coarse Dropout (CutOut) ───────────────────────────────────────
            RandCoarseDropoutd(
                keys=["image"], prob=0.2,
                holes=4, spatial_size=(32, 32),
                dropout_holes=True,
            ),
        ]
        pipeline = base + aug
    else:
        pipeline = base

    pipeline.append(CastToTyped(keys=load_keys, dtype=(torch.float32,) * len(load_keys)))
    pipeline.append(ToTensord(keys=load_keys))
    return Compose(pipeline)


# =============================================================================
# 3. Dataset Sınıfı
# =============================================================================

class ChestDataset(Dataset):
    """
    40 GB medikal görüntü veri seti için OOM güvenli tembel yükleme.

    Her çağrıda yalnızca tek görüntü diskten okunur.
    Hatalı örnekler atlanır; eğitim akışı kesilmez.
    """

    def __init__(
        self,
        file_list:  List[Dict],
        transforms: Compose,
        task:       str = "segmentation",
        max_retries: int = 3,
    ) -> None:
        self.file_list   = file_list
        self.transforms  = transforms
        self.task        = task
        self.max_retries = max_retries
        logger.info(f"Dataset: {len(self.file_list)} örnek ({task})")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                data = self.transforms(self.file_list[idx])
                return data
            except Exception as e:
                logger.debug(f"[idx={idx} attempt={attempt+1}] Hata: {e}")
                idx = (idx + 1) % len(self.file_list)  # Komşuya geç

        # Tüm denemeler başarısız → sıfır tensör döndür (batch atlanır)
        logger.error(f"Örnek {idx} {self.max_retries} denemede yüklenemedi.")
        h, w = 512, 512  # Fallback boyut
        dummy: Dict[str, Any] = {
            "image": torch.zeros(1, h, w),
            "label": torch.zeros(1, h, w) if self.task == "segmentation" else torch.tensor(0),
        }
        return dummy


class CachedChestDataset(PersistentDataset):
    """
    PersistentDataset: dönüştürülmüş tensörleri diske önbelleğe alır.
    İkinci çalışmada okuma hızı 5-10x artar.
    Disk alanı: ~görüntü başına 2-4 MB (FP32 tensör).
    """

    def __init__(
        self,
        file_list:  List[Dict],
        transforms: Compose,
        cache_dir:  Path,
        task:       str = "segmentation",
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(data=file_list, transform=transforms, cache_dir=str(cache_dir))
        self.task = task
        logger.info(f"CachedDataset: {len(file_list)} örnek → önbellek: {cache_dir}")


# =============================================================================
# 4. DataLoader Fabrikası
# =============================================================================

def simple_collate(batch: List[Dict]) -> Dict[str, Tensor]:
    """
    Yalnızca image/label anahtarlarını batch'ler.
    MONAI'nin eklediği meta sözlükleri atar — bunlar tensör ve skaler
    değerleri karıştırdığı için varsayılan collate ile batch'lenemez.
    """
    def _t(x: Any) -> Tensor:
        if hasattr(x, "as_tensor"):      # MetaTensor → düz Tensor
            x = x.as_tensor()
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    return {
        "image": torch.stack([_t(b["image"]) for b in batch]),
        "label": torch.stack([_t(b["label"]) for b in batch]),
    }


def build_loaders(
    cfg:         Any,
    train_files: List[Dict],
    val_files:   List[Dict],
    test_files:  List[Dict],
    train_tfm:   Compose,
    val_tfm:     Compose,
    use_cache:   bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Eğitim / validasyon / test DataLoader'larını oluşturur.

    Bellek stratejisi:
    - use_cache=False → tam tembel yükleme (lazy loading); büyük ölçekli
      veri setlerinde (≥10 GB) bellek taşması (OOM) riskini ortadan kaldırır.
    - use_cache=True  → PersistentDataset; dönüştürülmüş tensörler diske
      serileştirilerek tekrar eden epoch'larda I/O maliyeti azaltılır.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    DS = CachedChestDataset if use_cache else ChestDataset

    if use_cache:
        train_ds = DS(train_files, train_tfm, cfg.CACHE_DIR / "train", cfg.TASK)
        val_ds   = DS(val_files,   val_tfm,   cfg.CACHE_DIR / "val",   cfg.TASK)
        test_ds  = DS(test_files,  val_tfm,   cfg.CACHE_DIR / "test",  cfg.TASK)
    else:
        train_ds = DS(train_files, train_tfm, cfg.TASK)
        val_ds   = DS(val_files,   val_tfm,   cfg.TASK)
        test_ds  = DS(test_files,  val_tfm,   cfg.TASK)

    common_kwargs = dict(
        collate_fn=simple_collate,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=(cfg.NUM_WORKERS > 0),
        prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=True,   # Batch Norm için küçük son batch'i at
        **common_kwargs,
    )
    # Segmentasyonda sliding-window çıkarımı için batch=1 güvenli;
    # sınıflandırmada sabit boyutlu girdi olduğundan tam batch kullanılır.
    eval_bs = 1 if cfg.TASK == "segmentation" else cfg.BATCH_SIZE

    val_loader = DataLoader(
        val_ds,
        batch_size=eval_bs,
        shuffle=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_bs,
        shuffle=False,
        **common_kwargs,
    )

    logger.info(
        f"DataLoaders: Train={len(train_loader)} batch | "
        f"Val={len(val_loader)} | Test={len(test_loader)}"
    )
    return train_loader, val_loader, test_loader
