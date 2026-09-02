# =============================================================================
# prepare_nih_chestxray14.py — NIH ChestX-ray14 Veri Hazırlama Betiği
# -----------------------------------------------------------------------------
# Kaggle'da "nih-chest-xrays/data" veri setini chest_pipeline'ın beklediği
# metadata.csv formatına dönüştürür (çok-etiketli sınıflandırma görevi).
#
# Kaggle Notebook'ta ilk hücreye yapıştırın:
#
#   !pip install monai[all] segmentation-models-pytorch lion-pytorch --quiet
#
#   import sys
#   sys.path.insert(0, "/kaggle/working/chest_pipeline")
#   from prepare_nih_chestxray14 import prepare
#
#   meta_csv = prepare(output_dir="/kaggle/working")
#
# NIH veri seti ham yapısı (Kaggle "nih-chest-xrays/data"):
#   /kaggle/input/data/
#     ├── Data_Entry_2017.csv      — Image Index, Finding Labels, Patient ID, ...
#     ├── train_val_list.txt       — resmî eğitim+val bölme listesi (dosya adı)
#     ├── test_list.txt            — resmî test bölme listesi (dosya adı)
#     └── images_001/images/*.png ... images_012/images/*.png  (12 alt klasör)
#
# Bu betik ne yapar:
#   1. Data_Entry_2017.csv'yi bulur, "Finding Labels" (pipe-separated) alanını
#      14 ayrı 0/1 hastalık sütununa dönüştürür ("No Finding" bir sınıf değildir
#      — bulgusu olmayan görüntülerde 14 sütun da 0'dır).
#   2. images_001..images_012 altındaki tüm PNG'leri tarayarak dosya adı → tam
#      yol eşlemesi kurar (NIH görüntüleri 12 alt klasöre dağıtılmıştır; tek
#      IMAGE_DIR ile doğrudan birleştirme çalışmaz).
#   3. train_val_list.txt / test_list.txt resmî bölmesini uygular; train_val
#      kümesini HASTA BAZLI (Patient ID) olarak train/val'a ayırır — aynı
#      hastanın farklı çekimleri train ve val'a dağılırsa model hastayı
#      ezberler ve val skoru yapay yükselir.
#   4. chest_pipeline/datasets.py::build_file_list'in MULTI_LABEL dalının
#      beklediği şemada bir CSV yazar: yalnızca image_path + 14 etiket sütunu
#      + split — fazladan sütun (Patient ID, Age, ...) OLMAMALI, aksi halde
#      "MULTI_LABEL için 14 etiket sütunu bekleniyor" hatası alınır.
# =============================================================================

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# NIH ChestX-ray14 resmî 14 bulgu sınıfı (alfabetik değil, orijinal makale sırası).
# main.py / config.py içinde cfg.CLASS_NAMES olarak da kullanılmalıdır.
NIH_CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def _find_dataset_root(search_root: str = "/kaggle/input") -> Path:
    """Data_Entry_2017.csv dosyasını /kaggle/input altında arayarak veri
    setinin kök dizinini otomatik bulur (Kaggle mount yolu sürüme göre
    değişebildiği için sabit yol yerine arama tercih edilir)."""
    root = Path(search_root)
    matches = list(root.rglob("Data_Entry_2017.csv"))
    if not matches:
        raise FileNotFoundError(
            f"'{search_root}' altında Data_Entry_2017.csv bulunamadı. "
            "NIH ChestX-ray14 veri setini ('nih-chest-xrays/data') notebook'a "
            "eklediğinizden emin olun."
        )
    return matches[0].parent


def _index_images(dataset_root: Path) -> dict:
    """images_001/images ... images_012/images altındaki tüm PNG'leri
    dosya adı → tam yol sözlüğüne indeksler."""
    index = {}
    png_paths = list(dataset_root.rglob("*.png"))
    if not png_paths:
        raise FileNotFoundError(
            f"'{dataset_root}' altında hiç .png bulunamadı. "
            "images_001..images_012 klasörlerinin mount edildiğini kontrol edin."
        )
    for p in png_paths:
        index[p.name] = str(p)
    logger.info(f"Görüntü indeksi: {len(index):,} PNG dosyası bulundu.")
    return index


def _parse_finding_labels(series: pd.Series) -> pd.DataFrame:
    """'Atelectasis|Effusion' gibi pipe-separated metni 14 sütunlu 0/1
    DataFrame'e dönüştürür."""
    label_sets = series.fillna("").apply(lambda s: set(s.split("|")))
    return pd.DataFrame(
        {cls: label_sets.apply(lambda s: int(cls in s)) for cls in NIH_CLASS_NAMES}
    )


def _official_split(
    df: pd.DataFrame,
    dataset_root: Path,
    val_ratio: float,
    seed: int,
) -> pd.Series:
    """train_val_list.txt / test_list.txt resmî bölmesini uygular;
    train_val kümesini hasta bazlı (Patient ID) train/val'a ayırır."""
    test_list_path = dataset_root / "test_list.txt"
    trainval_list_path = dataset_root / "train_val_list.txt"
    if not (test_list_path.exists() and trainval_list_path.exists()):
        raise FileNotFoundError(
            "train_val_list.txt / test_list.txt bulunamadı — resmî hasta "
            "bazlı bölme uygulanamıyor."
        )

    test_names = set(test_list_path.read_text().split())
    split = pd.Series("train", index=df.index)
    split[df["Image Index"].isin(test_names)] = "test"

    trainval_mask = split == "train"
    patients = df.loc[trainval_mask, "Patient ID"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_val_patients = max(1, int(len(patients) * val_ratio))
    val_patients = set(patients[:n_val_patients])

    is_val = trainval_mask & df["Patient ID"].isin(val_patients)
    split[is_val] = "val"
    return split


def prepare(
    output_dir:      str = "/kaggle/working",
    search_root:      str = "/kaggle/input",
    val_ratio:       float = 0.1,
    seed:            int = 42,
    output_filename: str = "metadata.csv",
) -> Path:
    """
    NIH ChestX-ray14 ham verisini chest_pipeline uyumlu metadata.csv'ye
    dönüştürür ve yazılan dosyanın yolunu döner.

    Kullanım (Kaggle):
        meta_csv = prepare()
        cfg = Config()
        cfg.TASK        = "classification"
        cfg.MULTI_LABEL = True
        cfg.NUM_CLASSES = 14
        cfg.CLASS_NAMES = NIH_CLASS_NAMES
        cfg.CSV_PATH    = meta_csv
        cfg.IMAGE_DIR   = Path("/")   # metadata.csv zaten mutlak yol içeriyor
        cfg.LOSS_NAME   = "bce"
        cfg.MODEL_NAME  = "densenet121"
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    dataset_root = _find_dataset_root(search_root)
    logger.info(f"Veri seti kökü: {dataset_root}")

    df = pd.read_csv(dataset_root / "Data_Entry_2017.csv")
    logger.info(f"Data_Entry_2017.csv yüklendi: {len(df):,} satır")

    image_index = _index_images(dataset_root)
    df["image_path"] = df["Image Index"].map(image_index)
    missing = df["image_path"].isna().sum()
    if missing:
        logger.warning(f"{missing:,} görüntü dosya sisteminde bulunamadı, atlanıyor.")
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    labels_df = _parse_finding_labels(df["Finding Labels"])
    df["split"] = _official_split(df, dataset_root, val_ratio, seed)

    out = pd.concat([df[["image_path", "split"]], labels_df], axis=1)

    # Sınıf başına pozitif oranı raporla — sınıf dengesizliği (ör. Hernia
    # ~%0.2) BCEWithLogitsLoss'un pos_weight parametresiyle telafi edilmeli.
    pos_rate = labels_df.mean().sort_values(ascending=False)
    logger.info("Sınıf başına pozitif oran:\n" + pos_rate.to_string())
    logger.info("Split dağılımı:\n" + out["split"].value_counts().to_string())

    out_path = Path(output_dir) / output_filename
    out.to_csv(out_path, index=False)
    logger.info(f"metadata.csv yazıldı → {out_path} ({len(out):,} satır)")
    return out_path


def compute_pos_weight(meta_csv: Path, class_names=NIH_CLASS_NAMES, split: str = "train"):
    """
    BCEWithLogitsLoss(pos_weight=...) için sınıf başına ağırlık hesaplar.
    NIH veri setinde ciddi sınıf dengesizliği vardır (bazı bulgular <%1);
    pos_weight olmadan model çoğunlukla "bulgu yok" tahmini yapmaya yönelir.

    Kullanım:
        import torch
        pos_weight = compute_pos_weight(meta_csv)
        criterion  = loss_factory("bce", cfg.NUM_CLASSES, pos_weight=pos_weight)
    """
    import torch
    df = pd.read_csv(meta_csv)
    df = df[df["split"] == split] if split else df
    pos = df[class_names].sum()
    neg = len(df) - pos
    weight = (neg / pos.clip(lower=1)).to_numpy(dtype="float32")
    return torch.from_numpy(weight)
