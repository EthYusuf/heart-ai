# Chest Medical Imaging — Deep Learning Analysis Pipeline

> Büyük ölçekli (≥40 GB) medikal görüntü veri setleri için
> modüler, bellek verimli, yeniden üretilebilir derin öğrenme pipeline'ı.
> PyTorch + MONAI tabanlı; segmentasyon ve sınıflandırma görevlerini destekler.

---

## Proje Yapısı

```
chest_pipeline/
├── config.py        — Merkezi hiperparametre ve yol yönetimi; deney profilleri
├── eda.py           — Keşifsel veri analizi: istatistik, kalite denetimi, görselleştirme
├── datasets.py      — Tembel yükleme (lazy loading), DICOM/NIfTI/PNG desteği, DataLoader
├── models.py        — U-Net, U-Net++, Attention U-Net, Swin-UNet, DenseNet, EfficientNet, ViT
├── losses.py        — Dice-CE, Focal, Tversky, Lovász, Boundary, Combo, Asymmetric
├── trainer.py       — K-Fold CV, SAM optimizer, MixUp, erken durdurma, Warmup LR
├── evaluation.py    — Klinik metrikler, Grad-CAM, ONNX/TorchScript export
├── main.py          — Uçtan uca pipeline giriş noktası
└── requirements.txt — Python bağımlılıkları
```

---

## Kurulum (Kaggle)

```python
# Hücre 1 — Bağımlılıkları yükle
!pip install monai[all] segmentation-models-pytorch lion-pytorch --quiet

# Hücre 2 — Veri yollarını ayarla ve pipeline'ı başlat
import sys
from pathlib import Path
sys.path.insert(0, "/kaggle/working/chest_pipeline")

from config import Config
from main   import main

cfg = Config()
cfg.DATA_ROOT  = Path("/kaggle/input/<VERI-SETI-ADI>")
cfg.IMAGE_DIR  = cfg.DATA_ROOT / "images"
cfg.MASK_DIR   = cfg.DATA_ROOT / "masks"     # Segmentasyon görevi için
cfg.CSV_PATH   = cfg.DATA_ROOT / "labels.csv"

# ÖNEMLİ: main() çağrısına cfg MUTLAKA geçirilmeli — main(cfg=None) çağrılırsa
# yukarıdaki tüm özelleştirmeler yok sayılıp sıfırdan varsayılan bir Config()
# oluşturulur.
model, metrics, cfg = main(cfg)
```

---

## NIH ChestX-ray14 — Hazır Uçtan Uca Kurulum

`prepare_nih_chestxray14.py`, Kaggle'daki **"nih-chest-xrays/data"** veri setini
(112.120 görüntü, 14 çok-etiketli bulgu) doğrudan bu pipeline'ın beklediği
`metadata.csv` şemasına dönüştürür: 12 alt klasöre dağılmış görüntüleri
indeksler, `Finding Labels` metnini 14 ayrı 0/1 sütuna ayırır ve
`train_val_list.txt` / `test_list.txt` resmî bölmesini **hasta bazlı**
(Patient ID) train/val/test olarak uygular — aynı hastanın farklı çekimlerinin
train ve val'a dağılıp sızıntıya (data leakage) yol açmasını önler.

```python
# Hücre 1 — Bağımlılıklar
!pip install monai[all] segmentation-models-pytorch lion-pytorch --quiet

# Hücre 2 — Veri setini "nih-chest-xrays/data" olarak notebook'a ekleyin
# (sağ panel → Add Input → Search → "NIH Chest X-rays"), sonra:
import sys
sys.path.insert(0, "/kaggle/working/chest_pipeline")

from prepare_nih_chestxray14 import prepare, compute_pos_weight, NIH_CLASS_NAMES
from config   import Config
from main     import main

meta_csv = prepare(output_dir="/kaggle/working")   # ~1-2 dakika (12 klasör taraması)

# Hücre 3 — Yapılandırma
cfg = Config()
cfg.TASK        = "classification"
cfg.MULTI_LABEL = True
cfg.NUM_CLASSES = 14
cfg.CLASS_NAMES = NIH_CLASS_NAMES
cfg.CSV_PATH    = meta_csv
cfg.IMAGE_DIR   = Path("/")        # metadata.csv zaten mutlak yol içeriyor
cfg.IN_CHANNELS = 1
cfg.MODEL_NAME  = "densenet121"    # CheXNet mimarisi — bu görev için referans
cfg.LOSS_NAME   = "bce"
cfg.EPOCHS      = 30
cfg.BATCH_SIZE  = 32
cfg.KFOLD       = 1                 # 112K görüntüde K-Fold yerine tek bölme önerilir

# Hücre 4 — Sınıf dengesizliği telafisi (Hernia ~%0.2 pozitif vb.)
# main.py içindeki loss_factory çağrısına elle pos_weight geçirmek isterseniz:
#   from losses import loss_factory
#   pos_weight = compute_pos_weight(meta_csv)
#   criterion  = loss_factory("bce", cfg.NUM_CLASSES, pos_weight=pos_weight)
# main() bu özel criterion'ı kabul etmez; ileri seviye kullanım için
# run_standard_training()'i elle çağırıp kendi criterion'ınızı geçirin.

# Hücre 5 — Çalıştır
model, metrics, cfg = main(cfg)
```

> **Not:** `cfg.RUN_EDA = True` (varsayılan) ile ilk çalıştırmada veri kalitesi
> raporu (`eda/quality_report.csv`) ve sınıf dağılımı grafikleri otomatik
> üretilir — bozuk/aykırı görüntüleri eğitim başlamadan görebilirsiniz.

---

## Teknik Bileşenler

### Bellek Yönetimi ve OOM Önleme

| Teknik | Mekanizma |
|--------|-----------|
| Tembel yükleme (lazy loading) | Her örnek yalnızca `__getitem__` çağrıldığında diskten okunur |
| Otomatik karma hassasiyet (AMP) | FP32 → FP16 hesaplama; GPU bellek kullanımını yaklaşık %40 düşürür |
| `pin_memory=True` | Ana bellek sayfaları sabitlenir; CPU→GPU DMA transferi asenkron yürütülür |
| `persistent_workers=True` | DataLoader iş parçacıkları epoch arası yeniden başlatılmaz |
| `zero_grad(set_to_none=True)` | Gradient tensörleri bellekten tamamen kaldırılır (sıfırlama yerine) |
| Sliding window inference | Büyük görüntüler örtüşen alt bölgelere ayrılarak tahmin edilir |
| `torch.cuda.empty_cache()` | Belirli aralıklarla CUDA bellek önbelleği serbest bırakılır |

### Model Mimarileri

| Model | Görev | Literatür Referansı |
|-------|-------|---------------------|
| U-Net | Segmentasyon | Ronneberger et al., MICCAI 2015 |
| U-Net++ | Segmentasyon | Zhou et al., DLMIA 2018 |
| Attention U-Net | Segmentasyon | Oktay et al., MIDL 2018 |
| Swin-UNet | Segmentasyon | Cao et al., ECCV 2022 |
| DenseNet-121 | Sınıflandırma | Huang et al., CVPR 2017; Rajpurkar et al. (CheXNet) 2017 |
| EfficientNet-B4 | Sınıflandırma | Tan & Le, ICML 2019 |
| ViT-B/16 | Sınıflandırma | Dosovitskiy et al., ICLR 2021 |

### Kayıp Fonksiyonları

| Fonksiyon | Matematiksel Temel | Kullanım Durumu |
|-----------|-------------------|-----------------|
| Dice-CE | `λ·L_Dice + (1−λ)·L_CE` | Genel amaçlı medikal segmentasyon |
| Tversky | Asimetrik TP/FP/FN ağırlıklandırması | FN'in FP'ye oranla daha maliyetli olduğu durumlar |
| Lovász-Softmax | Jaccard indeksinin Lovász uzantısı | IoU metriğinin doğrudan optimizasyonu |
| Boundary | Mesafe dönüşümü tabanlı sınır kaybı | İnce yapı (damar, nodül) kenar hassasiyeti |
| Combo | `α·Dice + β·Boundary + γ·Focal` | Çoklu hedefin bileşik optimizasyonu |
| Asymmetric | Pozitif/negatif asimetrik odaklama | Çok etiketli sınıflandırma, ciddi sınıf dengesizliği |

### Eğitim Teknikleri

| Teknik | Referans | Etki |
|--------|----------|------|
| SAM (Sharpness-Aware Minimization) | Foret et al., ICLR 2021 | Düz minimum bölgelerine yönlendirerek genelleme kapasitesini artırır |
| Stratified K-Fold CV | Kohavi, IJCAI 1995 | Sınıf dağılımı korunarak yansız genelleme hatası tahmini sağlar |
| MixUp | Zhang et al., ICLR 2018 | Doğrusal ara değerleme ile düzenlileştirme ve olasılık kalibrasyonu |
| Test Time Augmentation (TTA) | Wang et al., MICCAI 2019 | 8 geometrik dönüşümün ortalamasıyla tahminde varyans azaltımı |
| Discriminative Fine-Tuning | Howard & Ruder, ACL 2018 | Katman derinliğine göre azalan öğrenme hızı; aşırı sığdırmayı sınırlar |
| Warmup + Cosine Annealing | Loshchilov & Hutter, ICLR 2017 | Başlangıç kararsızlığını önler; periyodik yeniden başlatma ile yerel minimumdan çıkar |

### Değerlendirme Metrikleri

**Segmentasyon**
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU / Jaccard)
- Hausdorff Distance (95. persentil, HD95) — sınır doğruluğu
- Sensitivite (Recall / TPR), Özgüllük (Specificity / TNR), Kesinlik (Precision)
- Hacim Benzerliği (Volume Similarity)

**Sınıflandırma**
- Area Under ROC Curve (AUC-ROC)
- Area Under Precision-Recall Curve (PR-AUC / Average Precision)
- F1 Skoru (makro ve ağırlıklı ortalama)
- Matthews Korelasyon Katsayısı (MCC)
- Brier Skoru — olasılıksal kalibrasyon hatası
- Güvenilirlik Diyagramı (Reliability Diagram / Calibration Curve)

**Açıklanabilirlik (XAI)**
- Grad-CAM (Gradient-weighted Class Activation Mapping, Selvaraju et al., ICCV 2017)

---

## Yapılandırma Profilleri

```python
from config import Config, QuickDebugConfig, HighPerformanceConfig

# Sistem entegrasyon testi (3 epoch, 256×256 görüntü)
cfg = QuickDebugConfig()

# Standart deneysel yapılandırma
cfg = Config()

# Tam ölçekli yayın yapılandırması (200 epoch, 5-Fold, TTA, SAM)
cfg = HighPerformanceConfig()
```

### Zorunlu Yol Güncellemeleri

```python
cfg.DATA_ROOT  = Path("/kaggle/input/<VERI-SETI-ADI>")
cfg.IMAGE_DIR  = cfg.DATA_ROOT / "images"
cfg.MASK_DIR   = cfg.DATA_ROOT / "masks"      # yalnızca segmentasyon görevi
cfg.CSV_PATH   = cfg.DATA_ROOT / "labels.csv" # sütunlar: image_path, label/mask_path

cfg.TASK        = "segmentation"   # "segmentation" | "classification"
cfg.MODEL_NAME  = "unet"
cfg.LOSS_NAME   = "dice_ce"
cfg.KFOLD       = 5
cfg.BATCH_SIZE  = 4
cfg.EPOCHS      = 100
```

---

## Çıktı Dizin Yapısı

```
/kaggle/working/outputs/<RUN_ID>/
├── best_model.pth             — En yüksek validasyon metriğine sahip model ağırlıkları
├── config.json                — Deney yapılandırması (yeniden üretilebilirlik için)
├── training_history.csv       — Epoch bazlı eğitim ve validasyon metrikleri
├── training_curves.png        — Kayıp, metrik ve öğrenme hızı eğrileri
├── kfold_summary.csv          — K-Fold fold bazlı performans özeti (uygulanabiliyorsa)
├── final_report.json          — Test seti nihai sonuçları
├── <EXP_NAME>.onnx            — ONNX formatında export edilmiş model
├── eda/
│   ├── image_stats.csv        — Görüntü başına istatistikler
│   ├── quality_report.csv     — Veri kalitesi sorunları (aykırı değer, bozuk dosya)
│   ├── label_distribution.png
│   ├── image_size_analysis.png
│   ├── intensity_analysis.png
│   ├── classwise_intensity.png
│   ├── correlation_matrix.png
│   ├── mask_analysis.png      — Segmentasyon görevlerinde ön plan istatistikleri
│   └── dicom_metadata.png     — DICOM formatında meta veri analizi
└── evaluation/
    ├── test_metrics.json
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── pr_curves.png
    ├── calibration_curve.png
    ├── segmentation_metrics.png
    ├── prediction_samples.png
    └── gradcam.png
```

---

## Referanslar

1. Ronneberger O. et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
2. Zhou Z. et al., *UNet++: A Nested U-Net Architecture for Medical Image Segmentation*, DLMIA 2018.
3. Oktay O. et al., *Attention U-Net: Learning Where to Look for the Pancreas*, MIDL 2018.
4. Cao H. et al., *Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation*, ECCV 2022.
5. Milletari F. et al., *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*, 3DV 2016.
6. Berman M. et al., *The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the IoU Measure in Neural Networks*, CVPR 2018.
7. Kervadec H. et al., *Boundary Loss for Highly Unbalanced Segmentation*, MIDL 2019.
8. Foret P. et al., *Sharpness-Aware Minimization for Efficiently Improving Generalization*, ICLR 2021.
9. Selvaraju R.R. et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
10. Tan M. & Le Q.V., *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019.
11. Rajpurkar P. et al., *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*, arXiv 2017.
12. Zhang H. et al., *mixup: Beyond Empirical Risk Minimization*, ICLR 2018.
