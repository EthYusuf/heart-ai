# =============================================================================
# main.py — Ana Giriş Noktası
# -----------------------------------------------------------------------------
# Kaggle Notebook'ta tek bir hücreye bu dosyanın içeriğini yapıştırın
# veya: exec(open("chest_pipeline/main.py").read())
#
# Pipeline Akışı:
#   1. Ortamı Hazırla (seed, logging, GPU doğrulama)
#   2. EDA — Keşifsel Veri Analizi
#   3. DataLoader'ları Kur
#   4. Model + Kayıp + Optimizer Seç
#   5. K-Fold Eğitim (veya standart eğitim)
#   6. En İyi Model Yükle + TTA Inference
#   7. Kapsamlı Değerlendirme + Grad-CAM
#   8. ONNX Export + Sonuç Raporu
# =============================================================================

# ── Gerekli kurulumlar (Kaggle'da ilk hücrede çalıştırın) ────────────────────
# !pip install monai[all] segmentation-models-pytorch albumentations lion_pytorch --quiet

import gc
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# DÜZELTME: torch.cuda.amp.GradScaler yerine cihaz-bağımsız torch.amp API'si.
from torch.amp import GradScaler

warnings.filterwarnings("ignore")

# ── Yerel modülleri içe aktar ─────────────────────────────────────────────────
# Kaggle'da tüm dosyalar aynı çalışma dizininde olmalı
sys.path.insert(0, str(Path(__file__).parent))

from config     import Config, QuickDebugConfig, HighPerformanceConfig
from eda        import MedicalEDA
from datasets   import build_file_list, get_transforms, build_loaders
from models     import model_factory, EnsembleModel, get_layer_wise_lr_params
from losses     import loss_factory
from trainer    import (
    KFoldTrainer, EarlyStopping, WarmupScheduler,
    optimizer_factory, scheduler_factory,
    train_epoch, validate_epoch,
)
from evaluation import (
    SegmentationEvaluator, ClassificationEvaluator,
    EvaluationVisualizer, GradCAM,
    full_evaluation, export_to_onnx,
)


# =============================================================================
# Logging Kurulumu
# =============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    """Hem terminale hem dosyaya log yazar."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # DÜZELTME: Windows Türkçe konsol (cp1254) '→' ve '✓' gibi Unicode
    # karakterleri kodlayamıyor; her log satırı için "--- Logging error ---"
    # yığını basılıyordu. Çalıştırmayı DURDURMAZ ama konsolu çöpe çevirir.
    # Konsol akışları UTF-8'e çevrilerek kökten çözülür; kodlanamayan tek
    # karakter kalırsa 'replace' ile '?' olarak basılır, hata yığını yok.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass  # stdin/stdout yönlendirilmişse veya zaten kapanmışsa umursama

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_path), mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,   # Jupyter kök logger'ı önceden yapılandırır; force olmadan
                      # basicConfig sessizce hiçbir şey yapmaz ve INFO logları kaybolur
    )
    return logging.getLogger(__name__)


# =============================================================================
# Metrik Fonksiyonu (trainer'a geçirilir)
# =============================================================================

def build_loss_kwargs(cfg: Config) -> dict:
    """
    LOSS_NAME'e göre loss_factory'ye geçirilecek doğru anahtar kelime
    argümanlarını üretir.

    DÜZELTME: Önceki sürümde tüm loss'lara aynı {"alpha", "alpha_tv", "beta"}
    sözlüğü geçiriliyordu. loss_factory yalnızca hedef fonksiyonun imzasında
    bulunan anahtarları kabul ettiğinden:
      - Tversky için "alpha_tv" hiçbir zaman eşleşmiyordu → cfg.TVERSKY_ALPHA
        sessizce yok sayılıyor, yanlışlıkla cfg.LOSS_ALPHA "alpha" olarak
        kullanılıyordu.
      - Combo için "beta" anahtarı cfg.TVERSKY_BETA (0.7) değerini taşıyordu;
        ComboLoss.beta parametresiyle çakışıyor ve alpha+beta+gamma toplamı
        1.0'ı aşarak `assert` hatasıyla pipeline'ı çökertiyordu
        (0.6 + 0.7 + 0.2 = 1.5).
    Artık her loss kendi anlamlı parametrelerini alıyor.
    """
    if cfg.LOSS_NAME == "tversky":
        return {"alpha": cfg.TVERSKY_ALPHA, "beta": cfg.TVERSKY_BETA}
    if cfg.LOSS_NAME == "combo":
        # ComboLoss alpha+beta+gamma == 1.0 zorunluluğu var; beta/gamma'yı
        # LOSS_ALPHA'dan bağımsız sabit değerler olarak geçirmek (ör. TVERSKY_BETA)
        # kullanıcı LOSS_ALPHA'yı değiştirdiğinde toplamı bozup assert hatası
        # verebiliyordu. Kalan payı beta/gamma'ya eşit bölerek toplamı her
        # zaman 1.0'da sabitliyoruz.
        remainder = (1.0 - cfg.LOSS_ALPHA) / 2
        return {"alpha": cfg.LOSS_ALPHA, "beta": remainder, "gamma": remainder}
    if cfg.LOSS_NAME == "focal":
        return {"gamma": 2.0}
    return {}


def make_metrics_fn(cfg: Config):
    """
    validate_epoch'a geçirilecek metrik fonksiyonunu oluşturur.
    Döndürdüğü değer erken durdurma ve checkpoint için kullanılır.
    """
    if cfg.TASK == "segmentation":
        from monai.metrics import DiceMetric
        from monai.networks.utils import one_hot
        dice_m = DiceMetric(include_background=False, reduction="mean")

        def seg_metric(preds: torch.Tensor, targets: torch.Tensor) -> float:
            # DÜZELTME: preds/targets tek kanallı sınıf indeksi (argmax) olarak
            # geçiriliyordu. include_background=False ile MONAI DiceMetric,
            # tek kanallı bir tensörde "arka planı" çıkaracak kanal
            # bulamadığı için bu yalnızca NUM_CLASSES=2 (ikili) durumda
            # tesadüfen doğru sonuç veriyordu; NUM_CLASSES>2 için sayısal
            # olarak yanlış Dice üretiyordu. One-hot kodlama ile düzeltildi.
            n_classes = preds.shape[1]
            preds_binary = torch.argmax(torch.softmax(preds, dim=1), dim=1, keepdim=True)
            preds_oh   = one_hot(preds_binary.long(), num_classes=n_classes)
            targets_oh = one_hot(targets.long(),       num_classes=n_classes)
            dice_m(y_pred=preds_oh, y=targets_oh)
            score = dice_m.aggregate().item()
            dice_m.reset()
            return score

        return seg_metric

    elif getattr(cfg, "MULTI_LABEL", False):
        from sklearn.metrics import roc_auc_score

        def multilabel_metric(preds: torch.Tensor, targets: torch.Tensor) -> float:
            """
            Sınıf başına AUC-ROC'un makro ortalaması.
            Validasyon setinde tek sınıflı (hep 0 veya hep 1) kalan sütunlar
            AUC tanımsız olduğu için ortalamaya dahil edilmez.
            """
            probs  = torch.sigmoid(preds.float()).numpy()
            labels = targets.numpy()
            aucs = [
                roc_auc_score(labels[:, c], probs[:, c])
                for c in range(labels.shape[1])
                if 0 < labels[:, c].sum() < len(labels)
            ]
            return float(np.mean(aucs)) if aucs else 0.0

        return multilabel_metric

    else:
        from sklearn.metrics import roc_auc_score

        def cls_metric(preds: torch.Tensor, targets: torch.Tensor) -> float:
            probs  = torch.softmax(preds, dim=1).numpy()
            labels = targets.numpy().flatten()
            try:
                if cfg.NUM_CLASSES == 2:
                    return float(roc_auc_score(labels, probs[:, 1]))
                else:
                    return float(roc_auc_score(
                        labels, probs, multi_class="ovr", average="macro"
                    ))
            except Exception:
                return float((probs.argmax(1) == labels).mean())

        return cls_metric


# =============================================================================
# Eğitim Döngüsü (K-Fold değil, tek fold)
# =============================================================================

def run_standard_training(
    cfg:          Config,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    logger:       logging.Logger,
) -> nn.Module:
    """
    Tek holdout bölmesiyle standart eğitim döngüsü (train/val/test split).
    Sınırlı hesaplama bütçesi veya ablasyon çalışmalarında tercih edilir.
    İstatistiksel güvenilirlik gerektiren nihai değerlendirmeler için
    K-Fold çapraz doğrulama (KFoldTrainer) kullanılması önerilir.
    """
    model     = model_factory(cfg.MODEL_NAME, cfg.IN_CHANNELS, cfg.NUM_CLASSES,
                               cfg.DEVICE, cfg.IMAGE_SIZE, cfg.PRETRAINED)
    criterion = loss_factory(cfg.LOSS_NAME, cfg.NUM_CLASSES, **build_loss_kwargs(cfg))
    metrics_fn  = make_metrics_fn(cfg)

    # Katman bazlı öğrenme hızı (discriminative fine-tuning)
    param_groups = get_layer_wise_lr_params(model, cfg.LEARNING_RATE, lr_factor=0.3)
    optimizer    = optimizer_factory(cfg.OPTIMIZER, param_groups,
                                     lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    base_sched   = scheduler_factory(cfg.SCHEDULER, optimizer, cfg.EPOCHS,
                                     len(train_loader), cfg.T0, cfg.T_MULT, cfg.MIN_LR)
    # DÜZELTME: OneCycleLR batch-başına, ReduceLROnPlateau ise metrik
    # argümanıyla adımlanmalı — bkz. trainer.py::WarmupScheduler.
    scheduler    = WarmupScheduler(
        optimizer, cfg.WARMUP_EPOCHS, base_sched, cfg.LEARNING_RATE,
        per_batch=(cfg.SCHEDULER == "one_cycle"),
        is_plateau=(cfg.SCHEDULER == "plateau"),
    )
    scaler       = GradScaler(cfg.DEVICE.type, enabled=cfg.USE_AMP)
    early_stop   = EarlyStopping(cfg.EARLY_STOPPING_PATIENCE, cfg.EARLY_STOPPING_MIN_DELTA)

    best_metric  = -np.inf
    history      = []

    logger.info("=" * 60)
    logger.info(f"Standart Eğitim: {cfg.EPOCHS} epoch | Model: {cfg.MODEL_NAME}")
    logger.info("=" * 60)

    for epoch in range(cfg.EPOCHS):
        use_mixup = (np.random.rand() < cfg.MIXUP_PROB and cfg.TASK == "classification")

        train_res = train_epoch(model, train_loader, optimizer, criterion,
                                scaler, cfg, epoch, use_mixup=use_mixup,
                                scheduler=scheduler)
        val_res   = validate_epoch(model, val_loader, criterion, metrics_fn, cfg,
                                   use_tta=(cfg.USE_TTA and epoch >= cfg.EPOCHS - 5))
        scheduler.step(metric=val_res["metric"])

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_res["loss"],
            "val_loss":   val_res["loss"],
            "val_metric": val_res["metric"],
            "lr":         train_res["lr"],
            "time_s":     train_res["time_s"],
        })

        logger.info(
            f"Epoch [{epoch+1:03d}/{cfg.EPOCHS}] "
            f"TrainL={train_res['loss']:.4f} "
            f"ValL={val_res['loss']:.4f} "
            f"Metric={val_res['metric']:.4f} "
            f"LR={train_res['lr']:.2e} "
            f"({train_res['time_s']:.0f}s)"
        )

        if val_res["metric"] > best_metric:
            best_metric = val_res["metric"]
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "metric": best_metric,
            }, cfg.CHECKPOINT_PATH)
            logger.info(f"  ★ Yeni en iyi model kaydedildi: {best_metric:.4f}")

        if cfg.EARLY_STOPPING and early_stop(val_res["metric"]):
            logger.info(f"Erken durdurma: epoch {epoch+1}")
            break

        del train_res, val_res
        torch.cuda.empty_cache(); gc.collect()

    # Eğitim geçmişini kaydet
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(cfg.CHECKPOINT_PATH.parent / "training_history.csv", index=False)
    _plot_training_curves(hist_df, cfg.CHECKPOINT_PATH.parent)

    logger.info(f"Eğitim tamamlandı. En iyi metrik: {best_metric:.4f}")
    return model


# =============================================================================
# Eğitim Eğrisi Grafiği
# =============================================================================

def _plot_training_curves(history_df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Eğitim Süreci", fontsize=13, fontweight="bold")

    e = history_df["epoch"]
    axes[0].plot(e, history_df["train_loss"], "b-", lw=2, label="Eğitim")
    axes[0].plot(e, history_df["val_loss"],   "r-", lw=2, label="Validasyon")
    axes[0].set_title("Kayıp"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(e, history_df["val_metric"], "g-", lw=2)
    axes[1].set_title("Validasyon Metriği (Dice/AUC)"); axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)

    axes[2].plot(e, history_df["lr"], "m-", lw=2)
    axes[2].set_title("Öğrenme Hızı"); axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log"); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# ANA PIPELINE
# =============================================================================

def main(cfg=None):
    # ── 0. Yapılandırma ───────────────────────────────────────────────────────
    # Deneysel koşula göre yapılandırma profili seçin:
    # cfg = QuickDebugConfig()         # Sistem entegrasyon testi (minimal ölçek)
    # cfg = HighPerformanceConfig()    # Tam ölçekli deneysel yayın yapılandırması
    if cfg is None:
        cfg = Config()                 # Standart deneysel yapılandırma

    logger = setup_logging(cfg.LOG_PATH)
    cfg.save()  # Deney yapılandırmasını kaydet (yeniden üretilebilirlik)

    logger.info("=" * 60)
    logger.info(f"DENEY: {cfg.RUN_ID}")
    logger.info(f"Görev: {cfg.TASK} | Model: {cfg.MODEL_NAME} | Kayıp: {cfg.LOSS_NAME}")
    logger.info("=" * 60)

    # ── 1. Veri Listesi ───────────────────────────────────────────────────────
    train_files, val_files, test_files = build_file_list(cfg)

    # ── 2. Keşifsel Veri Analizi (EDA) ───────────────────────────────────────
    # Eğitim öncesi veri setini tanımak, bozuk/anormal görüntüleri tespit
    # etmek için çalışır. cfg.RUN_EDA = False ile kapatılabilir
    # (bkz. config.py). Büyük veri setlerinde 10-30 dakika sürebilir.
    if cfg.RUN_EDA:
        all_files  = train_files + val_files + test_files
        image_paths = [Path(f["image"]) for f in all_files]

        def _readable_label(f):
            lbl = f.get("label", 0)
            if isinstance(lbl, torch.Tensor):
                # DÜZELTME: MULTI_LABEL=True iken f["label"] çok-etiketli
                # (multi-hot) bir tensördü ve doğrudan DataFrame'e
                # konuyordu. torch.Tensor kimlik (id) bazlı hash'lendiğinden
                # her satır "benzersiz" görünüyor; plot_label_distribution /
                # plot_classwise_intensity (value_counts/groupby kullanır)
                # her görüntü için ayrı bir çubuk çizmeye çalışıp anlamsız —
                # binlerce görüntüde pratikte kullanılamaz — bir grafik
                # üretiyordu. Aktif bulguların okunabilir isim listesine
                # (ör. "Effusion|Infiltration", hiçbiri yoksa "No Finding")
                # dönüştürülerek EDA anlamlı hale getirildi.
                names = getattr(cfg, "CLASS_NAMES", None) or [str(i) for i in range(len(lbl))]
                active = [names[i] for i, v in enumerate(lbl.tolist()) if v > 0]
                return "|".join(active) if active else "No Finding"
            return lbl

        label_data  = {
            "path":  [f["image"] for f in all_files],
            "label": [_readable_label(f) for f in all_files],
        }
        labels_df = pd.DataFrame(label_data)

        eda = MedicalEDA(
            image_paths=image_paths,
            labels_df=labels_df,
            output_dir=cfg.CHECKPOINT_PATH.parent / "eda",
            fmt=cfg.IMAGE_FORMAT,
            n_workers=cfg.NUM_WORKERS,
            sample_cap=min(cfg.EDA_SAMPLE_CAP, len(image_paths)),
        )
        eda_results = eda.run_full_analysis(
            mask_paths=[Path(f["label"]) for f in all_files
                        if cfg.TASK == "segmentation" and isinstance(f.get("label"), str)]
        )
        logger.info(f"EDA tamamlandı. Kalite raporu: {len(eda_results['quality_report'])} sorun")

    # ── 3. Dönüşümler & DataLoader'lar ───────────────────────────────────────
    train_tfm = get_transforms(cfg, mode="train")
    val_tfm   = get_transforms(cfg, mode="val")

    train_loader, val_loader, test_loader = build_loaders(
        cfg, train_files, val_files, test_files,
        train_tfm, val_tfm,
        use_cache=(cfg.CACHE_RATE > 0),
    )

    # ── 4. Eğitim Stratejisi ──────────────────────────────────────────────────
    if cfg.KFOLD > 1:
        # K-Fold Çapraz Doğrulama
        logger.info(f"{cfg.KFOLD}-Fold çapraz doğrulama başlıyor...")
        from datasets import ChestDataset

        all_files  = train_files + val_files   # Test seti dışarıda tutulur
        # DÜZELTME: Önceden burada hiçbir kwarg geçirilmiyordu; K-Fold yolu
        # cfg.LOSS_ALPHA / TVERSKY_ALPHA / TVERSKY_BETA'yı tamamen yok sayıp
        # her zaman fabrika varsayılanlarını kullanıyordu (standart eğitim
        # yoluyla tutarsız davranış).
        criterion  = loss_factory(cfg.LOSS_NAME, cfg.NUM_CLASSES, **build_loss_kwargs(cfg))
        metrics_fn = make_metrics_fn(cfg)

        kf_trainer = KFoldTrainer(
            cfg=cfg,
            dataset_files=all_files,
            model_fn=lambda: model_factory(
                cfg.MODEL_NAME, cfg.IN_CHANNELS, cfg.NUM_CLASSES,
                cfg.DEVICE, cfg.IMAGE_SIZE, cfg.PRETRAINED
            ),
            criterion=criterion,
            metrics_fn=metrics_fn,
            train_tfm=train_tfm,
            val_tfm=val_tfm,
            dataset_cls=ChestDataset,
        )
        kfold_summary = kf_trainer.run()
        logger.info(f"\nK-Fold özet:\n{kfold_summary}")

        # En iyi fold modelini seç (test için)
        best_fold  = kfold_summary.loc[kfold_summary["best_metric"].idxmax(), "fold"]
        best_ckpt  = cfg.CHECKPOINT_PATH.parent / f"fold{best_fold}_best.pth"
        final_model = model_factory(
            cfg.MODEL_NAME, cfg.IN_CHANNELS, cfg.NUM_CLASSES,
            cfg.DEVICE, cfg.IMAGE_SIZE, cfg.PRETRAINED,
        )
        state = torch.load(best_ckpt, map_location=cfg.DEVICE)
        final_model.load_state_dict(state["model_state"])
        logger.info(f"En iyi fold: {best_fold} → {best_ckpt}")

    else:
        # Standart Eğitim
        final_model = run_standard_training(cfg, train_loader, val_loader, logger)
        # En iyi checkpoint'i yükle
        state = torch.load(cfg.CHECKPOINT_PATH, map_location=cfg.DEVICE)
        final_model.load_state_dict(state["model_state"])
        logger.info("En iyi model yüklendi.")

    # ── 5. Kapsamlı Test Değerlendirmesi ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Test Seti Değerlendirmesi")
    logger.info("=" * 60)

    test_metrics = full_evaluation(
        model=final_model,
        loader=test_loader,
        cfg=cfg,
        output_dir=cfg.CHECKPOINT_PATH.parent / "evaluation",
        class_names=(cfg.CLASS_NAMES
                     or [f"Sınıf {i}" for i in range(cfg.NUM_CLASSES)]),
    )

    logger.info("\n──── SONUÇLAR ────")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:40s}: {v:.4f}")

    # ── 6. Grad-CAM (Sınıflandırma için Açıklanabilirlik) ────────────────────
    if cfg.TASK == "classification":
        try:
            # Son konvolüsyonel katmanı bul
            target_layer = None
            for name, module in reversed(list(final_model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    logger.info(f"Grad-CAM hedef katman: {name}")
                    break

            if target_layer:
                gradcam   = GradCAM(final_model, target_layer)
                batch     = next(iter(test_loader))
                images    = batch["image"].to(cfg.DEVICE)
                labels    = batch["label"]
                gradcam.visualize_batch(
                    images=images,
                    labels=labels,
                    output_path=cfg.CHECKPOINT_PATH.parent / "evaluation" / "gradcam.png",
                    class_names=(cfg.CLASS_NAMES
                     or [f"Sınıf {i}" for i in range(cfg.NUM_CLASSES)]),
                    n=min(6, images.shape[0]),
                )
        except Exception as e:
            logger.warning(f"Grad-CAM oluşturulamadı: {e}")

    # ── 7. Ensemble (K-Fold modelleri varsa) ──────────────────────────────────
    if cfg.KFOLD > 1:
        logger.info("Ensemble değerlendirmesi...")
        fold_ckpts = [
            cfg.CHECKPOINT_PATH.parent / f"fold{k+1}_best.pth"
            for k in range(cfg.KFOLD)
            if (cfg.CHECKPOINT_PATH.parent / f"fold{k+1}_best.pth").exists()
        ]
        if len(fold_ckpts) > 1:
            sub_models = [
                model_factory(cfg.MODEL_NAME, cfg.IN_CHANNELS, cfg.NUM_CLASSES,
                               cfg.DEVICE, cfg.IMAGE_SIZE)
                for _ in fold_ckpts
            ]
            ensemble = EnsembleModel(sub_models, task=cfg.TASK,
                                      multi_label=getattr(cfg, "MULTI_LABEL", False))
            ensemble.load_checkpoints(fold_ckpts, cfg.DEVICE)

            ensemble_metrics = full_evaluation(
                model=ensemble,
                loader=test_loader,
                cfg=cfg,
                output_dir=cfg.CHECKPOINT_PATH.parent / "ensemble_evaluation",
            )
            logger.info("Ensemble metrikleri kaydedildi.")

    # ── 8. Özet Rapor ─────────────────────────────────────────────────────────
    report = {
        "run_id":        cfg.RUN_ID,
        "task":          cfg.TASK,
        "model":         cfg.MODEL_NAME,
        "loss":          cfg.LOSS_NAME,
        "epochs":        cfg.EPOCHS,
        "best_val_metric": float(state.get("metric", 0.0)),
        "test_metrics":  {k: v for k, v in test_metrics.items()
                          if isinstance(v, (int, float))},
        "timestamp":     datetime.now().isoformat(),
    }
    with open(cfg.CHECKPOINT_PATH.parent / "final_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("Pipeline tamamlandı!")
    logger.info(f"Tüm çıktılar → {cfg.CHECKPOINT_PATH.parent}")
    logger.info("=" * 60)

    return final_model, test_metrics, cfg


# =============================================================================
# Başlangıç Noktası
# =============================================================================

if __name__ == "__main__":
    model, metrics, cfg = main()
