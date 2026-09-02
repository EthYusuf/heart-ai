# =============================================================================
# evaluation.py — Kapsamlı Değerlendirme & Metrik Modülü
# -----------------------------------------------------------------------------
# Segmentasyon ve sınıflandırma için tüm klinik ve makine öğrenimi metrikleri.
# Medikal değerlendirmede standard: sensitivite, özgüllük, AUC, Dice, IoU.
#
# Ek olarak:
#   - Sınıf bazlı metrik raporlama
#   - ROC / PR eğrileri
#   - Karmaşıklık matrisi (Confusion Matrix)
#   - Kalibasyon eğrisi (Reliability Diagram)
#   - Grad-CAM açıklanabilirlik haritaları
#   - Test Time Augmentation ile inference
#   - ONNX model export
# =============================================================================

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    f1_score, matthews_corrcoef,
    brier_score_loss,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# DÜZELTME: torch.cuda.amp.autocast yerine cihaz-bağımsız torch.amp API'si
# (bkz. trainer.py'deki aynı düzeltme).
from torch.amp import autocast
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.networks.utils import one_hot

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =============================================================================
# 1. Segmentasyon Metrikleri
# =============================================================================

class SegmentationEvaluator:
    """
    Segmentasyon görevi için kapsamlı metrik hesaplayıcı.

    Metrikler:
        - Dice Coefficient (her sınıf ve ortalama)
        - IoU / Jaccard Index
        - Hausdorff Distance (95. persentil) — sınır doğruluğu
        - Sensitivite (Recall)
        - Özgüllük (Specificity)
        - Precision
        - Volume Similarity
    """

    def __init__(self, num_classes: int, include_background: bool = False) -> None:
        self.num_classes        = num_classes
        self.include_background = include_background
        self.reset()

    def reset(self) -> None:
        """Tüm istatistik birikimini sıfırlar."""
        self.dice_metric = DiceMetric(
            include_background=self.include_background,
            reduction="mean_batch",
            get_not_nans=True,
        )
        self.iou_metric = MeanIoU(
            include_background=self.include_background,
            reduction="mean_batch",
            get_not_nans=True,
        )
        self.hd_metric = HausdorffDistanceMetric(
            include_background=self.include_background,
            percentile=95,
            reduction="mean_batch",
            get_not_nans=True,
        )
        self._tp = np.zeros(self.num_classes)
        self._fp = np.zeros(self.num_classes)
        self._fn = np.zeros(self.num_classes)
        self._tn = np.zeros(self.num_classes)

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Batch tahmini ile gerçek maskeyi karşılaştırarak istatistik biriktirir.
        preds  : (B, C, H, W) softmax çıktısı
        targets: (B, 1, H, W) sınıf indisleri
        """
        preds_binary = torch.argmax(preds, dim=1, keepdim=True).long()
        targets_long = targets.long()

        # DÜZELTME: MONAI'nin Dice/IoU/Hausdorff metrikleri include_background=False
        # ile doğru çalışabilmesi için one-hot kodlanmış (B, num_classes, H, W)
        # girdi bekler. Önceden tek kanallı sınıf-indeks tensörü doğrudan
        # geçiriliyordu; bu yalnızca NUM_CLASSES=2 (ikili) durumda tesadüfen
        # anlamlı sonuç veriyor, NUM_CLASSES>2 için matematiksel olarak
        # yanlış Dice/IoU üretiyordu.
        n_classes  = preds.shape[1]
        preds_oh   = one_hot(preds_binary, num_classes=n_classes)
        targets_oh = one_hot(targets_long, num_classes=n_classes)

        self.dice_metric(y_pred=preds_oh, y=targets_oh)
        self.iou_metric(y_pred=preds_oh,  y=targets_oh)
        # DÜZELTME: hd_metric önceden hiç çağrılmıyordu — HD95 (README'de
        # vaat edilen sınır doğruluğu metriği) hiçbir zaman hesaplanmıyordu.
        self.hd_metric(y_pred=preds_oh, y=targets_oh)

        # TP/FP/FN/TN hesabı (piksel bazlı)
        for c in range(self.num_classes):
            pred_c   = (preds_binary[:, 0] == c)
            target_c = (targets_long[:, 0] == c)
            self._tp[c] += (pred_c  & target_c).sum().item()
            self._fp[c] += (pred_c  & ~target_c).sum().item()
            self._fn[c] += (~pred_c & target_c).sum().item()
            self._tn[c] += (~pred_c & ~target_c).sum().item()

    def compute(self) -> Dict[str, Any]:
        """Tüm metrikleri hesaplar ve döner."""
        # DÜZELTME: Önceden `aggregate(reduction="none")` çağrılıp sonuç [0]
        # ile indeksleniyordu. get_not_nans=True olduğundan [0] tuple'dan
        # "data" bileşenini seçer — fakat reduction="none" bu veriyi
        # (örnek_sayısı, sınıf_sayısı) şeklinde HİÇ İNDİRGENMEMİŞ bırakır.
        # Sonucunda class_{c}_dice değeri, tüm test setinin ortalaması yerine
        # yalnızca İLK test örneğinin skorunu (çok sınıflı durumda ise
        # float()'a sığmayan bir diziyi, yani çökme hatasını) veriyordu.
        # Reduction override kaldırılarak constructor'daki "mean_batch"
        # kullanılıyor → doğru şekilde sınıf başına indirgenmiş (C,) vektörü.
        dice_vals = self.dice_metric.aggregate()[0].numpy()
        iou_vals  = self.iou_metric.aggregate()[0].numpy()
        hd_vals   = self.hd_metric.aggregate()[0].numpy()

        results: Dict[str, Any] = {
            "mean_dice":  float(np.nanmean(dice_vals)),
            "mean_iou":   float(np.nanmean(iou_vals)),
            "mean_hd95":  float(np.nanmean(hd_vals)),
        }

        # Sınıf bazlı metrikler
        start_c = 0 if self.include_background else 1
        for c in range(start_c, self.num_classes):
            tp, fp, fn, tn = self._tp[c], self._fp[c], self._fn[c], self._tn[c]
            eps    = 1e-8
            sens   = tp / (tp + fn + eps)    # Sensitivite = Recall
            spec   = tn / (tn + fp + eps)    # Özgüllük = Specificity
            prec   = tp / (tp + fp + eps)    # Kesinlik = Precision
            f1     = 2 * prec * sens / (prec + sens + eps)
            vol_sim = 1 - abs(tp + fn - tp - fp) / (2 * tp + fp + fn + eps)

            idx = c - start_c
            results[f"class_{c}_dice"]        = float(dice_vals[idx]) if idx < len(dice_vals) else 0.0
            results[f"class_{c}_iou"]         = float(iou_vals[idx])  if idx < len(iou_vals)  else 0.0
            results[f"class_{c}_hd95"]        = float(hd_vals[idx])  if idx < len(hd_vals)  else 0.0
            results[f"class_{c}_sensitivity"] = float(sens)
            results[f"class_{c}_specificity"] = float(spec)
            results[f"class_{c}_precision"]   = float(prec)
            results[f"class_{c}_f1"]          = float(f1)
            results[f"class_{c}_vol_sim"]     = float(vol_sim)

        return results


# =============================================================================
# 2. Sınıflandırma Metrikleri
# =============================================================================

class ClassificationEvaluator:
    """
    Sınıflandırma görevi için kapsamlı metrik paketi.
    """

    def __init__(
        self,
        num_classes:  int,
        class_names:  Optional[List[str]] = None,
        multi_label:  bool = False,
    ) -> None:
        self.num_classes  = num_classes
        self.class_names  = class_names or [f"Sınıf {i}" for i in range(num_classes)]
        self.multi_label  = multi_label
        self.all_probs:   List[np.ndarray] = []
        self.all_labels:  List[np.ndarray] = []

    def update(self, probs: Tensor, labels: Tensor) -> None:
        # Çok-etiketli: sınıflar bağımsız → sigmoid. Tek-etiketli: sınıflar
        # birbirini dışlar → softmax.
        act = torch.sigmoid if self.multi_label else (lambda x: torch.softmax(x, dim=1))
        self.all_probs.append(act(probs.float()).cpu().numpy())
        self.all_labels.append(labels.cpu().numpy())

    def _compute_multilabel(self) -> Dict[str, Any]:
        """
        Sınıf başına AUC-ROC ve ortalama hassasiyet.
        Çok-etiketli kurulumda accuracy ve confusion matrix anlamlı değildir;
        seyrek pozitifli sınıflarda hep-negatif tahmin %99 doğruluk verir.
        """
        probs  = np.concatenate(self.all_probs,  axis=0)    # (N, C)
        labels = np.concatenate(self.all_labels, axis=0)    # (N, C)

        per_class: Dict[str, Dict[str, float]] = {}
        aucs, aps = [], []
        for c, name in enumerate(self.class_names[:self.num_classes]):
            n_pos = int(labels[:, c].sum())
            if not 0 < n_pos < len(labels):
                per_class[name] = {"auc": float("nan"), "ap": float("nan"), "n_pos": n_pos}
                continue
            auc = float(roc_auc_score(labels[:, c], probs[:, c]))
            ap  = float(average_precision_score(labels[:, c], probs[:, c]))
            per_class[name] = {"auc": auc, "ap": ap, "n_pos": n_pos}
            aucs.append(auc)
            aps.append(ap)

        return {
            "auc_macro":     float(np.mean(aucs)) if aucs else float("nan"),
            "ap_macro":      float(np.mean(aps))  if aps  else float("nan"),
            "f1_macro":      float(f1_score(labels, (probs >= 0.5).astype(int),
                                            average="macro", zero_division=0)),
            "per_class":     per_class,
            "n_samples":     int(len(labels)),
            "_labels":       labels,
            "_probs":        probs,
        }

    def compute(self) -> Dict[str, Any]:
        if self.multi_label:
            return self._compute_multilabel()

        probs   = np.concatenate(self.all_probs,  axis=0)   # (N, C)
        labels  = np.concatenate(self.all_labels, axis=0)   # (N,)
        preds   = probs.argmax(axis=1)

        results: Dict[str, Any] = {
            "accuracy":  float((preds == labels).mean()),
            "mcc":       float(matthews_corrcoef(labels, preds)),
            "f1_macro":  float(f1_score(labels, preds, average="macro")),
            "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        }

        # AUC-ROC (çok sınıflı)
        try:
            if self.num_classes == 2:
                results["auc_roc"] = float(roc_auc_score(labels, probs[:, 1]))
            else:
                results["auc_roc"] = float(
                    roc_auc_score(labels, probs, multi_class="ovr", average="macro")
                )
        except Exception:
            results["auc_roc"] = float("nan")

        # Ortalama Hassasiyet (PR eğrisi altındaki alan)
        try:
            results["avg_precision"] = float(
                average_precision_score(
                    np.eye(self.num_classes)[labels], probs, average="macro"
                )
            )
        except Exception:
            results["avg_precision"] = float("nan")

        # Brier Skoru (olasılık kalibrasyonu)
        try:
            results["brier_score"] = float(
                brier_score_loss(labels == 1, probs[:, 1])
            ) if self.num_classes == 2 else float("nan")
        except Exception:
            results["brier_score"] = float("nan")

        results["report"]      = classification_report(
            labels, preds, target_names=self.class_names, output_dict=True
        )
        results["_probs"]      = probs
        results["_labels"]     = labels
        results["_preds"]      = preds
        return results


# =============================================================================
# 3. Görselleştirme
# =============================================================================

class EvaluationVisualizer:
    """
    Tüm değerlendirme grafiklerini oluşturur ve kaydeder.
    """

    PALETTE = sns.color_palette("husl", 12)

    def __init__(self, output_dir: Path) -> None:
        self.out = output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")

    def plot_confusion_matrix(
        self,
        labels:      np.ndarray,
        preds:       np.ndarray,
        class_names: List[str],
        normalize:   bool = True,
    ) -> None:
        """Normalize edilmiş ve normalize edilmemiş karmaşıklık matrisleri."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Karmaşıklık Matrisi (Confusion Matrix)", fontsize=13, fontweight="bold")

        for ax, norm, title in zip(
            axes,
            [None, "true"],
            ["Ham Sayılar", "Normalleştirilmiş (Satır %)"],
        ):
            cm = confusion_matrix(labels, preds, normalize=norm)
            fmt = ".0f" if norm is None else ".2%"
            sns.heatmap(
                cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, square=True,
            )
            ax.set_title(title)
            ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")

        self._save(fig, "confusion_matrix.png")

    def plot_roc_curves(
        self,
        labels: np.ndarray,
        probs:  np.ndarray,
        class_names: List[str],
    ) -> None:
        """Her sınıf için ROC eğrisi + mikro/makro ortalamalar."""
        n = probs.shape[1]
        labels_bin = np.eye(n)[labels]

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Rastgele (AUC=0.50)")

        for i, name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            auc         = roc_auc_score(labels_bin[:, i], probs[:, i])
            ax.plot(fpr, tpr, lw=2, color=self.PALETTE[i],
                    label=f"{name} (AUC={auc:.3f})")

        ax.set_xlabel("Yanlış Pozitif Oranı (FPR)")
        ax.set_ylabel("Doğru Pozitif Oranı (TPR)")
        ax.set_title("ROC Eğrileri — Sınıf Bazlı", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        self._save(fig, "roc_curves.png")

    def plot_precision_recall_curves(
        self,
        labels: np.ndarray,
        probs:  np.ndarray,
        class_names: List[str],
    ) -> None:
        """PR eğrisi — dengesiz veri setlerinde ROC'dan daha bilgilendirici."""
        n = probs.shape[1]
        labels_bin = np.eye(n)[labels]

        fig, ax = plt.subplots(figsize=(9, 8))
        for i, name in enumerate(class_names):
            prec, rec, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
            ap           = average_precision_score(labels_bin[:, i], probs[:, i])
            ax.plot(rec, prec, lw=2, color=self.PALETTE[i],
                    label=f"{name} (AP={ap:.3f})")

        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Eğrileri", fontsize=12, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        self._save(fig, "pr_curves.png")

    def plot_calibration_curve(
        self,
        labels: np.ndarray,
        probs:  np.ndarray,
        n_bins: int = 10,
    ) -> None:
        """
        Güvenilirlik diyagramı (Calibration Curve).
        Modelin tahmin güveni ile gerçek doğruluk oranını karşılaştırır.
        Eğri köşegen üzerindeyse model iyi kalibre edilmiştir.
        """
        if probs.shape[1] != 2:
            logger.info("Kalibrasyon eğrisi: sadece ikili sınıflandırma için.")
            return

        frac_pos, mean_pred = calibration_curve(labels, probs[:, 1], n_bins=n_bins)
        brier = brier_score_loss(labels, probs[:, 1])

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, 1], [0, 1], "k--", label="Mükemmel Kalibrasyon")
        ax.plot(mean_pred, frac_pos, "s-", color=self.PALETTE[0],
                lw=2, label=f"Model (Brier={brier:.4f})")
        ax.set_xlabel("Ortalama Tahmin Güveni")
        ax.set_ylabel("Gerçek Pozitif Oranı")
        ax.set_title("Güvenilirlik Diyagramı (Kalibrasyon)", fontsize=12, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        self._save(fig, "calibration_curve.png")

    def plot_segmentation_metrics(
        self,
        metrics:     Dict[str, float],
        num_classes: int,
    ) -> None:
        """Sınıf bazlı segmentasyon metrik çubuklarını yan yana gösterir."""
        metric_names = ["dice", "iou", "sensitivity", "specificity", "precision", "f1"]
        classes      = list(range(1, num_classes))

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Segmentasyon Metrikleri — Sınıf Bazlı", fontsize=13, fontweight="bold")

        for ax, m in zip(axes.flat, metric_names):
            vals = [metrics.get(f"class_{c}_{m}", 0.0) for c in classes]
            ax.bar([f"Sınıf {c}" for c in classes], vals,
                   color=self.PALETTE[:len(classes)])
            ax.set_ylim(0, 1)
            ax.set_title(m.capitalize())
            ax.set_ylabel("Değer")
            for i, v in enumerate(vals):
                ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

        plt.tight_layout()
        self._save(fig, "segmentation_metrics.png")

    def plot_prediction_samples(
        self,
        model:   nn.Module,
        loader:  DataLoader,
        cfg:     Any,
        n:       int = 6,
    ) -> None:
        """
        Görüntü / Tahmin / Gerçek Maske / Fark Haritası yan yana.
        Modelin güçlü ve zayıf olduğu örnekleri görmek için kullanılır.
        """
        model.eval()
        fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
        fig.suptitle("Örnek Tahminler", fontsize=13, fontweight="bold")
        headers = ["Giriş Görüntüsü", "Tahmin", "Gerçek Maske", "Fark Haritası"]
        for ax, h in zip(axes[0], headers):
            ax.set_title(h, fontweight="bold")

        batch = next(iter(loader))
        imgs  = batch["image"].to(cfg.DEVICE)
        lbls  = batch["label"]
        with torch.no_grad(), autocast(cfg.DEVICE.type, enabled=cfg.USE_AMP):
            outs = sliding_window_inference(imgs, cfg.IMAGE_SIZE, 1, model, cfg.SLIDING_OVERLAP)
            pred_masks = torch.argmax(torch.softmax(outs, dim=1), dim=1).cpu().numpy()

        imgs_np = imgs.cpu().numpy()
        lbls_np = lbls.numpy().squeeze(1) if lbls.ndim == 4 else lbls.numpy()

        for i in range(min(n, len(imgs_np))):
            img  = imgs_np[i, 0]
            pred = pred_masks[i]
            lbl  = lbls_np[i]
            diff = (pred != lbl).astype(float)

            axes[i, 0].imshow(img, cmap="gray"); axes[i, 0].axis("off")
            axes[i, 1].imshow(pred, cmap="jet", vmin=0, vmax=cfg.NUM_CLASSES - 1); axes[i, 1].axis("off")
            axes[i, 2].imshow(lbl,  cmap="jet", vmin=0, vmax=cfg.NUM_CLASSES - 1); axes[i, 2].axis("off")
            im = axes[i, 3].imshow(diff, cmap="Reds", vmin=0, vmax=1); axes[i, 3].axis("off")
            plt.colorbar(im, ax=axes[i, 3], fraction=0.046)

        plt.tight_layout()
        self._save(fig, "prediction_samples.png")

    def plot_multilabel_auc(self, per_class: Dict[str, Dict[str, float]]) -> None:
        """
        Sınıf başına AUC — pozitif örnek sayısına göre sıralı yatay çubuk grafik.
        Seyrek sınıflardaki güvenilmez AUC'leri görünür kılar.
        """
        rows = [(n, d["auc"], d["n_pos"]) for n, d in per_class.items()
                if not np.isnan(d["auc"])]
        if not rows:
            logger.warning("Çizilecek geçerli AUC yok.")
            return
        rows.sort(key=lambda r: r[1])
        names, aucs, n_pos = zip(*rows)

        fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 2))
        bars = ax.barh(names, aucs, color="#2D6A9F")
        ax.axvline(0.5, color="crimson", ls="--", lw=1, label="Rastgele (0.50)")
        ax.set_xlim(0.4, 1.0)
        ax.set_xlabel("AUC-ROC")
        ax.set_title("Sınıf Başına AUC-ROC")
        for bar, a, n in zip(bars, aucs, n_pos):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{a:.3f}  (n={n})", va="center", fontsize=8)
        ax.legend(loc="lower right")
        self._save(fig, "per_class_auc.png")

    def plot_roc_curves_multilabel(
        self,
        labels: np.ndarray,   # (N, C) 0/1
        probs:  np.ndarray,   # (N, C) olasılık
        class_names: List[str],
    ) -> None:
        """Çok-etiketli kurulumda her sınıf için ayrı ROC eğrisi."""
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Rastgele (AUC=0.50)")

        for i, name in enumerate(class_names[:probs.shape[1]]):
            if not 0 < labels[:, i].sum() < len(labels):
                continue
            fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
            ax.plot(fpr, tpr, lw=1.4,
                    label=f"{name} ({roc_auc_score(labels[:, i], probs[:, i]):.3f})")

        ax.set_xlabel("Yanlış Pozitif Oranı")
        ax.set_ylabel("Doğru Pozitif Oranı")
        ax.set_title("ROC Eğrileri — Sınıf Başına")
        ax.legend(loc="lower right", fontsize=7)
        self._save(fig, "roc_curves.png")

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self.out / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"  ✓ {filename}")


# =============================================================================
# 4. Grad-CAM — Açıklanabilir Yapay Zeka (XAI)
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Modelin hangi bölgelere odaklandığını ısı haritasıyla gösterir.
    Medikal AI'da güven ve şeffaflık için kritiktir.

    Referans: Selvaraju et al., ICCV 2017.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer
        self.gradients:   Optional[Tensor] = None
        self.activations: Optional[Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(
        self,
        input_tensor: Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Giriş tensörü için Grad-CAM ısı haritası üretir.

        Returns:
            (H, W) boyutlu normalize edilmiş ısı haritası [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Seçilen sınıfa göre gradient hesapla
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Kanal ağırlıkları (global average pooling)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze(0)
        cam     = F.relu(cam).cpu().numpy()

        # Normalize ve yeniden boyutlandır
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        h, w = input_tensor.shape[-2:]
        cam  = np.array(
            __import__("PIL").Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h))
        ) / 255.0
        return cam

    def visualize_batch(
        self,
        images:      Tensor,
        labels:      Tensor,
        output_path: Path,
        class_names: List[str],
        n:           int = 6,
    ) -> None:
        """Batch'ten n görüntü için Grad-CAM ısı haritası oluşturur."""
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
        fig.suptitle("Grad-CAM — Model Odak Bölgeleri", fontsize=13, fontweight="bold")
        headers = ["Giriş", "Grad-CAM Isı Haritası", "Katmanlı"]
        for ax, h in zip(axes[0], headers):
            ax.set_title(h, fontweight="bold")

        import matplotlib.cm as cm
        colormap = cm.get_cmap("jet")

        for i in range(min(n, images.shape[0])):
            img_tensor = images[i:i+1]
            cam        = self.generate(img_tensor)
            img_np     = images[i, 0].cpu().numpy()

            # Görüntüyü 3 kanala genişlet (katmanlı görselleştirme için)
            img_rgb    = np.stack([img_np] * 3, axis=-1)
            img_rgb    = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
            cam_rgb    = colormap(cam)[:, :, :3]
            overlay    = 0.5 * img_rgb + 0.5 * cam_rgb

            true_cls  = labels[i].item() if labels.ndim == 1 else labels[i, 0].item()
            axes[i, 0].imshow(img_np, cmap="gray"); axes[i, 0].axis("off")
            axes[i, 0].set_title(f"Gerçek: {class_names[true_cls]}", fontsize=8)
            axes[i, 1].imshow(cam, cmap="jet"); axes[i, 1].axis("off")
            axes[i, 2].imshow(overlay); axes[i, 2].axis("off")

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Grad-CAM kaydedildi → {output_path}")


# =============================================================================
# 5. Model Export (ONNX / TorchScript)
# =============================================================================

def export_to_onnx(
    model:      nn.Module,
    cfg:        Any,
    output_dir: Path,
) -> Path:
    """
    Modeli Open Neural Network Exchange (ONNX) formatına aktarır.
    ONNX, çerçeve bağımsız (framework-agnostic) çıkarım (inference) ve
    üretim ortamı dağıtımı için IEEE/ISO tarafından desteklenen açık standarttır.
    Statik hesap grafiği optimizasyonu (constant folding) etkinleştirilmiştir.
    """
    model.eval()
    h, w    = cfg.IMAGE_SIZE
    dummy   = torch.randn(1, cfg.IN_CHANNELS, h, w, device=cfg.DEVICE)
    onnx_path = output_dir / f"{cfg.EXP_NAME}.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=cfg.ONNX_OPSET,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        verbose=False,
    )

    # Doğrulama
    import onnx
    model_onnx = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_onnx)
    logger.info(f"ONNX export doğrulandı → {onnx_path} "
                f"({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return onnx_path


def export_to_torchscript(
    model:      nn.Module,
    cfg:        Any,
    output_dir: Path,
) -> Path:
    """
    Modeli TorchScript formatına aktarır (C++ deployment için).
    """
    model.eval()
    h, w   = cfg.IMAGE_SIZE
    dummy  = torch.randn(1, cfg.IN_CHANNELS, h, w, device=cfg.DEVICE)
    ts_path = output_dir / f"{cfg.EXP_NAME}.pt"

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    traced.save(str(ts_path))
    logger.info(f"TorchScript export → {ts_path}")
    return ts_path


# =============================================================================
# 6. Kapsamlı Test Değerlendirmesi
# =============================================================================

@torch.no_grad()
def full_evaluation(
    model:      nn.Module,
    loader:     DataLoader,
    cfg:        Any,
    output_dir: Path,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Test seti üzerinde tüm metrikleri hesaplar ve tüm grafikleri çizer.
    """
    class_names = class_names or [f"Sınıf {i}" for i in range(cfg.NUM_CLASSES)]
    visualizer  = EvaluationVisualizer(output_dir)

    if cfg.TASK == "segmentation":
        evaluator = SegmentationEvaluator(cfg.NUM_CLASSES)
    else:
        evaluator = ClassificationEvaluator(
            cfg.NUM_CLASSES, class_names,
            multi_label=getattr(cfg, "MULTI_LABEL", False),
        )

    model.eval()
    for batch in loader:
        images = batch["image"].to(cfg.DEVICE, non_blocking=True)
        labels = batch["label"].to(cfg.DEVICE, non_blocking=True)

        with autocast(cfg.DEVICE.type, enabled=cfg.USE_AMP):
            if cfg.TASK == "segmentation":
                preds = sliding_window_inference(
                    images, cfg.IMAGE_SIZE, 1, model, cfg.SLIDING_OVERLAP
                )
                preds = torch.softmax(preds, dim=1)
            else:
                preds = model(images)

        evaluator.update(preds.cpu(), labels.cpu())
        del images, labels, preds
        torch.cuda.empty_cache()

    metrics = evaluator.compute()

    # ── Grafikleri Oluştur ────────────────────────────────────────────────────
    if cfg.TASK == "segmentation":
        visualizer.plot_segmentation_metrics(metrics, cfg.NUM_CLASSES)
        visualizer.plot_prediction_samples(model, loader, cfg)
    elif getattr(cfg, "MULTI_LABEL", False):
        labels_arr = metrics.pop("_labels")
        probs_arr  = metrics.pop("_probs")
        per_class  = metrics["per_class"]

        # Sınıf başına AUC tablosu — raporlanabilir asıl çıktı
        pd.DataFrame(per_class).T.sort_values("auc", ascending=False).to_csv(
            output_dir / "per_class_metrics.csv"
        )
        visualizer.plot_multilabel_auc(per_class)
        visualizer.plot_roc_curves_multilabel(labels_arr, probs_arr, class_names)

    else:
        labels_arr = metrics.pop("_labels")
        probs_arr  = metrics.pop("_probs")
        preds_arr  = metrics.pop("_preds")
        _report    = metrics.pop("report", None)

        visualizer.plot_confusion_matrix(labels_arr, preds_arr, class_names)
        visualizer.plot_roc_curves(labels_arr, probs_arr, class_names)
        visualizer.plot_precision_recall_curves(labels_arr, probs_arr, class_names)
        visualizer.plot_calibration_curve(labels_arr, probs_arr)

        if _report:
            pd.DataFrame(_report).T.to_csv(output_dir / "classification_report.csv")

    # ── Metrikleri Kaydet ─────────────────────────────────────────────────────
    with open(output_dir / "test_metrics.json", "w") as f:
        def _jsonable(v):
            if isinstance(v, (int, float, np.floating, np.integer)):
                return float(v)
            if isinstance(v, dict):
                return {kk: _jsonable(vv) for kk, vv in v.items()}
            return str(v)
        json_metrics = {k: _jsonable(v) for k, v in metrics.items()}
        import json; json.dump(json_metrics, f, indent=2)

    # ── ONNX Export ───────────────────────────────────────────────────────────
    if cfg.EXPORT_ONNX:
        try:
            export_to_onnx(model, cfg, output_dir)
        except Exception as e:
            logger.warning(f"ONNX export başarısız: {e}")

    if cfg.EXPORT_TORCHSCRIPT:
        try:
            export_to_torchscript(model, cfg, output_dir)
        except Exception as e:
            logger.warning(f"TorchScript export başarısız: {e}")

    logger.info("Tam değerlendirme tamamlandı.")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:40s} = {v:.4f}")

    return metrics

import json  # noqa: E402 — dosya üstüne taşınamaz (metrics içinde kullanılıyor)
