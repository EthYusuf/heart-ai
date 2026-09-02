# =============================================================================
# trainer.py — Profesyonel Eğitim Motoru
# -----------------------------------------------------------------------------
# İçerik:
#   - Otomatik Karma Hassasiyet (AMP / FP16)
#   - SAM Optimizer desteği (Sharpness-Aware Minimization)
#   - K-Fold Çapraz Doğrulama
#   - Erken Durdurma (Early Stopping) — delta ve sabır tabanlı
#   - Isınma + Kosinüs tavlama LR scheduler
#   - Ağırlıklı örnekleme (sınıf dengesizliği)
#   - MixUp / CutMix artırma (eğitim içi)
#   - Gradient Checkpointing
#   - Weights & Biases deney izleme entegrasyonu (isteğe bağlı)
# =============================================================================

import gc
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
# DÜZELTME: torch.cuda.amp.{GradScaler,autocast} PyTorch 2.4+'ta kullanımdan
# kaldırıldı (yeni PyTorch sürümlerinde tamamen kaldırılma riski taşıyordu);
# cihaz-bağımsız torch.amp API'sine geçirildi.
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, KFold
from monai.inferers import sliding_window_inference

from datasets import simple_collate
from models import get_layer_wise_lr_params

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =============================================================================
# 1. Erken Durdurma
# =============================================================================

class EarlyStopping:
    """
    Validasyon metriği artmayı durdurunca eğitimi keser.
    """

    def __init__(
        self,
        patience:  int   = 15,
        min_delta: float = 1e-4,
        mode:      str   = "max",   # "max" = metrik artmalı, "min" = azalmalı
        verbose:   bool  = True,
    ) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.verbose   = verbose
        self.counter   = 0
        self.best      = -np.inf if mode == "max" else np.inf
        self.stop      = False

    def __call__(self, metric: float) -> bool:
        improved = (
            (self.mode == "max" and metric > self.best + self.min_delta) or
            (self.mode == "min" and metric < self.best - self.min_delta)
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping: sabır={self.counter}/{self.patience} "
                            f"(en iyi={self.best:.4f})")
            if self.counter >= self.patience:
                logger.info("Erken durdurma tetiklendi.")
                self.stop = True
        return self.stop


# =============================================================================
# 2. Isınma LR Scheduler Sarmalayıcı
# =============================================================================

class WarmupScheduler:
    """
    Lineer ısınma + ana scheduler kombinasyonu.
    İlk warmup_epochs epoch boyunca LR sıfırdan base_lr'e lineer artar,
    ardından ana scheduler devreye girer.
    """

    def __init__(
        self,
        optimizer:      torch.optim.Optimizer,
        warmup_epochs:  int,
        main_scheduler: Any,
        base_lr:        float,
        per_batch:      bool = False,
        is_plateau:     bool = False,
    ) -> None:
        """
        per_batch  : main_scheduler'ın epoch başına değil, HER BATCH'TE bir
                     adım atması gerektiğinde True (yalnızca OneCycleLR).
                     DÜZELTME: OneCycleLR, `epochs * steps_per_epoch` toplam
                     adım üzerine kurulur (scheduler_factory'de böyle
                     yapılandırılıyor) ama önceki sürümde `.step()` yalnızca
                     epoch sonunda çağrılıyordu — yani OneCycleLR, toplam
                     bütçesinin yalnızca ~%1'i kadar ilerleyip döngüsünü hiç
                     tamamlayamıyor, LR neredeyse hep başlangıç (ısınma)
                     değerinde takılı kalıyordu. OneCycleLR kendi dahili
                     ısınmasını (pct_start) zaten uyguladığından, per_batch=True
                     iken bu sınıfın kendi lineer ısınması devre dışı bırakılır
                     (çifte ısınmayı önlemek için).
        is_plateau : main_scheduler bir ReduceLROnPlateau ise True.
                     DÜZELTME: ReduceLROnPlateau.step() bir metrik değeri
                     ZORUNLU ister; önceki sürümde argümansız `.step()`
                     çağrıldığından SCHEDULER="plateau" seçildiğinde her
                     epoch sonunda `TypeError` ile çöküyordu.
        """
        self.optimizer       = optimizer
        self.warmup_epochs   = warmup_epochs
        self.main_scheduler  = main_scheduler
        self.base_lr         = base_lr
        self.per_batch        = per_batch
        self.is_plateau       = is_plateau
        self.current_epoch   = 0

    def step(self, metric: Optional[float] = None) -> None:
        """Her epoch sonunda bir kez çağrılır."""
        if self.per_batch:
            # OneCycleLR zaten step_batch() ile her iterasyonda ilerletildi;
            # burada yalnızca epoch sayacı güncellenir.
            pass
        elif self.current_epoch < self.warmup_epochs:
            progress = (self.current_epoch + 1) / self.warmup_epochs
            lr = self.base_lr * progress
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        elif self.is_plateau:
            if metric is None:
                raise ValueError(
                    "ReduceLROnPlateau scheduler'ı için step(metric=...) "
                    "zorunludur — validasyon metriği geçirilmedi."
                )
            self.main_scheduler.step(metric)
        else:
            self.main_scheduler.step()
        self.current_epoch += 1

    def step_batch(self) -> None:
        """Her eğitim iterasyonundan sonra çağrılır (yalnızca per_batch=True
        olduğunda, yani OneCycleLR için, etkilidir)."""
        if self.per_batch:
            self.main_scheduler.step()

    def get_last_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# =============================================================================
# 3. SAM Optimizer (Sharpness-Aware Minimization)
# =============================================================================

class SAM(torch.optim.Optimizer):
    """
    SAM: Keskin olmayan minimum noktalara yönlendirerek genellemeyi artırır.
    Her batch'te iki ileri + geri geçiş gerektirir (yaklaşık 2x hesaplama).
    Referans: Foret et al., ICLR 2021.
    """

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        defaults = {"rho": rho, **kwargs}
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        scale     = self.defaults["rho"] / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)                 # ε-pertürbasyon ekle
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # Pertürbasyonu geri al

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> Tensor:
        norms = [
            p.grad.norm(2.0)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.stack(norms).norm(2.0)


# =============================================================================
# 4. MixUp / CutMix Artırma (Eğitim İçi)
# =============================================================================

def mixup_data(
    x: Tensor, y: Tensor, alpha: float = 0.4
) -> Tuple[Tensor, Tensor, Tensor, float]:
    """
    MixUp: iki örneği lambda ağırlıkla karıştırır.
    Aşırı öğrenmeyi azaltır, kalibrasyon kalitesini artırır.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x  = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: Callable,
    preds: Tensor, y_a: Tensor, y_b: Tensor, lam: float,
) -> Tensor:
    """MixUp uygulandıktan sonra ağırlıklı kayıp hesabı."""
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


# =============================================================================
# 5. Ağırlıklı Örnekleyici (Sınıf Dengesizliği)
# =============================================================================

def build_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Her sınıfın eğitim verisi içinde eşit temsil edilmesini sağlar.
    Nadir hastalık sınıflarında kritik önem taşır.
    """
    counts  = np.bincount(labels)
    weights = 1.0 / counts
    sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    logger.info(f"Ağırlıklı örnekleyici: sınıf sayıları={dict(enumerate(counts))}")
    return sampler


# =============================================================================
# 6. Optimizer Fabrikası
# =============================================================================

def optimizer_factory(
    name:   str,
    params: Any,
    lr:     float,
    weight_decay: float = 1e-5,
    momentum: float = 0.9,
    rho:    float = 0.05,
) -> torch.optim.Optimizer:
    """AdamW, SAM, Lion, SGD optimizer seçimi."""
    name_lower = name.lower()
    if name_lower == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name_lower == "sam":
        return SAM(params, torch.optim.AdamW, rho=rho, lr=lr, weight_decay=weight_decay)
    elif name_lower == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum,
                               weight_decay=weight_decay, nesterov=True)
    elif name_lower == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            logger.warning("lion_pytorch bulunamadı → AdamW kullanılıyor.")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Bilinmeyen optimizer: {name}")


def scheduler_factory(
    name:        str,
    optimizer:   torch.optim.Optimizer,
    epochs:      int,
    steps_per_epoch: int = 1,
    T0:          int   = 10,
    T_mult:      int   = 2,
    min_lr:      float = 1e-7,
) -> Any:
    """LR scheduler fabrikası."""
    if name == "cosine_warm":
        return CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=T_mult, eta_min=min_lr)
    elif name == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=[pg["lr"] for pg in optimizer.param_groups],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1e4,
        )
    elif name == "poly":
        return PolynomialLR(optimizer, total_iters=epochs, power=0.9)
    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5,
                                 min_lr=min_lr, verbose=True)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        raise ValueError(f"Bilinmeyen scheduler: {name}")


# =============================================================================
# 7. Eğitim / Validasyon Adımları
# =============================================================================

def _grads_finite(model: nn.Module) -> bool:
    """SAM'in elle çağırdığı optimizer adımlarından önce gradyanların
    inf/NaN içermediğini doğrular (bkz. train_epoch SAM dalındaki not)."""
    return all(
        torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.grad is not None
    )


def train_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    scaler:     GradScaler,
    cfg:        Any,
    epoch:      int,
    use_mixup:  bool = False,
    scheduler:  Optional["WarmupScheduler"] = None,
) -> Dict[str, float]:
    """
    Tek epoch eğitimi.

    Dönüş: {"loss": float, "lr": float, "time_s": float}
    """
    model.train()
    running_loss = 0.0
    t_start      = time.time()
    use_sam      = isinstance(optimizer, SAM)

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(cfg.DEVICE, non_blocking=True)
        labels = batch["label"].to(cfg.DEVICE, non_blocking=True)

        # ── MixUp Artırma (sadece sınıflandırma görevi) ───────────────────────
        if use_mixup and cfg.TASK == "classification":
            images, y_a, y_b, lam = mixup_data(images, labels, alpha=cfg.MIXUP_ALPHA)

        # ── SAM: İki Geçişli Güncelleme ───────────────────────────────────────
        if use_sam:
            with autocast(cfg.DEVICE.type, enabled=cfg.USE_AMP):
                out  = model(images)
                # DÜZELTME: SAM dalı önceden MixUp karışık görüntülerle
                # (images) üretilen çıktıyı orijinal, karıştırılmamış
                # `labels` ile karşılaştırıyordu — mixup_data zaten
                # `images`'i karıştırdığı için hedef etiketler artık
                # girdiyle uyumsuzdu. Standart daldaki gibi
                # mixup_criterion kullanılarak düzeltildi.
                loss = mixup_criterion(criterion, out, y_a, y_b, lam) \
                       if (use_mixup and cfg.TASK == "classification") \
                       else criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            # DÜZELTME: scaler.step() kullanılmadan optimizer.first_step /
            # second_step doğrudan çağrıldığından, GradScaler'ın normalde
            # sağladığı "inf/nan gradyanlıysa adımı atla" koruması devre dışı
            # kalıyordu (özellikle AMP ölçek faktörünün kalibre olduğu ilk
            # birkaç iterasyonda ağırlıkların NaN'a düşme riski). Adımdan
            # önce basit bir sonluluk kontrolü ekleniyor.
            if _grads_finite(model):
                optimizer.first_step(zero_grad=True)
            else:
                optimizer.zero_grad(set_to_none=True)

            with autocast(cfg.DEVICE.type, enabled=cfg.USE_AMP):
                out2  = model(images)
                loss2 = mixup_criterion(criterion, out2, y_a, y_b, lam) \
                        if (use_mixup and cfg.TASK == "classification") \
                        else criterion(out2, labels)
            scaler.scale(loss2).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            if _grads_finite(model):
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad(set_to_none=True)
            scaler.update()

        # ── Standart Güncelleme ───────────────────────────────────────────────
        else:
            optimizer.zero_grad(set_to_none=True)
            with autocast(cfg.DEVICE.type, enabled=cfg.USE_AMP):
                out  = model(images)
                if use_mixup and cfg.TASK == "classification":
                    loss = mixup_criterion(criterion, out, y_a, y_b, lam)
                else:
                    loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

        # DÜZELTME: OneCycleLR her iterasyonda (epoch başına değil)
        # ilerletilmeli — bkz. WarmupScheduler.step_batch() docstring'i.
        # Diğer scheduler'lar için (per_batch=False) bu çağrı no-op'tur.
        if scheduler is not None:
            scheduler.step_batch()

        running_loss += loss.item()

        if (batch_idx + 1) % cfg.LOG_INTERVAL == 0:
            mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            logger.info(
                f"  Epoch [{epoch+1:03d}] Batch [{batch_idx+1:04d}/{len(loader)}] "
                f"Kayıp: {loss.item():.4f} | VRAM: {mem:.2f} GB"
            )

        # Bellek temizleme
        del images, labels, out, loss
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - t_start
    lr      = optimizer.param_groups[0]["lr"]
    return {"loss": running_loss / len(loader), "lr": lr, "time_s": elapsed}


@torch.no_grad()
def validate_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    metrics_fn: Callable,
    cfg:       Any,
    use_tta:   bool = False,
) -> Dict[str, float]:
    """
    Validasyon adımı. use_tta=True ise Test Time Augmentation uygulanır.

    Dönüş: {"loss": float, "metric": float, "time_s": float}
    """
    model.eval()
    running_loss = 0.0
    all_preds    = []
    all_labels   = []
    t_start      = time.time()

    for batch in loader:
        images = batch["image"].to(cfg.DEVICE, non_blocking=True)
        labels = batch["label"].to(cfg.DEVICE, non_blocking=True)

        with autocast(cfg.DEVICE.type, enabled=cfg.USE_AMP):
            if cfg.TASK == "segmentation":
                predictor = lambda x: sliding_window_inference(
                    inputs=x,
                    roi_size=cfg.IMAGE_SIZE,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=cfg.SLIDING_OVERLAP,
                )
                # DÜZELTME: use_tta bayrağı segmentasyon dalında hiç
                # kontrol edilmiyordu; cfg.USE_TTA=True olsa bile
                # segmentasyon validasyonu TTA'sız çalışıyordu (README'nin
                # vaat ettiği "8 geometrik dönüşüm" yalnızca sınıflandırmada
                # devredeydi).
                preds = _apply_tta(predictor, images, cfg.TTA_STEPS) if use_tta \
                        else predictor(images)
            else:
                predictor = model
                preds = _apply_tta(predictor, images, cfg.TTA_STEPS) if use_tta \
                        else model(images)

            loss = criterion(preds, labels)

        running_loss += loss.item()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        del images, labels, preds, loss
        torch.cuda.empty_cache()

    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metric     = metrics_fn(all_preds, all_labels)
    elapsed    = time.time() - t_start

    return {"loss": running_loss / len(loader), "metric": metric, "time_s": elapsed}


def _apply_tta(
    predictor: Callable[[Tensor], Tensor],
    images:    Tensor,
    n_steps:   int = 8,
) -> Tensor:
    """
    Test Time Augmentation:
    Flip (H/V) + 90° dönüş kombinasyonlarıyla birden fazla tahmin alır,
    ham logit (softmax öncesi) ortalamasıyla nihai tahmini üretir.

    DÜZELTME: Önceki sürüm her görünüm için softmax uygulayıp olasılıkları
    ortalıyordu ve bunu doğrudan `criterion`'a (CrossEntropy/BCEWithLogits —
    ham logit bekler) geçiriyordu. Bu, TTA aktif olduğu son epoch'larda
    (early stopping / en iyi checkpoint kararını etkileyen dönemde) validasyon
    kaybının ve metriklerin çifte-softmax nedeniyle yanlış hesaplanmasına yol
    açıyordu. Artık ortalama logit uzayında alınıyor; softmax/sigmoid
    dönüşümü — daha önce olduğu gibi — çağıran taraf (criterion, metrics_fn)
    tarafından uygulanıyor.

    predictor: model veya sliding_window_inference sarmalayıcısı — logit döner.
    """
    preds_list = []
    transforms = [
        lambda x: x,
        lambda x: torch.flip(x, [2]),
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2, 3]),
        lambda x: torch.rot90(x, 1, [2, 3]),
        lambda x: torch.rot90(x, 2, [2, 3]),
        lambda x: torch.rot90(x, 3, [2, 3]),
        lambda x: torch.flip(torch.rot90(x, 1, [2, 3]), [2]),
    ]

    reverse = [
        lambda x: x,
        lambda x: torch.flip(x, [2]),
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2, 3]),
        lambda x: torch.rot90(x, -1, [2, 3]),
        lambda x: torch.rot90(x, -2, [2, 3]),
        lambda x: torch.rot90(x, -3, [2, 3]),
        lambda x: torch.flip(torch.rot90(x, -1, [2, 3]), [2]),
    ]

    for fwd, rev in zip(transforms[:n_steps], reverse[:n_steps]):
        aug_img = fwd(images)
        pred    = predictor(aug_img)
        # DÜZELTME: `rev` (flip/rot90) yalnızca uzamsal (B,C,H,W) segmentasyon
        # tahminlerini geri döndürmek içindir. Önceki sürüm bunu sınıflandırma
        # çıktısına da (B,C — uzamsal boyutu yok) uyguluyordu; torch.flip/rot90
        # var olmayan 2./3. boyutu istediğinden bu, TTA + sınıflandırma
        # kombinasyonunda garantili bir RuntimeError'a yol açıyordu.
        if pred.dim() >= 4:
            pred = rev(pred)
        preds_list.append(pred)

    return torch.stack(preds_list).mean(0)


# =============================================================================
# 8. K-Fold Çapraz Doğrulama Motoru
# =============================================================================

class KFoldTrainer:
    """
    Stratified K-Fold çapraz doğrulama ile eğitim yöneticisi.

    Her fold:
      1. DataLoader oluştur
      2. Model, optimizer, scheduler sıfırla
      3. train_epoch + validate_epoch döngüsü
      4. En iyi modeli kaydet
      5. Fold sonuçlarını raporla
    """

    def __init__(
        self,
        cfg:           Any,
        dataset_files: List[Dict],
        model_fn:      Callable,
        criterion:     nn.Module,
        metrics_fn:    Callable,
        train_tfm:     Any,
        val_tfm:       Any,
        dataset_cls:   Any,
    ) -> None:
        self.cfg           = cfg
        self.dataset_files = dataset_files
        self.model_fn      = model_fn
        self.criterion     = criterion
        self.metrics_fn    = metrics_fn
        self.train_tfm     = train_tfm
        self.val_tfm       = val_tfm
        self.dataset_cls   = dataset_cls
        self.fold_results: List[Dict] = []

    def run(self) -> pd.DataFrame:
        """
        Tüm fold'ları çalıştırır ve sonuç DataFrame'i döner.
        """
        splitter = (
            StratifiedKFold(n_splits=self.cfg.KFOLD, shuffle=True, random_state=self.cfg.SEED)
            if self.cfg.STRATIFIED
            else KFold(n_splits=self.cfg.KFOLD, shuffle=True, random_state=self.cfg.SEED)
        )

        indices = np.arange(len(self.dataset_files))
        labels  = [f.get("label", 0) for f in self.dataset_files]

        for fold, (train_idx, val_idx) in enumerate(splitter.split(indices, labels)):
            logger.info("=" * 60)
            logger.info(f"FOLD {fold+1}/{self.cfg.KFOLD}")
            logger.info("=" * 60)

            train_files = [self.dataset_files[i] for i in train_idx]
            val_files   = [self.dataset_files[i] for i in val_idx]

            train_ds = self.dataset_cls(train_files, self.train_tfm, self.cfg.TASK)
            val_ds   = self.dataset_cls(val_files,   self.val_tfm,   self.cfg.TASK)

            # DÜZELTME: collate_fn=simple_collate eksikti. datasets.py'deki
            # açıklamaya göre MONAI'nin MetaTensor/meta sözlük çıktısı
            # varsayılan collate ile batch'lenemez; build_loaders() bunu
            # doğru yapıyordu ama K-Fold yolu (burası) unutmuştu.
            train_loader = DataLoader(
                train_ds,
                batch_size=self.cfg.BATCH_SIZE,
                shuffle=True,
                collate_fn=simple_collate,
                num_workers=self.cfg.NUM_WORKERS,
                pin_memory=self.cfg.PIN_MEMORY,
                drop_last=True,
                persistent_workers=self.cfg.NUM_WORKERS > 0,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                collate_fn=simple_collate,
                num_workers=self.cfg.NUM_WORKERS,
                pin_memory=self.cfg.PIN_MEMORY,
            )

            model = self.model_fn()
            # DÜZELTME: Katman bazlı öğrenme hızı (discriminative fine-tuning)
            # yalnızca run_standard_training'de uygulanıyordu; K-Fold yolu
            # model.parameters() ile düz LR kullanıyordu — README'nin
            # vaat ettiği teknik K-Fold'da devre dışıydı. Artık her iki yol
            # da aynı stratejiyi kullanıyor.
            param_groups = get_layer_wise_lr_params(model, self.cfg.LEARNING_RATE, lr_factor=0.3)
            optimizer = optimizer_factory(
                self.cfg.OPTIMIZER, param_groups,
                lr=self.cfg.LEARNING_RATE, weight_decay=self.cfg.WEIGHT_DECAY,
            )
            base_sched = scheduler_factory(
                self.cfg.SCHEDULER, optimizer,
                epochs=self.cfg.EPOCHS,
                steps_per_epoch=len(train_loader),
                T0=self.cfg.T0, T_mult=self.cfg.T_MULT, min_lr=self.cfg.MIN_LR,
            )
            # DÜZELTME: per_batch/is_plateau bayrakları eklendi — bkz.
            # WarmupScheduler docstring'i (OneCycleLR yanlış frekansta
            # ilerliyordu; ReduceLROnPlateau argümansız step()'te çöküyordu).
            scheduler = WarmupScheduler(
                optimizer, self.cfg.WARMUP_EPOCHS, base_sched, self.cfg.LEARNING_RATE,
                per_batch=(self.cfg.SCHEDULER == "one_cycle"),
                is_plateau=(self.cfg.SCHEDULER == "plateau"),
            )
            scaler       = GradScaler(self.cfg.DEVICE.type, enabled=self.cfg.USE_AMP)
            early_stop   = EarlyStopping(
                patience=self.cfg.EARLY_STOPPING_PATIENCE,
                min_delta=self.cfg.EARLY_STOPPING_MIN_DELTA,
            )

            best_metric    = -np.inf
            fold_history   = []
            ckpt_path      = self.cfg.CHECKPOINT_PATH.parent / f"fold{fold+1}_best.pth"

            for epoch in range(self.cfg.EPOCHS):
                train_res = train_epoch(
                    model, train_loader, optimizer, self.criterion,
                    scaler, self.cfg, epoch,
                    use_mixup=(self.cfg.MIXUP_PROB > 0),
                    scheduler=scheduler,
                )
                val_res = validate_epoch(
                    model, val_loader, self.criterion, self.metrics_fn, self.cfg,
                    use_tta=(self.cfg.USE_TTA and epoch >= self.cfg.EPOCHS - 5),
                )
                scheduler.step(metric=val_res["metric"])

                fold_history.append({
                    "epoch": epoch + 1, "fold": fold + 1,
                    **{f"train_{k}": v for k, v in train_res.items()},
                    **{f"val_{k}":   v for k, v in val_res.items()},
                })

                logger.info(
                    f"  [F{fold+1} E{epoch+1:03d}] "
                    f"TrainL={train_res['loss']:.4f} "
                    f"ValL={val_res['loss']:.4f} "
                    f"Metric={val_res['metric']:.4f} "
                    f"LR={train_res['lr']:.2e}"
                )

                if val_res["metric"] > best_metric:
                    best_metric = val_res["metric"]
                    torch.save({
                        "fold":        fold + 1,
                        "epoch":       epoch + 1,
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "metric":      best_metric,
                    }, ckpt_path)
                    logger.info(f"  ★ Yeni en iyi: {best_metric:.4f} → {ckpt_path.name}")

                if self.cfg.EARLY_STOPPING and early_stop(val_res["metric"]):
                    break

                del train_res, val_res
                torch.cuda.empty_cache()
                gc.collect()

            self.fold_results.append({
                "fold":        fold + 1,
                "best_metric": best_metric,
                "history":     fold_history,
                "ckpt":        str(ckpt_path),
            })
            logger.info(f"Fold {fold+1} tamamlandı. En iyi metrik: {best_metric:.4f}")

        return self._summarize()

    def _summarize(self) -> pd.DataFrame:
        """K-Fold sonuçlarını özetleyen DataFrame oluşturur."""
        rows = [{"fold": r["fold"], "best_metric": r["best_metric"]}
                for r in self.fold_results]
        df   = pd.DataFrame(rows)
        logger.info("\n" + "=" * 40)
        logger.info("K-FOLD ÖZET")
        logger.info(df.to_string(index=False))
        logger.info(f"Ortalama Metrik: {df['best_metric'].mean():.4f} ± {df['best_metric'].std():.4f}")
        logger.info("=" * 40)
        df.to_csv(self.cfg.CHECKPOINT_PATH.parent / "kfold_summary.csv", index=False)
        return df
