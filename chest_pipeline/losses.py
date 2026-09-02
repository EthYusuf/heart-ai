# =============================================================================
# losses.py — Kayıp Fonksiyonları Kütüphanesi
# -----------------------------------------------------------------------------
# Medikal görüntü analizinde yaygın kullanılan tüm kayıp fonksiyonları.
# Özellikle sınıf dengesizliği durumunda (küçük lezyon, nadir hastalık)
# standart Cross-Entropy yetersiz kalır; bu modül alternatifler sunar.
#
# Desteklenen kayıplar:
#   DiceCE, Focal, Tversky, Lovász, Boundary, Combo, Asymmetric, SoftDice
# =============================================================================

import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from monai.losses import DiceCELoss, FocalLoss, TverskyLoss, DiceLoss
from monai.networks.utils import one_hot

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Temel Sarmalayıcılar (MONAI tabanlı)
# =============================================================================

def build_dice_ce(num_classes: int, include_background: bool = False) -> nn.Module:
    """
    Dice kayıp fonksiyonu ile piksel bazlı çapraz entropi kaybının
    ağırlıklı toplamından oluşan bileşik kayıp fonksiyonu.
    Medikal segmentasyon literatüründe yaygın biçimde benimsenmiştir
    (Milletari et al., V-Net, 3DV 2016; de Bruijne, Med. Image Anal. 2022).
    """
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=include_background,
        lambda_dice=0.6,
        lambda_ce=0.4,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    )


def build_focal(gamma: float = 2.0, alpha: Optional[float] = None) -> nn.Module:
    """
    Focal Loss — kolay örneklerin ağırlığını azaltır, zor örneklere odaklanır.
    Ciddi sınıf dengesizliğinde Dice'dan daha kararlı davranır.
    """
    weight = torch.tensor([alpha, 1 - alpha]) if alpha is not None else None
    return FocalLoss(
        gamma=gamma,
        weight=weight,
        reduction="mean",
        use_softmax=True,
    )


def build_tversky(alpha: float = 0.3, beta: float = 0.7) -> nn.Module:
    """
    Tversky Loss — FP ve FN'i farklı ağırlıklarla cezalandırır.
    alpha: FP ağırlığı (düşük → FP'ye toleranslı)
    beta:  FN ağırlığı (yüksek → kaçırılan lezyonlara daha duyarlı)
    """
    return TverskyLoss(
        to_onehot_y=True,
        softmax=True,
        alpha=alpha,
        beta=beta,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        include_background=False,
    )


# =============================================================================
# 2. Özel Kayıp Fonksiyonları
# =============================================================================

class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász uzantısı kullanılarak Jaccard / IoU metriğini doğrudan
    türevlenebilir biçimde optimize eden kayıp fonksiyonu.
    Segmentasyon değerlendirme metriği ile kayıp fonksiyonu arasındaki
    uyumsuzluğu (surrogate gap) ortadan kaldırır.
    Referans: Berman et al., CVPR 2018.
    """

    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        super().__init__()
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        preds   : (B, C, H, W) — ham logits
        targets : (B, 1, H, W) — sınıf indisleri
        """
        probs = F.softmax(preds, dim=1)
        loss  = self._lovasz_softmax(probs, targets.squeeze(1))
        return loss

    def _lovasz_softmax(self, probs: Tensor, labels: Tensor) -> Tensor:
        losses = []
        for c in range(self.num_classes):
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probs[:, c]).abs()
            errors_sorted, perm = torch.sort(errors.view(-1), descending=True)
            fg_sorted = fg.view(-1)[perm]
            losses.append(torch.dot(errors_sorted, self._lovasz_grad(fg_sorted)))
        return torch.stack(losses).mean() if losses else probs.sum() * 0

    @staticmethod
    def _lovasz_grad(gt_sorted: Tensor) -> Tensor:
        n  = gt_sorted.numel()
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union        = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard      = 1.0 - intersection / union
        jaccard[1:]  = jaccard[1:] - jaccard[:-1]
        return jaccard


class BoundaryLoss(nn.Module):
    """
    Boundary Loss — tahmin ve gerçek sınırlar arasındaki mesafeyi minimize eder.
    İnce yapıların (damarlar, çatlaklar) kenar hassasiyetini artırır.
    Referans: Kervadec et al., MIDL 2019.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        from scipy.ndimage import distance_transform_edt
        import numpy as np

        probs = F.softmax(preds, dim=1)
        batch_loss = []

        targets_np = targets.squeeze(1).cpu().numpy()
        for b in range(targets_np.shape[0]):
            dist_maps = []
            for c in range(self.num_classes):
                mask    = (targets_np[b] == c).astype(np.float32)
                dist    = distance_transform_edt(1 - mask) - distance_transform_edt(mask)
                dist_maps.append(dist)

            dist_tensor = torch.from_numpy(np.stack(dist_maps)).float().to(preds.device)
            # Normalize
            dist_tensor = dist_tensor / (dist_tensor.abs().max() + 1e-8)
            batch_loss.append((probs[b] * dist_tensor).sum())

        return torch.stack(batch_loss).mean()


class ComboLoss(nn.Module):
    """
    Combo Loss = α·DiceLoss + β·BoundaryLoss + γ·FocalLoss
    Her kaybın bütünleyici bilgisini birleştirir.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.6,
        beta:  float = 0.2,
        gamma: float = 0.2,
    ) -> None:
        super().__init__()
        assert abs(alpha + beta + gamma - 1.0) < 1e-5, "Ağırlıklar toplamı 1 olmalı"
        self.alpha    = alpha
        self.beta     = beta
        self.gamma    = gamma
        self.dice     = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
        self.boundary = BoundaryLoss(num_classes)
        self.focal    = FocalLoss(gamma=2.0, use_softmax=True)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        d_loss = self.dice(preds, targets)
        b_loss = self.boundary(preds, targets)
        f_loss = self.focal(preds, targets)
        total  = self.alpha * d_loss + self.beta * b_loss + self.gamma * f_loss
        return total


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss — çok etiketli sınıflandırma için.
    Pozitif ve negatif örneklere farklı odaklama uygular.
    Referans: Ridnik et al., ICCV 2021.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip:      float = 0.05,
        eps:       float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        preds_sigmoid = torch.sigmoid(preds)
        # Clip: negatif örnekler için eşik
        preds_sigmoid_neg = (preds_sigmoid + self.clip).clamp(max=1.0)

        # Pozitif / negatif kayıplar
        loss_pos = targets         * torch.log(preds_sigmoid.clamp(min=self.eps))
        loss_neg = (1 - targets)   * torch.log((1 - preds_sigmoid_neg).clamp(min=self.eps))

        # Odaklama ağırlıkları
        loss = loss_pos + loss_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt   = preds_sigmoid * targets + (1 - preds_sigmoid) * (1 - targets)
            gam  = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss *= (1 - pt) ** gam

        return -loss.mean()


class SoftDiceLoss(nn.Module):
    """
    Yumuşak Dice Loss — sınır bölgelerine dikkat çekmek için
    probability map üzerinde doğrudan hesaplanır (argmax yok).
    """

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        probs = torch.softmax(preds, dim=1)
        # One-hot encoding
        n_classes = preds.shape[1]
        targets_oh = F.one_hot(targets.squeeze(1).long(), n_classes).permute(0, 3, 1, 2).float()
        # Dice her sınıf için ayrı hesaplanır, arka plan hariç
        dice_scores = []
        for c in range(1, n_classes):
            p = probs[:, c]
            t = targets_oh[:, c]
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            dice_scores.append(1 - (2 * inter + self.smooth) / (union + self.smooth))
        return torch.stack(dice_scores).mean()


# =============================================================================
# 3. Kayıp Fabrikası
# =============================================================================

def build_cross_entropy(label_smoothing: float = 0.0) -> nn.Module:
    """
    Çok sınıflı sınıflandırma için standart Cross-Entropy.
    Girdi: (B, C) logit — Hedef: (B,) sınıf indeksi.
    Segmentasyon kayıpları (Dice vb.) bu şekli kabul etmez.
    """
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def build_bce(pos_weight: Optional[Tensor] = None) -> nn.Module:
    """
    Çok-etiketli sınıflandırma için sigmoid + binary cross-entropy.
    Girdi: (B, C) logit — Hedef: (B, C) 0/1 float.

    pos_weight: sınıf başına pozitif örnek ağırlığı. NIH gibi seyrek pozitifli
    veri setlerinde (bazı sınıflarda %1'in altında) modelin her şeye "negatif"
    demesini engeller.
    """
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


LOSS_REGISTRY = {
    "ce":         build_cross_entropy,
    "bce":        build_bce,
    "dice_ce":    build_dice_ce,
    "focal":      build_focal,
    "tversky":    build_tversky,
    "lovasz":     LovaszSoftmaxLoss,
    "boundary":   BoundaryLoss,
    "combo":      ComboLoss,
    "asymmetric": AsymmetricLoss,
    "soft_dice":  SoftDiceLoss,
}


def loss_factory(
    loss_name:   str,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    """
    İsme göre kayıp fonksiyonu oluşturur.

    Örnek:
        criterion = loss_factory("combo", num_classes=2, alpha=0.5, beta=0.3, gamma=0.2)
    """
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Bilinmeyen kayıp: '{loss_name}'. "
            f"Geçerli seçenekler: {list(LOSS_REGISTRY.keys())}"
        )

    builder = LOSS_REGISTRY[loss_name]

    import inspect
    sig    = inspect.signature(builder)
    params = sig.parameters

    call_kwargs: dict = {}
    if "num_classes" in params:
        call_kwargs["num_classes"] = num_classes
    call_kwargs.update({k: v for k, v in kwargs.items() if k in params})

    loss = builder(**call_kwargs)
    logger.info(f"Kayıp fonksiyonu: {loss_name} | Parametreler: {call_kwargs}")
    return loss
