# =============================================================================
# models.py — Çoklu Model Mimarileri
# -----------------------------------------------------------------------------
# Desteklenen mimariler:
#   Segmentasyon : U-Net, U-Net++, Attention U-Net, Swin-UNet, SegResNet
#   Sınıflandırma: DenseNet121, EfficientNet-B4, ResNet50, Vision Transformer
#
# Tüm modeller MONAI'nin pretrained ağırlıklarından veya torchvision'dan yüklenir.
# model_factory() ile tek çağrıda doğru modeli alın.
# =============================================================================

import logging
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import monai
from monai.networks.nets import (
    UNet,
    BasicUNet,
    AttentionUnet,
    SwinUNETR,
    SegResNet,
    DenseNet121,
    EfficientNetBN,
    ViT,
)
from monai.networks.layers import Norm, Act

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Segmentasyon Mimarileri
# =============================================================================

def build_unet(in_channels: int, num_classes: int) -> nn.Module:
    """
    MONAI U-Net — standart encoder-decoder mimarisi.
    Residual birimler ve batch normalizasyon ile güçlendirilmiştir.
    """
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=3,
        dropout=0.15,
        act="PRELU",
        norm=Norm.BATCH,
        bias=True,
    )


def build_unet_pp(in_channels: int, num_classes: int) -> nn.Module:
    """
    U-Net++ (Nested U-Net) — yeniden tasarlanmış skip bağlantıları ile
    standart U-Net'e kıyasla daha keskin sınır segmentasyonu sağlar.
    """
    # MONAI BasicUNet üzerinde ++  skip bağlantıları manuel eklenerek
    # hafif bir U-Net++ benzetimi kurulur. Tam implementasyon için
    # segmentation-models-pytorch kullanılabilir.
    try:
        import segmentation_models_pytorch as smp
        model = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet" if in_channels == 3 else None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
        logger.info("Model: U-Net++ (smp, ResNet50 encoder)")
    except ImportError:
        logger.warning("segmentation-models-pytorch bulunamadı → MONAI BasicUNet kullanılıyor.")
        model = BasicUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=num_classes,
            features=(32, 64, 128, 256, 512, 32),
            dropout=0.15,
            act="mish",
        )
    return model


def build_attention_unet(in_channels: int, num_classes: int) -> nn.Module:
    """
    Attention U-Net — geri yayılım yoluna dikkat kapıları ekler.
    Küçük/ince yapıların (nodüller, damarlar) segmentasyonunda üstündür.
    """
    return AttentionUnet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=(64, 128, 256, 512),
        strides=(2, 2, 2),
        dropout=0.15,
    )


def build_swin_unet(in_channels: int, num_classes: int,
                    image_size: Tuple[int, int] = (512, 512)) -> nn.Module:
    """
    Swin-UNET (Swin Transformer + U-Net decoder).
    Uzun menzilli bağımlılıkları yakalayan transformer tabanlı segmentasyon.
    Yüksek çözünürlüklü görüntülerde daha yavaş ama daha doğrudur.
    """
    return SwinUNETR(
        img_size=image_size,
        in_channels=in_channels,
        out_channels=num_classes,
        feature_size=48,
        use_checkpoint=True,   # Gradient checkpointing — VRAM tasarrufu
        spatial_dims=2,
        drop_rate=0.0,
        attn_drop_rate=0.0,
    )


def build_segresnet(in_channels: int, num_classes: int) -> nn.Module:
    """
    SegResNet — 3D medikal segmentasyon için geliştirilmiş residual ağ.
    2D görüntüler için de kullanılabilir.
    """
    return SegResNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=num_classes,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
        act=("relu", {"inplace": True}),
        norm=("group", {"num_groups": 8}),
    )


# =============================================================================
# 2. Sınıflandırma Mimarileri
# =============================================================================

def build_densenet(in_channels: int, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    DenseNet-121 — göğüs X-ray sınıflandırmasında standart referans model.
    CheXNet makalesinde kullanılan mimari.
    """
    return DenseNet121(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=num_classes,
        dropout_prob=0.25,
        pretrained=pretrained,
    )


def build_efficientnet(in_channels: int, num_classes: int) -> nn.Module:
    """
    EfficientNet-B4 — genişlik, derinlik ve giriş çözünürlüğünü bileşik
    ölçekleme (compound scaling) yöntemiyle eş zamanlı optimize eden mimari.
    Eşdeğer hesaplama maliyetinde standart ResNet ailesi üzerinde tutarlı
    doğruluk kazanımı gözlemlenmiştir (Tan & Le, ICML 2019).
    """
    model = EfficientNetBN(
        model_name="efficientnet-b4",
        spatial_dims=2,
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=True,
        adv_prop=False,
    )
    return model


def build_resnet50(in_channels: int, num_classes: int) -> nn.Module:
    """
    ResNet-50 — artık bağlantı (residual connection) mimarisi ile
    degradasyon sorununu aşan 50 katmanlı derin ağ (He et al., CVPR 2016).
    ImageNet ön-eğitim ağırlıkları transfer öğrenme yoluyla uyarlanır;
    sınıflandırıcı başlık hedef veri setine göre yeniden yapılandırılır.
    """
    import torchvision.models as tv_models
    # DÜZELTME: `pretrained=` argümanı torchvision'da 0.13'ten beri
    # kullanımdan kaldırılmış durumda ve güncel sürümlerde tamamen
    # kaldırılmış olabilir (TypeError riski). `weights=` API'si kullanılır;
    # çok eski torchvision (<0.13) ile geriye dönük uyumluluk korunur.
    try:
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if in_channels == 3 else None
        model = tv_models.resnet50(weights=weights)
    except AttributeError:
        model = tv_models.resnet50(pretrained=(in_channels == 3))

    # Tek kanallı giriş için ilk konvülsyonu değiştir
    if in_channels != 3:
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        # Kanal ortalamasıyla ağırlık transferi.
        # DÜZELTME: Sondaki `/ in_channels` fazladan bir ölçeklendirmeydi —
        # `.mean(dim=1)` zaten 3 RGB filtresinin ortalamasını alıp tek bir
        # kanala indirgiyor; bu değeri `in_channels` kez tekrarlamak
        # (expand/repeat) evrişimin beklediği aktivasyon büyüklüğünü zaten
        # korur, ek bölme yalnızca ağırlıkları gereksiz yere küçültüp
        # (in_channels=1 dışında) transfer öğrenmenin faydasını azaltıyordu.
        with torch.no_grad():
            model.conv1.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            )

    # Sınıflandırma başlığını değiştir
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    logger.info("Model: ResNet-50 (özel sınıflandırma başlığı)")
    return model


def build_vit(in_channels: int, num_classes: int,
              image_size: Tuple[int, int] = (512, 512)) -> nn.Module:
    """
    Vision Transformer (ViT-B/16) — büyük veri setlerinde üstün performans.
    """
    model = ViT(
        in_channels=in_channels,
        img_size=image_size,
        patch_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        num_classes=num_classes,
        dropout_rate=0.1,
        spatial_dims=2,
        classification=True,
    )
    logger.info("Model: Vision Transformer ViT-B/16")
    return model


# =============================================================================
# 3. Model Fabrikası
# =============================================================================

MODEL_REGISTRY: Dict[str, Any] = {
    # Segmentasyon
    "unet":           build_unet,
    "unet_pp":        build_unet_pp,
    "attention_unet": build_attention_unet,
    "swin_unet":      build_swin_unet,
    "segresnet":      build_segresnet,
    # Sınıflandırma
    "densenet121":    build_densenet,
    "efficientnet":   build_efficientnet,
    "resnet50":       build_resnet50,
    "vit":            build_vit,
}


def model_factory(
    model_name:  str,
    in_channels: int,
    num_classes: int,
    device:      torch.device,
    image_size:  Optional[Tuple[int, int]] = None,
    pretrained:  bool = True,
) -> nn.Module:
    """
    İsme göre model oluşturur, cihaza taşır ve model özetini loglar.

    Args:
        model_name  : MODEL_REGISTRY anahtarı
        in_channels : Giriş kanalı sayısı
        num_classes : Çıkış sınıf sayısı
        device      : torch.device
        image_size  : Transformer modelleri için gerekli
        pretrained  : Transfer öğrenme

    Returns:
        Eğitime hazır nn.Module
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Bilinmeyen model: '{model_name}'. "
            f"Geçerli seçenekler: {list(MODEL_REGISTRY.keys())}"
        )

    builder = MODEL_REGISTRY[model_name]

    # İmzaya göre parametre geçirme
    import inspect
    sig = inspect.signature(builder)
    kwargs: Dict[str, Any] = {"in_channels": in_channels, "num_classes": num_classes}
    if "image_size" in sig.parameters and image_size:
        kwargs["image_size"] = image_size
    if "pretrained" in sig.parameters:
        kwargs["pretrained"] = pretrained

    model = builder(**kwargs)

    # Çoklu GPU desteği
    if torch.cuda.device_count() > 1:
        logger.info(f"DataParallel: {torch.cuda.device_count()} GPU")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Parametre sayısı raporu
    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_name} | "
                f"Toplam parametre: {total/1e6:.2f}M | "
                f"Eğitilebilir: {train/1e6:.2f}M")
    return model


# =============================================================================
# 4. Model Dondurma / Çözme (Fine-Tuning Stratejisi)
# =============================================================================

def freeze_encoder(model: nn.Module, freeze_until: Optional[str] = None) -> None:
    """
    Transfer öğrenmede encoder'ı dondurur, decoder'ı eğitilebilir bırakır.
    İlk epoch'larda encoder'ı dondurmak, decoder'ın hızla uyum sağlamasını sağlar.

    Args:
        freeze_until : Bu katman ismine kadar dondur (None = tamamını)
    """
    frozen = 0
    for name, param in model.named_parameters():
        if freeze_until and freeze_until in name:
            break
        param.requires_grad = False
        frozen += 1
    logger.info(f"Dondurulmuş parametre grubu sayısı: {frozen}")


def unfreeze_all(model: nn.Module) -> None:
    """Tüm parametreleri eğitilebilir yapar (tam ince ayar için)."""
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Tüm parametreler serbest bırakıldı.")


def get_layer_wise_lr_params(
    model:     nn.Module,
    base_lr:   float,
    lr_factor: float = 0.1,
) -> list:
    """
    Katman bazlı öğrenme hızı — encoder katmanları daha düşük LR alır.
    Discriminative fine-tuning olarak da bilinir.

    Args:
        base_lr   : Decoder / çıktı katmanı LR
        lr_factor : Her önceki katman grubu için çarpan (varsayılan: 0.1x)

    Returns:
        optimizer'a geçirilecek param grupları listesi
    """
    param_groups = []
    named_params = list(model.named_parameters())

    # Parametreleri derinliğe göre sırala
    depths = {}
    for name, param in named_params:
        depth = name.count(".")
        depths.setdefault(depth, []).append((name, param))

    max_depth = max(depths.keys())
    for depth, params in sorted(depths.items()):
        lr = base_lr * (lr_factor ** (max_depth - depth))
        param_groups.append({
            "params": [p for _, p in params if p.requires_grad],
            "lr":     lr,
            "name":   f"depth_{depth}",
        })

    logger.info(f"Katman bazlı LR: {len(param_groups)} grup, "
                f"[{param_groups[-1]['lr']:.2e} ... {param_groups[0]['lr']:.2e}]")
    return param_groups


# =============================================================================
# 5. Model Ensemble
# =============================================================================

class EnsembleModel(nn.Module):
    """
    Çoklu modelden yumuşak oy birliği (soft voting) ile tahmin üretir.
    Her modelin ağırlığı ayarlanabilir.
    """

    def __init__(
        self,
        models:  list,
        weights: Optional[list] = None,
        task:    str  = "segmentation",
        multi_label: bool = False,
    ) -> None:
        super().__init__()
        self.models      = nn.ModuleList(models)
        self.weights     = weights or [1.0 / len(models)] * len(models)
        self.task        = task
        self.multi_label = multi_label
        assert abs(sum(self.weights) - 1.0) < 1e-5, "Ağırlıklar toplamı 1 olmalı"
        logger.info(f"Ensemble: {len(models)} model, ağırlıklar={[f'{w:.2f}' for w in self.weights]}")

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        for model, w in zip(self.models, self.weights):
            model.eval()
            out = model(x)
            if self.task == "segmentation":
                out = torch.softmax(out, dim=1)
            elif self.multi_label:
                # Çok-etiketli sınıflandırma: sınıflar bağımsız → sigmoid.
                out = torch.sigmoid(out)
            else:
                # DÜZELTME: Tek-etiketli (standart) sınıflandırmada sınıflar
                # birbirini dışlar; sigmoid kullanmak (her sınıfı bağımsız
                # olasılık gibi ele alıp) yanlış/normalize olmayan tahminler
                # üretir. ClassificationEvaluator ve trainer.py'deki metrik
                # fonksiyonlarıyla tutarlı olması için softmax kullanılır.
                out = torch.softmax(out, dim=1)
            outputs.append(out * w)
        return sum(outputs)

    def load_checkpoints(self, checkpoint_paths: list, device: torch.device) -> None:
        """Her alt modele ait checkpoint'i yükler."""
        for model, path in zip(self.models, checkpoint_paths):
            state = torch.load(path, map_location=device)
            model.load_state_dict(state["model_state"])
            logger.info(f"Checkpoint yüklendi: {path}")
