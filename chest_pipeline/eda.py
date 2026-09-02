# =============================================================================
# eda.py — Kapsamlı Keşifsel Veri Analizi (Exploratory Data Analysis)
# -----------------------------------------------------------------------------
# 40 GB medikal görüntü veri seti için bellek güvenli EDA.
# Görüntüleri teker teker okur, istatistikleri biriktirerek RAM taşmasını önler.
# Çıktı: istatistik tabloları, dağılım grafikleri, örnek görselleştirmeler,
#        korelasyon analizleri, veri kalitesi raporu.
# =============================================================================

import gc
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from scipy import stats
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı: Tek Görüntü İstatistiği Çıkarma
# ─────────────────────────────────────────────────────────────────────────────

def _extract_image_stats(args: Tuple[Path, str]) -> Optional[Dict[str, Any]]:
    """
    Tek bir görüntüden bellek verimli istatistik çıkarır.
    ThreadPoolExecutor ile paralel kullanım için bağımsız fonksiyon.
    """
    path, fmt = args
    try:
        if fmt == "dcm":
            import pydicom
            ds  = pydicom.dcmread(str(path), stop_before_pixels=False)
            arr = ds.pixel_array.astype(np.float32)
            meta = {
                "modality":     getattr(ds, "Modality", "?"),
                "manufacturer": getattr(ds, "Manufacturer", "?"),
                "kvp":          float(getattr(ds, "KVP", np.nan)),
                "rows":         int(getattr(ds, "Rows", 0)),
                "cols":         int(getattr(ds, "Columns", 0)),
            }
        elif fmt in ("nii", "nii.gz"):
            import nibabel as nib
            img = nib.load(str(path))
            arr = img.get_fdata(dtype=np.float32)
            meta = {"voxel_dims": str(img.header.get_zooms())}
        else:
            arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
            meta = {}

        flat = arr.flatten()
        return {
            "path":       str(path),
            "height":     arr.shape[0],
            "width":      arr.shape[1] if arr.ndim > 1 else 1,
            "channels":   arr.shape[2] if arr.ndim > 2 else 1,
            "min":        float(flat.min()),
            "max":        float(flat.max()),
            "mean":       float(flat.mean()),
            "std":        float(flat.std()),
            "median":     float(np.median(flat)),
            "p5":         float(np.percentile(flat, 5)),
            "p95":        float(np.percentile(flat, 95)),
            "skewness":   float(stats.skew(flat)),
            "kurtosis":   float(stats.kurtosis(flat)),
            "zero_ratio": float((flat == 0).mean()),   # Siyah piksel oranı
            "file_kb":    path.stat().st_size / 1024,
            **meta,
        }
    except Exception as e:
        logger.debug(f"İstatistik çıkarılamadı [{path.name}]: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# EDA Ana Sınıfı
# ─────────────────────────────────────────────────────────────────────────────

class MedicalEDA:
    """
    40 GB medikal görüntü veri seti için kapsamlı EDA motoru.

    Kullanım:
        eda = MedicalEDA(image_paths, labels_df, output_dir, fmt="dcm")
        report = eda.run_full_analysis()
    """

    # Seaborn renk paleti
    PALETTE = sns.color_palette("husl", 12)

    def __init__(
        self,
        image_paths: List[Path],
        labels_df:   pd.DataFrame,
        output_dir:  Path,
        fmt:         str = "png",
        n_workers:   int = 4,
        sample_cap:  int = 5000,   # İstatistik için maksimum görüntü sayısı
    ) -> None:
        self.image_paths = image_paths
        self.labels_df   = labels_df
        self.output_dir  = output_dir
        self.fmt         = fmt.lstrip(".")
        self.n_workers   = n_workers
        self.sample_cap  = sample_cap
        self.stats_df: Optional[pd.DataFrame] = None
        output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.1)

    # ── 1. Ham İstatistik Toplama ─────────────────────────────────────────────

    def collect_image_statistics(self) -> pd.DataFrame:
        """
        Görüntü başına istatistikleri paralel iş parçacıklarıyla toplar.
        sample_cap sayıdan fazla görüntü varsa rastgele örnekler.
        """
        paths = self.image_paths
        if len(paths) > self.sample_cap:
            rng   = np.random.default_rng(42)
            paths = list(rng.choice(paths, self.sample_cap, replace=False))
            logger.info(f"Büyük veri seti: {len(self.image_paths):,} görüntüden "
                        f"{self.sample_cap:,} örneklendi")

        args    = [(p, self.fmt) for p in paths]
        records = []

        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {pool.submit(_extract_image_stats, a): a for a in args}
            for fut in tqdm(as_completed(futures), total=len(args),
                            desc="📊 İstatistik toplama", unit="img"):
                result = fut.result()
                if result:
                    records.append(result)

        self.stats_df = pd.DataFrame(records)
        self.stats_df.to_csv(self.output_dir / "image_stats.csv", index=False)
        logger.info(f"İstatistik tablosu: {self.stats_df.shape} → image_stats.csv")
        return self.stats_df

    # ── 2. Etiket Dağılımı ────────────────────────────────────────────────────

    def plot_label_distribution(self) -> None:
        """Sınıf dengesizliğini pasta + bar grafikleriyle gösterir."""
        if "label" not in self.labels_df.columns:
            logger.warning("'label' sütunu bulunamadı — etiket analizi atlandı.")
            return

        counts = self.labels_df["label"].value_counts().sort_index()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Etiket (Sınıf) Dağılımı", fontsize=14, fontweight="bold")

        # Bar grafiği
        bars = axes[0].bar(counts.index.astype(str), counts.values,
                           color=self.PALETTE[:len(counts)], edgecolor="white", linewidth=1.2)
        axes[0].set_xlabel("Sınıf"); axes[0].set_ylabel("Örnek Sayısı")
        axes[0].set_title("Sınıf Frekansları")
        for bar, v in zip(bars, counts.values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                         f"{v:,}\n({v/counts.sum()*100:.1f}%)",
                         ha="center", va="bottom", fontsize=9)

        # Pasta grafiği
        axes[1].pie(counts.values, labels=[f"Sınıf {i}" for i in counts.index],
                    colors=self.PALETTE[:len(counts)], autopct="%1.1f%%",
                    startangle=90, shadow=True)
        axes[1].set_title("Sınıf Oranları")

        # Dengesizlik oranı
        imbalance = counts.max() / counts.min()
        fig.text(0.5, 0.01, f"Dengesizlik Oranı: {imbalance:.2f}x",
                 ha="center", fontsize=10, style="italic",
                 color="red" if imbalance > 3 else "green")

        self._save_fig(fig, "label_distribution.png")

    # ── 3. Görüntü Boyutu Analizi ─────────────────────────────────────────────

    def plot_image_size_analysis(self) -> None:
        """Yükseklik/genişlik dağılımı ve en-boy oranı histogramları."""
        if self.stats_df is None:
            self.collect_image_statistics()

        df  = self.stats_df
        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
        fig.suptitle("Görüntü Boyutu ve Geometri Analizi", fontsize=14, fontweight="bold")

        # Yükseklik dağılımı
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.hist(df["height"], bins=40, color=self.PALETTE[0], edgecolor="white")
        ax0.axvline(df["height"].median(), color="red", linestyle="--", label=f"Medyan={df['height'].median():.0f}")
        ax0.set_title("Yükseklik Dağılımı"); ax0.set_xlabel("px"); ax0.legend()

        # Genişlik dağılımı
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.hist(df["width"], bins=40, color=self.PALETTE[1], edgecolor="white")
        ax1.axvline(df["width"].median(), color="red", linestyle="--", label=f"Medyan={df['width'].median():.0f}")
        ax1.set_title("Genişlik Dağılımı"); ax1.set_xlabel("px"); ax1.legend()

        # En-boy oranı
        ax2 = fig.add_subplot(gs[0, 2])
        aspect = df["width"] / df["height"].replace(0, np.nan)
        ax2.hist(aspect.dropna(), bins=40, color=self.PALETTE[2], edgecolor="white")
        ax2.axvline(1.0, color="gray", linestyle=":", label="Kare (1:1)")
        ax2.set_title("En-Boy Oranı"); ax2.set_xlabel("W/H"); ax2.legend()

        # Boyut scatter (width vs height)
        ax3 = fig.add_subplot(gs[1, 0])
        scatter = ax3.scatter(df["width"], df["height"], alpha=0.4, s=8,
                              c=df["file_kb"], cmap="viridis")
        plt.colorbar(scatter, ax=ax3, label="Dosya Boyutu (KB)")
        ax3.set_xlabel("Genişlik (px)"); ax3.set_ylabel("Yükseklik (px)")
        ax3.set_title("Boyut Scatter (renkler = dosya KB)")

        # Dosya boyutu dağılımı
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(df["file_kb"], bins=50, color=self.PALETTE[3], edgecolor="white", log=True)
        ax4.set_title("Dosya Boyutu Dağılımı (log)"); ax4.set_xlabel("KB")

        # Özet tablo
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis("off")
        summary = df[["height", "width", "file_kb"]].describe().round(1)
        table = ax5.table(
            cellText=summary.values.astype(str),
            rowLabels=summary.index,
            colLabels=["Yükseklik", "Genişlik", "Dosya(KB)"],
            cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False); table.set_fontsize(8)
        ax5.set_title("Özet İstatistikler", pad=20)

        self._save_fig(fig, "image_size_analysis.png")

    # ── 4. Piksel Yoğunluğu Analizi ──────────────────────────────────────────

    def plot_intensity_analysis(self) -> None:
        """
        Piksel yoğunluk dağılımı — ortalama, std, çarpıklık, basıklık.
        Sınıfa göre karşılaştırmalı yoğunluk profilleri.
        """
        if self.stats_df is None:
            self.collect_image_statistics()

        df  = self.stats_df
        fig = plt.figure(figsize=(18, 12))
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
        fig.suptitle("Piksel Yoğunluğu Analizi", fontsize=14, fontweight="bold")

        metrics = [
            ("mean",     "Ortalama Yoğunluk",    self.PALETTE[0]),
            ("std",      "Std. Sapma",            self.PALETTE[1]),
            ("median",   "Medyan Yoğunluk",       self.PALETTE[2]),
            ("p5",       "5. Persentil",           self.PALETTE[3]),
            ("p95",      "95. Persentil",          self.PALETTE[4]),
            ("skewness", "Çarpıklık (Skewness)",   self.PALETTE[5]),
            ("kurtosis", "Basıklık (Kurtosis)",    self.PALETTE[6]),
            ("zero_ratio","Sıfır Piksel Oranı",    self.PALETTE[7]),
            ("file_kb",  "Dosya Boyutu (KB)",      self.PALETTE[8]),
        ]

        for idx, (col, title, color) in enumerate(metrics):
            if col not in df.columns:
                continue
            ax  = fig.add_subplot(gs[idx // 3, idx % 3])
            data = df[col].dropna()
            ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.85)
            ax.axvline(data.mean(),   color="red",    linestyle="--", lw=1.5,
                       label=f"Ort={data.mean():.2f}")
            ax.axvline(data.median(), color="orange", linestyle=":",  lw=1.5,
                       label=f"Med={data.median():.2f}")
            ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        self._save_fig(fig, "intensity_analysis.png")

    # ── 5. Sınıf Bazlı Yoğunluk Karşılaştırması ──────────────────────────────

    def plot_classwise_intensity(self) -> None:
        """Her sınıf için violin plot ile yoğunluk dağılımı karşılaştırması."""
        if self.stats_df is None or "label" not in self.labels_df.columns:
            return

        df = self.stats_df.copy()
        # Path eşleştirme ile etiket birleştirme
        if "path" in self.labels_df.columns:
            df["label"] = df["path"].map(
                dict(zip(self.labels_df["path"].astype(str),
                         self.labels_df["label"]))
            )
        else:
            logger.warning("Etiket eşleştirme yapılamadı — classwise analiz atlandı.")
            return

        df = df.dropna(subset=["label"])
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Sınıf Bazlı Yoğunluk Karşılaştırması", fontsize=14, fontweight="bold")

        for ax, col, title in zip(
            axes.flat,
            ["mean", "std", "p5", "p95"],
            ["Ortalama", "Standart Sapma", "5. Persentil", "95. Persentil"],
        ):
            sns.violinplot(data=df, x="label", y=col, ax=ax,
                           palette=self.PALETTE, inner="box", linewidth=1.2)
            ax.set_title(title); ax.set_xlabel("Sınıf"); ax.grid(alpha=0.3)

            # İstatistiksel anlamlılık testi (Mann-Whitney U)
            classes = df["label"].unique()
            if len(classes) == 2:
                a = df.loc[df["label"] == classes[0], col].dropna()
                b = df.loc[df["label"] == classes[1], col].dropna()
                _, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
                sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
                ax.set_title(f"{title}  [p={pval:.4f} {sig}]")

        self._save_fig(fig, "classwise_intensity.png")

    # ── 6. Korelasyon Matrisi ─────────────────────────────────────────────────

    def plot_correlation_matrix(self) -> None:
        """Sayısal özellikler arası Spearman korelasyon ısı haritası."""
        if self.stats_df is None:
            self.collect_image_statistics()

        num_cols = self.stats_df.select_dtypes(include=np.number).columns.tolist()
        corr     = self.stats_df[num_cols].corr(method="spearman")

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            square=True, ax=ax, annot_kws={"size": 7},
        )
        ax.set_title("Özellik Korelasyon Matrisi (Spearman)", fontsize=13, fontweight="bold")
        self._save_fig(fig, "correlation_matrix.png")

    # ── 7. Örnek Görüntüler Grid'i ────────────────────────────────────────────

    def plot_sample_grid(self, n_per_class: int = 4) -> None:
        """
        Her sınıftan n_per_class kadar görüntüyü grid formatında gösterir.
        Histogram yerleşimi ile gerçek piksel dağılımı da eklenir.
        """
        if "label" not in self.labels_df.columns:
            return

        classes  = sorted(self.labels_df["label"].unique())
        n_cols   = n_per_class
        n_rows   = len(classes) * 2     # Her sınıf için görüntü + histogram satırı
        fig      = plt.figure(figsize=(n_cols * 3 + 1, n_rows * 3))
        fig.suptitle("Sınıf Başına Örnek Görüntüler + Histogramlar",
                     fontsize=13, fontweight="bold", y=1.01)

        for cls_idx, cls in enumerate(classes):
            subset  = self.labels_df[self.labels_df["label"] == cls]
            samples = subset.sample(min(n_per_class, len(subset)), random_state=42)

            for col_idx, (_, row) in enumerate(samples.iterrows()):
                try:
                    path = Path(row.get("image_path", row.get("path", "")))
                    arr  = np.array(Image.open(path).convert("L"), dtype=np.float32)

                    # Görüntü
                    ax_img = fig.add_subplot(n_rows, n_cols,
                                             cls_idx * 2 * n_cols + col_idx + 1)
                    ax_img.imshow(arr, cmap="gray")
                    ax_img.set_title(f"Sınıf {cls}", fontsize=8)
                    ax_img.axis("off")

                    # Histogram
                    ax_hist = fig.add_subplot(n_rows, n_cols,
                                              (cls_idx * 2 + 1) * n_cols + col_idx + 1)
                    ax_hist.hist(arr.flatten(), bins=50, color=self.PALETTE[cls_idx],
                                 edgecolor="none", density=True)
                    ax_hist.set_xlabel("Yoğunluk", fontsize=7)
                    ax_hist.tick_params(labelsize=6)
                except Exception as e:
                    logger.debug(f"Örnek görüntü yüklenemedi: {e}")

        plt.tight_layout()
        self._save_fig(fig, "sample_grid.png")

    # ── 8. Mask / Segmentasyon EDA ────────────────────────────────────────────

    def plot_mask_analysis(self, mask_paths: List[Path]) -> None:
        """
        Segmentasyon maskesi EDA:
        - Nesne piksel oranı (foreground ratio)
        - Nesne sınır sıklığı (boundary density)
        - Maske boyutu korelasyonu
        """
        records = []
        for mp in tqdm(mask_paths[:self.sample_cap], desc="🎭 Maske analizi"):
            try:
                mask  = np.array(Image.open(mp).convert("L")) > 0
                total = mask.size
                fg    = mask.sum()
                # Konvülsyonla kenar yoğunluğu (gradyan magnitude)
                from scipy.ndimage import sobel
                gx  = sobel(mask.astype(float), axis=0)
                gy  = sobel(mask.astype(float), axis=1)
                bnd = np.sqrt(gx**2 + gy**2).mean()
                records.append({
                    "path":           str(mp),
                    "fg_ratio":       fg / total,
                    "boundary_density": bnd,
                    "mask_area_px":   int(fg),
                })
            except Exception:
                pass

        if not records:
            return

        mask_df = pd.DataFrame(records)
        mask_df.to_csv(self.output_dir / "mask_stats.csv", index=False)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Segmentasyon Maskesi Analizi", fontsize=13, fontweight="bold")

        axes[0].hist(mask_df["fg_ratio"], bins=40, color=self.PALETTE[0], edgecolor="white")
        axes[0].set_title("Ön Plan Piksel Oranı"); axes[0].set_xlabel("Oran")

        axes[1].hist(mask_df["boundary_density"], bins=40, color=self.PALETTE[1], edgecolor="white")
        axes[1].set_title("Kenar Yoğunluğu"); axes[1].set_xlabel("Yoğunluk")

        axes[2].scatter(mask_df["fg_ratio"], mask_df["boundary_density"],
                        alpha=0.3, s=8, color=self.PALETTE[2])
        axes[2].set_xlabel("Ön Plan Oranı"); axes[2].set_ylabel("Kenar Yoğunluğu")
        axes[2].set_title("Oran vs Kenar Yoğunluğu")

        self._save_fig(fig, "mask_analysis.png")

    # ── 9. DICOM Meta Veri Analizi ────────────────────────────────────────────

    def plot_dicom_metadata(self) -> None:
        """DICOM modalitesi, üretici, kV parametrelerinin dağılımı."""
        if self.stats_df is None or "modality" not in self.stats_df.columns:
            return

        df  = self.stats_df
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("DICOM Meta Veri Analizi", fontsize=13, fontweight="bold")

        # Modalite dağılımı
        mod_counts = df["modality"].value_counts()
        axes[0].barh(mod_counts.index, mod_counts.values, color=self.PALETTE[:len(mod_counts)])
        axes[0].set_title("Modalite Dağılımı"); axes[0].set_xlabel("Sayı")

        # Üretici dağılımı
        mfr_counts = df["manufacturer"].value_counts().head(10)
        axes[1].barh(mfr_counts.index, mfr_counts.values, color=self.PALETTE[:len(mfr_counts)])
        axes[1].set_title("Üretici Dağılımı (İlk 10)")

        # kVp dağılımı
        kvp = df["kvp"].dropna()
        if len(kvp):
            axes[2].hist(kvp, bins=30, color=self.PALETTE[2], edgecolor="white")
            axes[2].axvline(kvp.median(), color="red", linestyle="--",
                            label=f"Medyan={kvp.median():.0f}")
            axes[2].set_title("kVp Dağılımı"); axes[2].set_xlabel("kV"); axes[2].legend()

        self._save_fig(fig, "dicom_metadata.png")

    # ── 10. Veri Kalitesi Raporu ─────────────────────────────────────────────

    def generate_quality_report(self) -> pd.DataFrame:
        """
        Veri kalitesi sorunlarını tespit eder ve raporlar:
        - Hasarlı / okunamayan dosyalar
        - Boyut aykırı değerler (outlier)
        - Yoğunluk aykırı değerleri
        - Sıfır varyans (sabit görüntüler)
        - Çok yüksek sıfır piksel oranı (tamamen siyah)
        """
        if self.stats_df is None:
            self.collect_image_statistics()

        df    = self.stats_df.copy()
        issues = pd.DataFrame()

        # Z-skor ile boyut aykırı değerleri
        for col in ["height", "width"]:
            z = np.abs(stats.zscore(df[col].dropna()))
            outlier_idx = df[col].dropna().index[z > 3.5]
            for i in outlier_idx:
                issues = pd.concat([issues, pd.DataFrame({
                    "path": [df.loc[i, "path"]],
                    "issue": [f"Boyut aykırı değer ({col}={df.loc[i, col]:.0f})"],
                    "severity": ["WARNING"],
                })])

        # Sabit görüntüler (std ≈ 0)
        zero_var = df[df["std"] < 1.0]
        for _, row in zero_var.iterrows():
            issues = pd.concat([issues, pd.DataFrame({
                "path": [row["path"]], "issue": ["Sabit görüntü (std≈0)"], "severity": ["ERROR"],
            })])

        # Tamamen siyah görüntüler
        black = df[df["zero_ratio"] > 0.98]
        for _, row in black.iterrows():
            issues = pd.concat([issues, pd.DataFrame({
                "path": [row["path"]], "issue": ["Tamamen siyah görüntü"], "severity": ["ERROR"],
            })])

        # Dosya boyutu sıfır
        tiny = df[df["file_kb"] < 1.0]
        for _, row in tiny.iterrows():
            issues = pd.concat([issues, pd.DataFrame({
                "path": [row["path"]], "issue": ["Dosya boyutu < 1 KB"], "severity": ["ERROR"],
            })])

        issues = issues.reset_index(drop=True)
        issues.to_csv(self.output_dir / "quality_report.csv", index=False)

        # DÜZELTME: hiç sorun bulunamazsa issues sütunsuz boş DataFrame olur;
        # issues.severity erişimi AttributeError fırlatıyordu. Sütun varlık
        # kontrolüyle güvenli sayım yapıldı.
        has_sev = "severity" in issues.columns
        n_error = int((issues["severity"] == "ERROR").sum()) if has_sev else 0
        n_warn  = int((issues["severity"] == "WARNING").sum()) if has_sev else 0
        logger.info(f"Kalite Raporu: {len(issues)} sorun tespit edildi "
                    f"(ERROR={n_error}, WARNING={n_warn})")

        # Görsel özet
        if len(issues):
            fig, ax = plt.subplots(figsize=(10, 4))
            issue_counts = issues.groupby(["issue", "severity"]).size().reset_index(name="count")
            colors       = ["#e74c3c" if s == "ERROR" else "#f39c12"
                            for s in issue_counts["severity"]]
            ax.barh(issue_counts["issue"], issue_counts["count"], color=colors)
            ax.set_title("Veri Kalitesi Sorunları", fontsize=12, fontweight="bold")
            ax.set_xlabel("Etkilenen Görüntü Sayısı")
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color="#e74c3c", label="ERROR"),
                                Patch(color="#f39c12", label="WARNING")])
            self._save_fig(fig, "quality_report.png")

        return issues

    # ── 11. Tam Analiz Çalıştırıcı ───────────────────────────────────────────

    def run_full_analysis(self, mask_paths: Optional[List[Path]] = None) -> Dict[str, Any]:
        """
        Tüm EDA adımlarını sırayla çalıştırır.

        Returns:
            Analiz sonuçlarını içeren sözlük
        """
        logger.info("=" * 60)
        logger.info("Keşifsel Veri Analizi başlıyor...")
        logger.info("=" * 60)

        results = {}

        logger.info("1/7 → İstatistik toplama")
        results["stats_df"] = self.collect_image_statistics()

        logger.info("2/7 → Etiket dağılımı")
        self.plot_label_distribution()

        logger.info("3/7 → Görüntü boyutu analizi")
        self.plot_image_size_analysis()

        logger.info("4/7 → Yoğunluk analizi")
        self.plot_intensity_analysis()

        logger.info("5/7 → Sınıf bazlı karşılaştırma")
        self.plot_classwise_intensity()

        logger.info("6/7 → Korelasyon matrisi")
        self.plot_correlation_matrix()

        if mask_paths:
            logger.info("6b/7 → Maske analizi")
            self.plot_mask_analysis(mask_paths)

        if self.fmt == "dcm":
            logger.info("6c/7 → DICOM meta veri analizi")
            self.plot_dicom_metadata()

        logger.info("7/7 → Veri kalitesi raporu")
        results["quality_report"] = self.generate_quality_report()

        logger.info(f"EDA tamamlandı. Çıktılar → {self.output_dir}")
        return results

    # ── Yardımcı ──────────────────────────────────────────────────────────────

    def _save_fig(self, fig: plt.Figure, filename: str) -> None:
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        gc.collect()
        logger.info(f"  ✓ {filename}")
