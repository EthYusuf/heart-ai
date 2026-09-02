# =============================================================================
# run_local.py — Yerel makine (CPU) çalıştırıcısı
# -----------------------------------------------------------------------------
# Kaggle GPU olmadan, sentetik test verisiyle pipeline'ı uçtan uca koşturur.
#   python run_local.py
# Gerçek deneyler için main.py + Kaggle GPU kullanılır; bu dosya yalnızca
# kurulumun ve pipeline bütününün doğru çalıştığını kanıtlar.
# =============================================================================

from config import LocalDebugConfig
from main import main

if __name__ == "__main__":
    main(LocalDebugConfig())
