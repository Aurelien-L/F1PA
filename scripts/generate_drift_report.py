"""
Script for générer un report Evidently de test
"""
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.evidently.drift_monitor import DriftMonitor
from ml.config import DATA_DIR

def main():
    print("\n" + "=" * 80)
    print("F1PA - Génération d'un rapport Evidently de test")
    print("=" * 80 + "\n")

    # Charger les données
    data_path = DATA_DIR / "processed" / "dataset_ml_lap_level_2023_2024_2025.csv"

    if not data_path.exists():
        # Essayer l'ancien nom
        data_path_alt = DATA_DIR / "f1_processed_2023_2025.csv"
        if data_path_alt.exists():
            data_path = data_path_alt
        else:
            print(f"❌ Dataset introuvable: {data_path}")
            print("Exécuter d'abord: python -m data.fetch")
            return 1

    print(f"Chargement du dataset depuis {data_path}...")
    df = pd.read_csv(data_path)
    print(f"✅ {len(df)} tours chargés\n")

    # Split 70/30 pour simuler référence vs production
    split_idx = int(len(df) * 0.7)
    reference_data = df[:split_idx]
    current_data = df[split_idx:]

    print(f"Données de référence (training): {len(reference_data)} tours")
    print(f"Données actuelles (production): {len(current_data)} tours\n")

    # Initialiser le monitor
    monitor = DriftMonitor()

    # Générer le rapport de drift
    print("Génération du rapport de drift des données...")
    report_path = monitor.generate_data_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        report_name="test_data_drift"
    )

    print("\n" + "=" * 80)
    print("✅ RAPPORT GÉNÉRÉ AVEC SUCCÈS")
    print("=" * 80)
    print(f"\nRapport disponible:")
    print(f"  {report_path}")
    print(f"\nPour le visualiser:")
    print(f"  - Ouvrir directement dans un navigateur")
    print(f"  - Ou via l'API: http://localhost:8000/monitoring/drift/latest")
    print("=" * 80 + "\n")

    return 0

if __name__ == '__main__':
    sys.exit(main())
