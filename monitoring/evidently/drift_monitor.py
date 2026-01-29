"""
F1PA - Evidently Drift Monitoring

Generate drift reports to monitor ML model performance in production.
Compatible with Evidently 0.4.33
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.pipeline.column_mapping import ColumnMapping

# Ajouter le root du projet au path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class DriftMonitor:
    """Service de monitoring du drift ML avec Evidently."""

    def __init__(self, reports_dir: str = "monitoring/evidently/reports"):
        """
        Initialise le monitor de drift.

        Args:
            reports_dir: Dossier pour stocker les rapports HTML
        """
        self.reports_dir = Path(PROJECT_ROOT) / reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Colonnes du dataset F1PA (features utilisées for les predictions)
        self.feature_columns = [
            'driver_number', 'circuit_key', 'st_speed', 'i1_speed', 'i2_speed',
            'temp', 'rhum', 'pres', 'lap_number', 'year'
        ]
        self.target_column = 'lap_duration'

        # Column mapping pour Evidently 0.4.33
        self.column_mapping = ColumnMapping(
            target=self.target_column,
            numerical_features=self.feature_columns,
            prediction='prediction'
        )

    def generate_data_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        report_name: str = None
    ) -> str:
        """
        Génère un rapport de drift des données.

        Args:
            reference_data: Données de référence (training)
            current_data: Données actuelles (production)
            report_name: Nom du rapport (optionnel)

        Returns:
            Chemin du rapport HTML généré
        """
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"data_drift_{timestamp}"

        # Sélectionner les colonnes pertinentes
        ref_data = reference_data[self.feature_columns + [self.target_column]].copy()
        curr_data = current_data[self.feature_columns + [self.target_column]].copy()

        print(f"  Données de référence: {len(ref_data)} tours")
        print(f"  Données actuelles: {len(curr_data)} tours")

        # Creater le report Evidently with DataDriftPreset
        report = Report(metrics=[
            DataDriftPreset()
        ])

        # Executer le report
        print("  Exécution de l'analyse de drift...")
        report.run(
            reference_data=ref_data,
            current_data=curr_data,
            column_mapping=self.column_mapping
        )

        # Sauvegarder le rapport HTML
        report_path = self.reports_dir / f"{report_name}.html"
        report.save_html(str(report_path))

        print(f"✅ Rapport de drift généré: {report_path}")
        return str(report_path)

    def generate_model_performance_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        reference_predictions: pd.Series,
        current_predictions: pd.Series,
        report_name: str = None
    ) -> str:
        """
        Génère un rapport de performance du modèle.

        Args:
            reference_data: Données de référence avec target
            current_data: Données actuelles avec target
            reference_predictions: Prédictions sur référence
            current_predictions: Prédictions sur current
            report_name: Nom du rapport

        Returns:
            Chemin du rapport HTML
        """
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"model_performance_{timestamp}"

        # Ajouter les predictions aux DataFrames
        ref_data = reference_data[self.feature_columns + [self.target_column]].copy()
        ref_data['prediction'] = reference_predictions.values

        curr_data = current_data[self.feature_columns + [self.target_column]].copy()
        curr_data['prediction'] = current_predictions.values

        # Creater le report with RegresifonPreset
        report = Report(metrics=[
            RegressionPreset()
        ])

        # Executer le report
        report.run(
            reference_data=ref_data,
            current_data=curr_data,
            column_mapping=self.column_mapping
        )

        # Sauvegarder
        report_path = self.reports_dir / f"{report_name}.html"
        report.save_html(str(report_path))

        print(f"✅ Rapport de performance généré: {report_path}")
        return str(report_path)

    def list_reports(self) -> list:
        """Liste tous les reports générés."""
        reports = sorted(self.reports_dir.glob("*.html"), reverse=True)
        return [str(r) for r in reports]


def example_usage():
    """Exemple d'utilisation du DriftMonitor."""
    print("\n" + "=" * 80)
    print("F1PA - Evidently Drift Monitoring - Exemple")
    print("=" * 80 + "\n")

    from ml.config import DATA_DIR

    data_path = DATA_DIR / "processed" / "dataset_ml_lap_level_2023_2024_2025.csv"
    if not data_path.exists():
        # Essayer l'ancien nom
        data_path_alt = DATA_DIR / "f1_processed_2023_2025.csv"
        if data_path_alt.exists():
            data_path = data_path_alt
        else:
            print(f"❌ Dataset introuvable: {data_path}")
            print("Executer d'abord: python -m data.fetch")
            return

    # Loadr les data
    print("Chargement du dataset...")
    df = pd.read_csv(data_path)

    # Simuler une split référence/production (70/30)
    split_idx = int(len(df) * 0.7)
    reference_data = df[:split_idx]
    current_data = df[split_idx:]

    print(f"  Référence: {len(reference_data)} tours")
    print(f"  Production: {len(current_data)} tours\n")

    # Initialiser le monitor
    monitor = DriftMonitor()

    # Générer report de drift
    print("Génération du report de drift...")
    drift_report = monitor.generate_data_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        report_name="example_data_drift"
    )

    print("\n" + "=" * 80)
    print(f"Rapport disponible: {drift_report}")
    print(f"Ouvrir dans le navigateur pour visualiser le drift")
    print("=" * 80)


if __name__ == '__main__':
    example_usage()
