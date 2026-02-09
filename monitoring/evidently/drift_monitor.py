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
from evidently.pipeline.column_mapping import ColumnMapping  # pylint: disable=no-name-in-module

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class DriftMonitor:
    """ML drift monitoring service with Evidently."""

    def __init__(self, reports_dir: str = "monitoring/evidently/reports"):
        """
        Initialize drift monitor.

        Args:
            reports_dir: Directory to store HTML reports
        """
        self.reports_dir = Path(PROJECT_ROOT) / reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # F1PA dataset columns (features used for predictions)
        self.feature_columns = [
            'driver_number', 'circuit_key', 'st_speed', 'i1_speed', 'i2_speed',
            'temp', 'rhum', 'pres', 'lap_number', 'year'
        ]
        self.target_column = 'lap_duration'

        # Column mapping for Evidently 0.4.33
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
        Generate data drift report.

        Args:
            reference_data: Reference data (training)
            current_data: Current data (production)
            report_name: Report name (optional)

        Returns:
            Path to generated HTML report
        """
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"data_drift_{timestamp}"

        # Select relevant columns
        ref_data = reference_data[self.feature_columns + [self.target_column]].copy()
        curr_data = current_data[self.feature_columns + [self.target_column]].copy()

        print(f"  Données de référence: {len(ref_data)} tours")
        print(f"  Données actuelles: {len(curr_data)} tours")

        # Create Evidently report with DataDriftPreset
        report = Report(metrics=[
            DataDriftPreset()
        ])

        # Execute report
        print("  Exécution de l'analyse de drift...")
        report.run(
            reference_data=ref_data,
            current_data=curr_data,
            column_mapping=self.column_mapping
        )

        # Save HTML report
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
        Generate model performance report.

        Args:
            reference_data: Reference data with target
            current_data: Current data with target
            reference_predictions: Predictions on reference
            current_predictions: Predictions on current
            report_name: Report name

        Returns:
            Path to HTML report
        """
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"model_performance_{timestamp}"

        # Add predictions to DataFrames
        ref_data = reference_data[self.feature_columns + [self.target_column]].copy()
        ref_data['prediction'] = reference_predictions.values

        curr_data = current_data[self.feature_columns + [self.target_column]].copy()
        curr_data['prediction'] = current_predictions.values

        # Create report with RegressionPreset
        report = Report(metrics=[
            RegressionPreset()
        ])

        # Execute report
        report.run(
            reference_data=ref_data,
            current_data=curr_data,
            column_mapping=self.column_mapping
        )

        # Save report
        report_path = self.reports_dir / f"{report_name}.html"
        report.save_html(str(report_path))

        print(f"✅ Rapport de performance généré: {report_path}")
        return str(report_path)

    def list_reports(self) -> list:
        """List all generated reports."""
        reports = sorted(self.reports_dir.glob("*.html"), reverse=True)
        return [str(r) for r in reports]


def example_usage():
    """Example usage of DriftMonitor."""
    print("\n" + "=" * 80)
    print("F1PA - Evidently Drift Monitoring - Example")
    print("=" * 80 + "\n")

    from ml.config import DATA_DIR

    data_path = DATA_DIR / "processed" / "dataset_ml_lap_level_2023_2024_2025.csv"
    if not data_path.exists():
        # Try old name
        data_path_alt = DATA_DIR / "f1_processed_2023_2025.csv"
        if data_path_alt.exists():
            data_path = data_path_alt
        else:
            print(f"❌ Dataset not found: {data_path}")
            print("Run first: python -m data.fetch")
            return

    # Load data
    print("Loading dataset...")
    df = pd.read_csv(data_path)

    # Simulate reference/production split (70/30)
    split_idx = int(len(df) * 0.7)
    reference_data = df[:split_idx]
    current_data = df[split_idx:]

    print(f"  Reference: {len(reference_data)} laps")
    print(f"  Production: {len(current_data)} laps\n")

    # Initialize monitor
    monitor = DriftMonitor()

    # Generate drift report
    print("Generating drift report...")
    drift_report = monitor.generate_data_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        report_name="example_data_drift"
    )

    print("\n" + "=" * 80)
    print(f"Report available: {drift_report}")
    print(f"Open in browser to visualize drift")
    print("=" * 80)


if __name__ == '__main__':
    example_usage()
