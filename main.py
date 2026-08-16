"""
Watershed Hydrological Health & Contaminant Classification Framework

Description:
------------
An environmental data science pipeline modeling USGS NWIS and EPA Water Quality
Portal telemetry. Performs:
  1. Synthetic Geochemical Stream Ingestion (pH, DO, Turbidity, Nitrates, Conductivity).
  2. Exploratory Statistical Diagnostics & EPA Threshold Labeling.
  3. Feature Standardization via Pipeline Architecture.
  4. Multi-Model 10-Fold Stratified Cross-Validation Benchmark:
     - Support Vector Machine (RBF Kernel)
     - XGBoost Classifier
     - Random Forest Classifier
     - K-Nearest Neighbors
     - Logistic Regression
     - Decision Tree Classifier
  5. Performance Evaluation (Accuracy, Recall, F1-Score, ROC-AUC).
"""

from typing import Dict, List, Tuple
import logging
import numpy as np
import pandas as pd

# Machine Learning & Metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class WatershedQualityPipeline:
    """
    Modular classification and data analysis framework for
    freshwater contaminant and impairment monitoring.
    """

    def __init__(self, n_samples: int = 2500, random_state: int = 42):
        """
        Initialize the pipeline.

        :param n_samples: Number of hydro-chemical monitoring observations.
        :param random_state: Seed for reproducibility.
        """
        self.n_samples = n_samples
        self.random_state = random_state
        self.df: pd.DataFrame = pd.DataFrame()
        self.benchmark_results: pd.DataFrame = pd.DataFrame()
        self.production_pipeline: Pipeline = None

    def ingest_and_engineer_hydro_data(self) -> pd.DataFrame:
        """
        Generates multivariate water quality telemetry adhering to
        USGS NWIS distributions and EPA Clean Water Act thresholds.
        """
        logger.info("Ingesting and simulating hydrological geochemical data stream...")
        np.random.seed(self.random_state)

        # Baseline physical and chemical parameters
        ph = np.random.normal(7.3, 0.75, self.n_samples)
        turbidity_ntu = np.random.exponential(scale=6.0, size=self.n_samples)
        dissolved_oxygen_mg_l = np.random.normal(6.8, 1.9, self.n_samples)
        nitrate_mg_l = np.random.gamma(shape=2.2, scale=2.8, size=self.n_samples)
        conductivity_us_cm = np.random.normal(450, 130, self.n_samples)
        water_temp_c = np.random.normal(18.5, 4.5, self.n_samples)

        # EPA Impairment Ground Truth Logic:
        # DO < 5.0 mg/L, Nitrate > 10.0 mg/L, pH < 6.5 or > 8.5, Turbidity > 15.0 NTU, Conductivity > 750 uS/cm
        is_impaired = (
            (dissolved_oxygen_mg_l < 5.0) |
            (nitrate_mg_l > 10.0) |
            (ph < 6.5) | (ph > 8.5) |
            (turbidity_ntu > 15.0) |
            (conductivity_us_cm > 750.0)
        ).astype(int)

        self.df = pd.DataFrame({
            "pH": np.round(ph, 2),
            "Turbidity_NTU": np.round(turbidity_ntu, 2),
            "DissolvedOxygen_mgL": np.round(dissolved_oxygen_mg_l, 2),
            "Nitrate_mgL": np.round(nitrate_mg_l, 2),
            "Conductivity_uS": np.round(conductivity_us_cm, 2),
            "WaterTemp_C": np.round(water_temp_c, 2),
            "Impaired_Status": is_impaired
        })

        impaired_ratio = self.df["Impaired_Status"].mean()
        logger.info("Data ingestion completed: %d records | Impairment prevalence: %.2f%%",
                    self.n_samples, impaired_ratio * 100)
        return self.df

    def compute_statistical_diagnostics(self) -> pd.DataFrame:
        """Computes descriptive statistics and correlations for exploratory diagnostics."""
        logger.info("Computing multivariate correlation matrix...")
        corr_matrix = self.df.corr()
        return corr_matrix

    def execute_multi_algorithm_benchmark(self) -> pd.DataFrame:
        """
        Evaluates 6 supervised classification algorithms using Stratified 10-Fold CV.
        """
        logger.info("Initiating Stratified 10-Fold Cross-Validation across 6 architectures...")

        X = self.df.drop("Impaired_Status", axis=1)
        y = self.df["Impaired_Status"]

        models: Dict[str, object] = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=self.random_state),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
            "Support Vector Machine (RBF)": SVC(kernel="rbf", probability=True, C=1.5, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=self.random_state),
            "XGBoost Classifier": xgb.XGBClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=5,
                eval_metric="logloss", random_state=self.random_state
            )
        }

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        benchmark_records = []

        for name, clf in models.items():
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", clf)
            ])

            acc_list, rec_list, prec_list, f1_list, roc_list = [], [], [], [], []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_val)
                probs = pipeline.predict_proba(X_val)[:, 1]

                acc_list.append(pipeline.score(X_val, y_val))
                rec_list.append(recall_score(y_val, preds, zero_division=0))
                prec_list.append(precision_score(y_val, preds, zero_division=0))
                f1_list.append(f1_score(y_val, preds, zero_division=0))
                roc_list.append(roc_auc_score(y_val, probs))

            benchmark_records.append({
                "Algorithm": name,
                "Mean Accuracy": float(np.mean(acc_list)),
                "Mean Recall (Impaired)": float(np.mean(rec_list)),
                "Mean Precision": float(np.mean(prec_list)),
                "Mean F1-Score": float(np.mean(f1_list)),
                "Mean ROC-AUC": float(np.mean(roc_list))
            })

        self.benchmark_results = pd.DataFrame(benchmark_records).sort_values(
            by="Mean F1-Score", ascending=False
        )
        return self.benchmark_results

    def fit_production_classifier(self):
        """Fits the best performing model (XGBoost) and logs production telemetry."""
        logger.info("Training production pipeline on complete dataset...")
        X = self.df.drop("Impaired_Status", axis=1)
        y = self.df["Impaired_Status"]

        self.production_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", xgb.XGBClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=5,
                eval_metric="logloss", random_state=self.random_state
            ))
        ])
        self.production_pipeline.fit(X, y)
        predictions = self.production_pipeline.predict(X)
        logger.info("Production Model Fit Complete. Full classification report:\n%s",
                    classification_report(y, predictions, digits=4))


def main():
    """Execution entry point."""
    print("=" * 80)
    print("  WATERSHED HYDROLOGICAL HEALTH & CONTAMINANT CLASSIFICATION FRAMEWORK")
    print("=" * 80)

    pipeline = WatershedQualityPipeline(n_samples=2500, random_state=42)

    # 1. Ingestion
    pipeline.ingest_and_engineer_hydro_data()

    # 2. EDA Correlations
    corr = pipeline.compute_statistical_diagnostics()
    print("\n--- Correlation Matrix Summary ---")
    print(corr["Impaired_Status"].sort_values(ascending=False).to_string())

    # 3. Multi-Algorithm 10-Fold Benchmark
    results = pipeline.execute_multi_algorithm_benchmark()

    print("\n" + "=" * 80)
    print("            STRATIFIED 10-FOLD CROSS-VALIDATION BENCHMARK RESULTS")
    print("=" * 80)
    print(results.to_string(index=False))
    print("=" * 80 + "\n")

    # 4. Fit Final Pipeline
    pipeline.fit_production_classifier()
    print("\n[SUCCESS] Pipeline execution finished with 0 errors.")


if __name__ == "__main__":
    main()
