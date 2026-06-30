"""
Hybrid IAQ prediction script
============================

This script trains a hybrid model for the AirAware project:

- Random Forest Regressor for indoor PM2.5
- Random Forest Regressor for indoor PM10
- Gradient Boosting Regressor for indoor NO2

It then calculates a final weighted IAQ score using the same WHO-style
threshold approach used in the Ridge Regression notebook.

Run from the project root:
    python scripts/07_hybrid_rf_gb_iaq_score.py

Outputs are saved to:
    outputs/hybrid_rf_gb_iaq_score/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGETS = {
    "pm25": "indoor_pm25",
    "pm10": "indoor_pm10",
    "no2": "indoor_no2",
}

# Same scoring approach as the Ridge Regression notebook.
THRESHOLDS = {
    "pm25": 15,  # WHO-style 24-hour PM2.5 reference threshold, µg/m³
    "pm10": 45,  # WHO-style 24-hour PM10 reference threshold, µg/m³
    "no2": 25,  # WHO-style 24-hour NO2 reference threshold, µg/m³
}

WEIGHTS = {
    "pm25": 0.40,
    "pm10": 0.30,
    "no2": 0.30,
}


def get_project_root() -> Path:
    """Return the project root whether the script is run from root or scripts/."""
    current = Path.cwd().resolve()
    if current.name == "scripts":
        return current.parent
    return current


def classify_iaq(score: float) -> str:
    """Classify weighted IAQ score using the Ridge notebook categories."""
    if score < 1:
        return "Good"
    if score < 2:
        return "Moderate"
    if score < 3:
        return "Poor"
    return "Very Poor"


def calculate_iaq_score(
    df: pd.DataFrame,
    pm25_col: str,
    pm10_col: str,
    no2_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Add pollutant scores, weighted IAQ score and IAQ category to a dataframe."""
    scored = df.copy()

    scored[f"{prefix}_pm25_score"] = scored[pm25_col] / THRESHOLDS["pm25"]
    scored[f"{prefix}_pm10_score"] = scored[pm10_col] / THRESHOLDS["pm10"]
    scored[f"{prefix}_no2_score"] = scored[no2_col] / THRESHOLDS["no2"]

    scored[f"{prefix}_weighted_iaq_score"] = (
        WEIGHTS["pm25"] * scored[f"{prefix}_pm25_score"]
        + WEIGHTS["pm10"] * scored[f"{prefix}_pm10_score"]
        + WEIGHTS["no2"] * scored[f"{prefix}_no2_score"]
    )

    scored[f"{prefix}_iaq_category"] = scored[f"{prefix}_weighted_iaq_score"].apply(
        classify_iaq
    )
    return scored


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create X and identify numeric/categorical columns while avoiding leakage."""
    leakage_cols = [
        # final indoor targets
        "indoor_pm25",
        "indoor_pm10",
        "indoor_no2",
        # rate/exposure targets and target-derived labels
        "pm25_rate",
        "pm10_rate",
        "no2_rate",
        "pm25_exposure",
        "pm10_exposure",
        "no2_exposure",
        "log1p_pm25_rate",
        "log1p_pm10_rate",
        "log1p_no2_rate",
        "log1p_pm25_exposure",
        "log1p_pm10_exposure",
        "log1p_no2_exposure",
        "pm25_rate_event",
        "pm10_rate_event",
        "no2_rate_event",
        "pm25_rate_high_event",
        "pm10_rate_high_event",
        "no2_rate_high_event",
        # generated pollutant mechanics that are too close to the simulated target
        "cooking_pm25_raw",
        "cooking_pm10_raw",
        "cooking_no2_raw",
        "vent_pm25_multiplier",
        "vent_pm10_multiplier",
        "vent_no2_multiplier",
        "pm25_infiltration_factor",
        "pm10_infiltration_factor",
        "no2_infiltration_factor",
    ]

    datetime_cols = [
        "day",
        "start_time",
        "end_time",
        "start_time_local",
        "end_time_local",
    ]

    cols_to_drop = [c for c in leakage_cols + datetime_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()

    return X, numeric_features, categorical_features


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical data."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def get_model(pollutant: str):
    """Return the required model type for each pollutant."""
    if pollutant in {"pm25", "pm10"}:
        return RandomForestRegressor(
            n_estimators=25,
            max_depth=14,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    if pollutant == "no2":
        return GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unknown pollutant: {pollutant}")


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return common regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def main() -> None:
    project_root = get_project_root()
    data_path = project_root / "data" / "processed" / "air_quality_cleaned_featured.csv"
    out_dir = project_root / "outputs" / "hybrid_rf_gb_iaq_score"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Run notebooks 01 and 02 first, "
            "or check that the processed dataset exists."
        )

    df = pd.read_csv(data_path)

    missing_targets = [col for col in TARGETS.values() if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    X, numeric_features, categorical_features = build_feature_matrix(df)
    y_all = df[list(TARGETS.values())]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    metrics = []
    predictions = pd.DataFrame(index=X_test.index)

    for pollutant, target_col in TARGETS.items():
        model_type = "RandomForestRegressor" if pollutant in {"pm25", "pm10"} else "GradientBoostingRegressor"

        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
                ("model", get_model(pollutant)),
            ]
        )

        pipeline.fit(X_train, y_train[target_col])
        y_pred = pipeline.predict(X_test)
        y_pred = np.clip(y_pred, a_min=0, a_max=None)

        predictions[f"actual_{pollutant}"] = y_test[target_col]
        predictions[f"predicted_{pollutant}"] = y_pred

        row = {
            "pollutant": pollutant,
            "target_column": target_col,
            "model": model_type,
            **evaluate_model(y_test[target_col], y_pred),
        }
        metrics.append(row)

    predictions = calculate_iaq_score(
        predictions,
        pm25_col="actual_pm25",
        pm10_col="actual_pm10",
        no2_col="actual_no2",
        prefix="actual",
    )

    predictions = calculate_iaq_score(
        predictions,
        pm25_col="predicted_pm25",
        pm10_col="predicted_pm10",
        no2_col="predicted_no2",
        prefix="predicted",
    )

    score_metrics = evaluate_model(
        predictions["actual_weighted_iaq_score"],
        predictions["predicted_weighted_iaq_score"],
    )

    metrics_df = pd.DataFrame(metrics)
    score_metrics_df = pd.DataFrame(
        [
            {
                "target": "weighted_iaq_score",
                "model": "Hybrid RF/GB from pollutant predictions",
                **score_metrics,
            }
        ]
    )

    category_accuracy = (
        predictions["actual_iaq_category"] == predictions["predicted_iaq_category"]
    ).mean()

    category_summary = pd.DataFrame(
        [
            {
                "actual_vs_predicted_category_accuracy": category_accuracy,
                "test_rows": len(predictions),
            }
        ]
    )

    metrics_df.to_csv(out_dir / "hybrid_model_pollutant_metrics.csv", index=False)
    score_metrics_df.to_csv(out_dir / "hybrid_model_iaq_score_metrics.csv", index=False)
    category_summary.to_csv(out_dir / "hybrid_model_category_accuracy.csv", index=False)
    predictions.reset_index(names="source_row_id").to_csv(
        out_dir / "hybrid_model_iaq_predictions.csv",
        index=False,
    )

    config = {
        "input_data": str(data_path.relative_to(project_root)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "targets": TARGETS,
        "models": {
            "pm25": "RandomForestRegressor",
            "pm10": "RandomForestRegressor",
            "no2": "GradientBoostingRegressor",
        },
        "thresholds": THRESHOLDS,
        "weights": WEIGHTS,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    with open(out_dir / "hybrid_model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("\nHybrid model pollutant metrics")
    print(metrics_df.round(4).to_string(index=False))

    print("\nHybrid model IAQ score metrics")
    print(score_metrics_df.round(4).to_string(index=False))

    print("\nIAQ category accuracy")
    print(category_summary.round(4).to_string(index=False))

    print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
