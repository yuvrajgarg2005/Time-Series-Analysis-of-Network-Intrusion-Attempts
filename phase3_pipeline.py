from __future__ import annotations

import json
import math
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import tensorflow as tf
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
)

RANDOM_STATE = 42
CHUNK_SIZE = 250_000
MODEL_SAMPLE_TARGET = 500_000
LSTM_SAMPLE_TARGET = 200_000
SHAP_SAMPLE_TARGET = 5_000
WINDOW_SIZE = 7
TEST_SIZE = 0.2

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = OUTPUTS_DIR / "logs"

DATA_PATH = RAW_DATA_DIR / "NF-ToN-IoT-v3.csv"
FEATURE_INFO_PATH = RAW_DATA_DIR / "NetFlow_v3_Features.csv"

CLEANED_SAMPLE_PATH = PROCESSED_DATA_DIR / "nf_ton_iot_v3_cleaned_sample.csv"
DAILY_COUNTS_PATH = PROCESSED_DATA_DIR / "nf_ton_iot_v3_daily_counts.csv"
HOURLY_COUNTS_PATH = PROCESSED_DATA_DIR / "nf_ton_iot_v3_hourly_attack_counts.csv"
PROFILE_PATH = REPORTS_DIR / "phase3_profile.json"
METRICS_LOG_PATH = REPORTS_DIR / "phase3_metrics.txt"

PLOT_CLASS_DIST = PLOTS_DIR / "plot_class_dist.png"
PLOT_DAILY_TS = PLOTS_DIR / "plot_daily_ts.png"
PLOT_ATTACK_TYPES = PLOTS_DIR / "plot_attack_types.png"
PLOT_HEATMAP = PLOTS_DIR / "plot_heatmap.png"
PLOT_ROLLING_MEAN = PLOTS_DIR / "plot_rolling_mean.png"
PLOT_LSTM_LOSS = PLOTS_DIR / "plot_lstm_loss.png"
PLOT_LSTM_PRED = PLOTS_DIR / "plot_lstm_pred_vs_actual.png"
PLOT_PROPHET_FORECAST = PLOTS_DIR / "plot_prophet_forecast.png"
PLOT_PROPHET_COMPONENTS = PLOTS_DIR / "plot_prophet_components.png"
PLOT_SHAP_SUMMARY = PLOTS_DIR / "plot_shap_summary.png"
PLOT_SHAP_WATERFALL = PLOTS_DIR / "plot_shap_waterfall.png"
PLOT_SHAP_TEMPORAL = PLOTS_DIR / "plot_shap_temporal_heatmap.png"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.h5"
COMPARISON_TABLE_PATH = REPORTS_DIR / "comparison_table.csv"

ALL_PLOTS = [
    PLOT_CLASS_DIST,
    PLOT_DAILY_TS,
    PLOT_ATTACK_TYPES,
    PLOT_HEATMAP,
    PLOT_ROLLING_MEAN,
    PLOT_LSTM_LOSS,
    PLOT_LSTM_PRED,
    PLOT_PROPHET_FORECAST,
    PLOT_PROPHET_COMPONENTS,
    PLOT_SHAP_SUMMARY,
    PLOT_SHAP_WATERFALL,
    PLOT_SHAP_TEMPORAL,
]


def ensure_project_dirs() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        PLOTS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def set_seeds() -> None:
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)


def save_and_show(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def format_percentage(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value * 100:.2f}%"


def format_auc(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.4f}"


def format_rmse(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:,.2f}"


def print_metrics_block(name: str, metrics: dict[str, float | str | None]) -> str:
    lines = [f"{name}"]
    if metrics.get("precision") is not None:
        lines.append(f"Precision : {metrics['precision'] * 100:.2f}%")
    if metrics.get("recall") is not None:
        lines.append(f"Recall    : {metrics['recall'] * 100:.2f}%")
    if metrics.get("f1") is not None:
        lines.append(f"F1-Score  : {metrics['f1'] * 100:.2f}%")
    if metrics.get("auc") is not None:
        lines.append(f"AUC-ROC   : {metrics['auc']:.4f}")
    if metrics.get("rmse") is not None:
        lines.append(f"RMSE      : {metrics['rmse']:.2f}")
    if metrics.get("mae") is not None:
        lines.append(f"MAE       : {metrics['mae']:.2f}")
    if metrics.get("note"):
        lines.append(f"Note      : {metrics['note']}")
    if metrics.get("changepoints"):
        lines.append(f"Changepoints: {', '.join(metrics['changepoints'])}")
    block = "\n".join(lines)
    print(block)
    print()
    return block


def parse_feature_dictionary() -> dict[str, str]:
    feature_info = pd.read_csv(FEATURE_INFO_PATH)
    feature_info["Feature"] = feature_info["Feature"].astype(str).str.strip()
    feature_info["Description"] = feature_info["Description"].astype(str).str.strip()
    return dict(zip(feature_info["Feature"], feature_info["Description"]))


def load_cached_profile() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame] | None:
    if not (PROFILE_PATH.exists() and DAILY_COUNTS_PATH.exists() and HOURLY_COUNTS_PATH.exists()):
        return None
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    daily_df = pd.read_csv(DAILY_COUNTS_PATH, parse_dates=["ds"])
    hourly_df = pd.read_csv(HOURLY_COUNTS_PATH, parse_dates=["ds"])
    return profile, daily_df, hourly_df


def load_cached_sample() -> pd.DataFrame | None:
    if not CLEANED_SAMPLE_PATH.exists():
        return None
    return pd.read_csv(
        CLEANED_SAMPLE_PATH,
        parse_dates=["FLOW_START_DATETIME", "FLOW_END_DATETIME"],
    )


def first_pass_profile() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    total_rows = 0
    null_counts: pd.Series | None = None
    label_counts: Counter[int] = Counter()
    attack_counts: Counter[str] = Counter()
    daily_attack_counts: Counter[pd.Timestamp] = Counter()
    daily_total_counts: Counter[pd.Timestamp] = Counter()
    hourly_attack_counts: Counter[pd.Timestamp] = Counter()
    min_start = None
    max_end = None
    sorted_by_time = True
    prev_start = None
    columns: list[str] | None = None

    for chunk_id, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE), start=1):
        total_rows += len(chunk)
        columns = list(chunk.columns)
        chunk_nulls = chunk.isnull().sum()
        null_counts = chunk_nulls if null_counts is None else null_counts.add(chunk_nulls, fill_value=0)
        label_counts.update(chunk["Label"].value_counts().to_dict())
        attack_counts.update(chunk["Attack"].value_counts().to_dict())

        starts = chunk["FLOW_START_MILLISECONDS"]
        ends = chunk["FLOW_END_MILLISECONDS"]
        chunk_min = int(starts.min())
        chunk_max = int(ends.max())
        min_start = chunk_min if min_start is None else min(min_start, chunk_min)
        max_end = chunk_max if max_end is None else max(max_end, chunk_max)

        if prev_start is not None and int(starts.iloc[0]) < prev_start:
            sorted_by_time = False
        prev_start = int(starts.iloc[-1])

        timestamps = pd.to_datetime(starts, unit="ms")
        days = timestamps.dt.floor("D")
        hours = timestamps.dt.floor("h")
        daily_total_counts.update(days.value_counts().to_dict())
        daily_attack_counts.update(days[chunk["Label"] == 1].value_counts().to_dict())
        hourly_attack_counts.update(hours[chunk["Label"] == 1].value_counts().to_dict())

        print(f"Profiled chunk {chunk_id:03d} | rows processed: {total_rows:,}")

    if columns is None or null_counts is None or min_start is None or max_end is None:
        raise RuntimeError("Profiling failed because the dataset could not be read.")

    daily_index = pd.date_range(
        pd.to_datetime(min_start, unit="ms").floor("D"),
        pd.to_datetime(max_end, unit="ms").floor("D"),
        freq="D",
    )
    hourly_index = pd.date_range(
        pd.to_datetime(min_start, unit="ms").floor("h"),
        pd.to_datetime(max_end, unit="ms").floor("h"),
        freq="h",
    )

    daily_df = pd.DataFrame(index=daily_index)
    daily_df["total_flows"] = daily_df.index.map(lambda x: daily_total_counts.get(x, 0))
    daily_df["attack_count"] = daily_df.index.map(lambda x: daily_attack_counts.get(x, 0))
    daily_df = daily_df.reset_index().rename(columns={"index": "ds"})

    hourly_df = pd.DataFrame(index=hourly_index)
    hourly_df["attack_count"] = hourly_df.index.map(lambda x: hourly_attack_counts.get(x, 0))
    hourly_df = hourly_df.reset_index().rename(columns={"index": "ds"})

    benign_like = {"benign", "normal"}
    attack_only_counts = {k: v for k, v in attack_counts.items() if str(k).strip().lower() not in benign_like}

    profile = {
        "shape": [int(total_rows), len(columns)],
        "columns": columns,
        "null_counts": {key: int(value) for key, value in null_counts.to_dict().items()},
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "attack_counts": {str(key): int(value) for key, value in attack_counts.items()},
        "attack_only_counts": {str(key): int(value) for key, value in attack_only_counts.items()},
        "date_range": {
            "start": str(pd.to_datetime(min_start, unit="ms")),
            "end": str(pd.to_datetime(max_end, unit="ms")),
        },
        "sorted_by_flow_start": sorted_by_time,
    }

    return profile, daily_df, hourly_df


def second_pass_sample(total_rows: int) -> tuple[pd.DataFrame, int]:
    sample_fraction = min(1.0, MODEL_SAMPLE_TARGET / total_rows)
    sample_parts: list[pd.DataFrame] = []

    for chunk_id, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE), start=1):
        sampled = chunk.sample(frac=sample_fraction, random_state=RANDOM_STATE + chunk_id)
        sample_parts.append(sampled)
        print(
            f"Sampled chunk {chunk_id:03d} | sampled rows: {sum(len(part) for part in sample_parts):,}"
        )

    sample_df = pd.concat(sample_parts, ignore_index=True)
    if len(sample_df) > MODEL_SAMPLE_TARGET:
        sample_df = sample_df.sample(n=MODEL_SAMPLE_TARGET, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        sample_df = sample_df.reset_index(drop=True)

    duplicate_rows = int(sample_df.duplicated().sum())
    sample_df = sample_df.drop_duplicates().reset_index(drop=True)
    sample_df["FLOW_START_DATETIME"] = pd.to_datetime(sample_df["FLOW_START_MILLISECONDS"], unit="ms")
    sample_df["FLOW_END_DATETIME"] = pd.to_datetime(sample_df["FLOW_END_MILLISECONDS"], unit="ms")
    sample_df = sample_df.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)

    sample_df.to_csv(CLEANED_SAMPLE_PATH, index=False)
    return sample_df, duplicate_rows


def build_eda_plots(profile: dict[str, object], sample_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    label_counts = pd.Series(profile["label_counts"]).astype(int).sort_index()
    attack_counts = (
        pd.Series(profile["attack_only_counts"]).astype(int).sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=label_counts.index.astype(str), y=label_counts.values, palette="Set2", ax=ax)
    ax.set_title("Class Distribution (Full Dataset)")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    save_and_show(fig, PLOT_CLASS_DIST)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_df["ds"], daily_df["attack_count"], marker="o", linewidth=2, color="#1f77b4")
    ax.set_title("Daily Attack Count")
    ax.set_xlabel("Date")
    ax.set_ylabel("Attack Count")
    ax.tick_params(axis="x", rotation=30)
    save_and_show(fig, PLOT_DAILY_TS)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=attack_counts.values,
        y=attack_counts.index,
        palette="crest",
        ax=ax,
        orient="h",
    )
    ax.set_title("Attack Type Breakdown (Attacks Only)")
    ax.set_xlabel("Count")
    ax.set_ylabel("Attack Type")
    save_and_show(fig, PLOT_ATTACK_TYPES)

    heatmap_columns = [
        "IN_BYTES",
        "OUT_BYTES",
        "IN_PKTS",
        "OUT_PKTS",
        "FLOW_DURATION_MILLISECONDS",
        "L4_SRC_PORT",
        "L4_DST_PORT",
        "PROTOCOL",
        "L7_PROTO",
        "TCP_FLAGS",
        "SRC_TO_DST_AVG_THROUGHPUT",
        "DST_TO_SRC_AVG_THROUGHPUT",
        "Label",
    ]
    heatmap_columns = [col for col in heatmap_columns if col in sample_df.columns]
    corr = sample_df[heatmap_columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap (Sampled Data)")
    save_and_show(fig, PLOT_HEATMAP)

    rolling_df = daily_df.copy()
    rolling_df["rolling_mean_3d"] = rolling_df["attack_count"].rolling(window=3, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rolling_df["ds"], rolling_df["attack_count"], marker="o", label="Daily attack count")
    ax.plot(
        rolling_df["ds"],
        rolling_df["rolling_mean_3d"],
        linestyle="--",
        linewidth=2,
        label="3-day rolling mean",
    )
    ax.set_title("Daily Attack Count with Rolling Mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("Attack Count")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    save_and_show(fig, PLOT_ROLLING_MEAN)


def get_model_features(sample_df: pd.DataFrame) -> list[str]:
    excluded = {
        "Label",
        "Attack",
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "FLOW_START_MILLISECONDS",
        "FLOW_END_MILLISECONDS",
        "FLOW_START_DATETIME",
        "FLOW_END_DATETIME",
    }
    return [col for col in sample_df.columns if col not in excluded]


def create_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    start_target: int,
    end_target: int,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_seq = []
    y_seq = []
    for target_idx in range(start_target, end_target):
        X_seq.append(features[target_idx - window_size : target_idx])
        y_seq.append(labels[target_idx])
    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def run_lstm(sample_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float | str]:
    lstm_source = sample_df.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)
    if len(lstm_source) > LSTM_SAMPLE_TARGET:
        keep_idx = np.linspace(0, len(lstm_source) - 1, LSTM_SAMPLE_TARGET, dtype=int)
        lstm_source = lstm_source.iloc[np.unique(keep_idx)].reset_index(drop=True)

    X = lstm_source[feature_cols].astype(np.float32).to_numpy()
    y = lstm_source["Label"].astype(np.int8).to_numpy()

    split_idx = int(len(lstm_source) * (1 - TEST_SIZE))
    scaler = MinMaxScaler()
    scaler.fit(X[:split_idx])
    X_scaled = scaler.transform(X).astype(np.float32)

    X_train_seq, y_train_seq = create_sequences(
        X_scaled,
        y,
        start_target=WINDOW_SIZE,
        end_target=split_idx,
        window_size=WINDOW_SIZE,
    )
    X_test_seq, y_test_seq = create_sequences(
        X_scaled,
        y,
        start_target=split_idx,
        end_target=len(y),
        window_size=WINDOW_SIZE,
    )

    class_counts = np.bincount(y_train_seq.astype(int))
    class_weight = {
        0: len(y_train_seq) / (2 * class_counts[0]),
        1: len(y_train_seq) / (2 * class_counts[1]),
    }

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, len(feature_cols))),
            Dropout(0.2),
            LSTM(32),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")

    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_split=0.2,
        shuffle=False,
        epochs=10,
        batch_size=256,
        class_weight=class_weight,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=1,
    )

    probabilities = model.predict(X_test_seq, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_seq, predictions, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test_seq, probabilities)

    model.save(LSTM_MODEL_PATH)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history["loss"], label="Training loss")
    ax.plot(history.history["val_loss"], label="Validation loss")
    ax.set_title("LSTM Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    save_and_show(fig, PLOT_LSTM_LOSS)

    preview_length = min(200, len(probabilities))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(preview_length), y_test_seq[:preview_length], label="Actual label", linewidth=2)
    ax.plot(
        np.arange(preview_length),
        probabilities[:preview_length],
        label="Predicted probability",
        linewidth=2,
    )
    ax.set_title("LSTM Predicted vs Actual (First 200 Test Sequences)")
    ax.set_xlabel("Test Sequence Index")
    ax.set_ylabel("Probability / Label")
    ax.legend()
    save_and_show(fig, PLOT_LSTM_PRED)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "note": f"Sequence classifier trained on {len(lstm_source):,} time-ordered sampled flows.",
    }


def run_prophet(daily_df: pd.DataFrame, hourly_df: pd.DataFrame) -> dict[str, float | str | list[str]]:
    prophet_source = daily_df[["ds", "attack_count"]].rename(columns={"attack_count": "y"}).copy()
    freq = "D"
    periods = 7
    note = "Daily attack counts used directly."

    if len(prophet_source) < 20:
        prophet_source = hourly_df[["ds", "attack_count"]].rename(columns={"attack_count": "y"}).copy()
        freq = "h"
        periods = 24 * 7
        note = (
            "Daily series had only 7 observations, so Prophet was fitted on hourly attack counts "
            "to preserve enough temporal signal."
        )

    holdout = max(24 if freq == "h" else 2, int(len(prophet_source) * 0.2))
    holdout = min(holdout, len(prophet_source) - 5)
    train_df = prophet_source.iloc[:-holdout].copy()
    test_df = prophet_source.iloc[-holdout:].copy()

    prophet_eval = Prophet(
        daily_seasonality=freq == "h",
        weekly_seasonality=True,
        changepoint_prior_scale=0.1,
    )
    prophet_eval.fit(train_df)
    future_eval = prophet_eval.make_future_dataframe(periods=holdout, freq=freq)
    forecast_eval = prophet_eval.predict(future_eval)
    eval_frame = forecast_eval[["ds", "yhat"]].set_index("ds")
    aligned_pred = eval_frame.loc[test_df["ds"], "yhat"].to_numpy()

    rmse = math.sqrt(mean_squared_error(test_df["y"], aligned_pred))
    mae = mean_absolute_error(test_df["y"], aligned_pred)

    prophet_full = Prophet(
        daily_seasonality=freq == "h",
        weekly_seasonality=True,
        changepoint_prior_scale=0.1,
    )
    prophet_full.fit(prophet_source)
    future_full = prophet_full.make_future_dataframe(periods=periods, freq=freq)
    forecast_full = prophet_full.predict(future_full)

    fig_forecast = prophet_full.plot(forecast_full)
    fig_forecast.axes[0].set_title("Prophet Forecast")
    save_and_show(fig_forecast, PLOT_PROPHET_FORECAST)

    fig_components = prophet_full.plot_components(forecast_full)
    save_and_show(fig_components, PLOT_PROPHET_COMPONENTS)

    changepoints = sorted({str(pd.to_datetime(point).date()) for point in prophet_full.changepoints})

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "note": note,
        "changepoints": changepoints[:15],
    }


def run_xgboost_with_shap(sample_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float | str]:
    X = sample_df[feature_cols].astype(np.float32)
    y = sample_df["Label"].astype(int)
    timestamps = sample_df["FLOW_START_DATETIME"]

    X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(
        X,
        y,
        timestamps,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    count_neg = int((y_train == 0).sum())
    count_pos = int((y_train == 1).sum())
    scale_pos_weight = count_neg / max(count_pos, 1)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    shap_size = min(SHAP_SAMPLE_TARGET, len(X_test))
    shap_indices = X_test.sample(n=shap_size, random_state=RANDOM_STATE).index
    X_shap = X_test.loc[shap_indices]
    ts_shap = ts_test.loc[shap_indices]

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_shap)

    plt.figure(figsize=(12, 7))
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    fig = plt.gcf()
    save_and_show(fig, PLOT_SHAP_SUMMARY)

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation[0], max_display=15, show=False)
    fig = plt.gcf()
    save_and_show(fig, PLOT_SHAP_WATERFALL)

    shap_values = np.abs(explanation.values)
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df["day"] = ts_shap.dt.floor("D").to_numpy()
    temporal = shap_df.groupby("day").mean(numeric_only=True)
    top_features = temporal.mean(axis=0).sort_values(ascending=False).head(15).index

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(temporal[top_features].T, cmap="mako", ax=ax)
    ax.set_title("Temporal Mean Absolute SHAP by Day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Feature")
    save_and_show(fig, PLOT_SHAP_TEMPORAL)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "note": f"Trained on {len(sample_df):,} sampled flows with scale_pos_weight={scale_pos_weight:.4f}.",
    }


def build_comparison_table(
    prophet_metrics: dict[str, float | str],
    lstm_metrics: dict[str, float | str],
    xgb_metrics: dict[str, float | str],
) -> pd.DataFrame:
    comparison_df = pd.DataFrame(
        [
            {
                "Model": "ARIMA (1,1,1)",
                "Phase": 2,
                "Dataset": "Sample TON_IoT",
                "Precision": "-",
                "Recall": "-",
                "F1": "-",
                "AUC": "-",
                "RMSE": "22,214.00",
                "Notes": "Forecasting only",
            },
            {
                "Model": "Holt-Winters",
                "Phase": 2,
                "Dataset": "Sample TON_IoT",
                "Precision": "-",
                "Recall": "-",
                "F1": "-",
                "AUC": "-",
                "RMSE": "N/A",
                "Notes": "Simple ES fallback",
            },
            {
                "Model": "Random Forest",
                "Phase": 2,
                "Dataset": "Sample TON_IoT",
                "Precision": "99.40%",
                "Recall": "99.78%",
                "F1": "99.59%",
                "AUC": "0.9999",
                "RMSE": "-",
                "Notes": "Classifier",
            },
            {
                "Model": "Prophet",
                "Phase": 3,
                "Dataset": "NF-ToN-IoT-v3",
                "Precision": "-",
                "Recall": "-",
                "F1": "-",
                "AUC": "-",
                "RMSE": format_rmse(float(prophet_metrics["rmse"])),
                "Notes": str(prophet_metrics["note"]),
            },
            {
                "Model": "LSTM",
                "Phase": 3,
                "Dataset": "NF-ToN-IoT-v3",
                "Precision": format_percentage(float(lstm_metrics["precision"])),
                "Recall": format_percentage(float(lstm_metrics["recall"])),
                "F1": format_percentage(float(lstm_metrics["f1"])),
                "AUC": format_auc(float(lstm_metrics["auc"])),
                "RMSE": "-",
                "Notes": str(lstm_metrics["note"]),
            },
            {
                "Model": "XGBoost+SHAP",
                "Phase": 3,
                "Dataset": "NF-ToN-IoT-v3",
                "Precision": format_percentage(float(xgb_metrics["precision"])),
                "Recall": format_percentage(float(xgb_metrics["recall"])),
                "F1": format_percentage(float(xgb_metrics["f1"])),
                "AUC": format_auc(float(xgb_metrics["auc"])),
                "RMSE": "-",
                "Notes": "Best explainability",
            },
        ]
    )
    comparison_df.to_csv(COMPARISON_TABLE_PATH, index=False)
    return comparison_df


def write_metrics_log(blocks: list[str], profile: dict[str, object], duplicate_rows: int) -> None:
    summary_lines = [
        "Phase 3 Metrics and Notes",
        "=" * 80,
        "",
        f"Dataset shape              : {profile['shape'][0]:,} rows x {profile['shape'][1]} columns",
        f"Date range                 : {profile['date_range']['start']} to {profile['date_range']['end']}",
        f"Sorted by FLOW_START       : {profile['sorted_by_flow_start']}",
        f"Null columns with values   : {sum(1 for v in profile['null_counts'].values() if int(v) > 0)}",
        f"Duplicate rows in sample   : {duplicate_rows:,}",
        "",
        *blocks,
    ]
    METRICS_LOG_PATH.write_text("\n".join(summary_lines), encoding="utf-8")


def verify_outputs() -> None:
    expected = ALL_PLOTS + [LSTM_MODEL_PATH, COMPARISON_TABLE_PATH, CLEANED_SAMPLE_PATH]
    missing = [str(path.name) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected output files: {', '.join(missing)}")


def main() -> None:
    ensure_project_dirs()
    set_seeds()
    sns.set_theme(style="whitegrid")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    feature_dictionary = parse_feature_dictionary()
    print(f"Loaded {len(feature_dictionary):,} feature descriptions.")

    cached_profile = load_cached_profile()
    if cached_profile is None:
        profile, daily_df, hourly_df = first_pass_profile()
        daily_df.to_csv(DAILY_COUNTS_PATH, index=False)
        hourly_df.to_csv(HOURLY_COUNTS_PATH, index=False)
        PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    else:
        profile, daily_df, hourly_df = cached_profile
        print("Loaded cached profile and aggregated time series.")

    cached_sample = load_cached_sample()
    if cached_sample is None:
        sample_df, duplicate_rows = second_pass_sample(total_rows=int(profile["shape"][0]))
    else:
        sample_df = cached_sample
        duplicate_rows = int(sample_df.duplicated().sum())
        print("Loaded cached cleaned sample.")

    feature_cols = get_model_features(sample_df)

    build_eda_plots(profile, sample_df, daily_df)

    prophet_metrics = run_prophet(daily_df, hourly_df)
    lstm_metrics = run_lstm(sample_df, feature_cols)
    xgb_metrics = run_xgboost_with_shap(sample_df, feature_cols)

    metric_blocks = [
        print_metrics_block("Prophet", prophet_metrics),
        print_metrics_block("LSTM", lstm_metrics),
        print_metrics_block("XGBoost + SHAP", xgb_metrics),
    ]

    comparison_df = build_comparison_table(prophet_metrics, lstm_metrics, xgb_metrics)
    write_metrics_log(metric_blocks, profile, duplicate_rows)
    verify_outputs()

    print("Comparison Table")
    print(comparison_df.to_string(index=False))
    print()
    print("Saved cleaned sample:", CLEANED_SAMPLE_PATH.name)
    print("Saved daily counts  :", DAILY_COUNTS_PATH.name)
    print("Saved hourly counts :", HOURLY_COUNTS_PATH.name)
    print("Saved metrics log   :", METRICS_LOG_PATH.name)


if __name__ == "__main__":
    main()
