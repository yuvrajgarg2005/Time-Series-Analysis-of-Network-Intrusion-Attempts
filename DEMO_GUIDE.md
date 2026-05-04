# 2-Minute Demo Script

## Demo Order
1. `src/phase3_pipeline.py`
2. `outputs/reports/phase3_metrics.txt`
3. `outputs/reports/comparison_table.csv`
4. EDA plots in `outputs/plots/`
5. `outputs/models/lstm_model.h5` if asked

## What To Say

### 1. Open `src/phase3_pipeline.py`
Say:

> This is our main Phase 3 pipeline. It handles full-dataset profiling in chunks, generates the cleaned modeling sample, builds the EDA plots, runs Prophet, trains the LSTM, trains XGBoost, generates SHAP explanations, and saves the final comparison table.

### 2. Open `outputs/reports/phase3_metrics.txt`
Say:

> These are the final recorded metrics from the run. The full dataset had 27.52 million rows and 55 columns, with no null values. Prophet was evaluated using RMSE and MAE, while LSTM and XGBoost were evaluated using classification metrics.

Read the important numbers:
- Prophet: `RMSE 449144.17`, `MAE 439727.69`
- LSTM: `Precision 83.58%`, `Recall 97.69%`, `F1 90.09%`, `AUC 0.8656`
- XGBoost + SHAP: `Precision 95.13%`, `Recall 91.69%`, `F1 93.38%`, `AUC 0.9861`

### 3. Open `outputs/reports/comparison_table.csv`
Say:

> This file compares the Phase 2 and Phase 3 models. It shows that Phase 3 introduced advanced models, and among them XGBoost with SHAP gave the best balance of strong performance and explainability.

### 4. Open the EDA and result plots
Recommended order:
- `plot_class_dist.png`
- `plot_attack_types.png`
- `plot_daily_ts.png`
- `plot_rolling_mean.png`
- `plot_prophet_forecast.png`
- `plot_lstm_loss.png`
- `plot_lstm_pred_vs_actual.png`
- `plot_shap_summary.png`
- `plot_shap_waterfall.png`
- `plot_shap_temporal_heatmap.png`

Say:

> These plots show the complete analysis flow: first the dataset behavior, then forecasting, then sequential learning, and finally explainable AI. The SHAP plots are especially important because they show which NetFlow features influenced the intrusion predictions and how their importance changed over time.

### 5. If asked whether the model was saved
Open `outputs/models/lstm_model.h5`

Say:

> This confirms that the trained LSTM model was saved successfully after training.

## Strong Closing Line
> Overall, our project demonstrates that intrusion attempts can be studied as both a time-series problem and a classification problem, with XGBoost plus SHAP giving the best Phase 3 explainable result.
