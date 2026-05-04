# Time-Series Analysis of Network Intrusion Attempts using Explainable AI

Phase 3 implementation for **Intelligent Model Design with AI (216)** using the **NF-ToN-IoT-v3** NetFlow benchmark.

## Group Members
- Atindriya Tyagi — `S24BCAU0005`
- Yuvraj Garg — `S24BCAU00058`
- Ansh Garg — `S24BCAU0011`

## What This Project Includes
- Chunked preprocessing and profiling for the full NF-ToN-IoT-v3 CSV
- EDA plots for class balance, attack types, daily activity, correlations, and rolling trend
- Prophet forecasting with changepoint analysis
- LSTM sequential attack classification
- XGBoost classification with SHAP explanations
- Final comparison table across Phase 2 and Phase 3 models

## Project Structure
```text
CBCA216/
├── src/
│   └── phase3_pipeline.py
├── data/
│   ├── raw/
│   │   ├── NF-ToN-IoT-v3.csv              # local only, ignored on GitHub
│   │   └── NetFlow_v3_Features.csv
│   └── processed/
│       ├── nf_ton_iot_v3_cleaned_sample.csv   # local only, ignored on GitHub
│       ├── nf_ton_iot_v3_daily_counts.csv
│       └── nf_ton_iot_v3_hourly_attack_counts.csv
├── outputs/
│   ├── plots/
│   ├── models/
│   ├── reports/
│   └── logs/
├── docs/
│   └── DEMO_GUIDE.md
├── metadata/
│   └── original_package/
├── README.md
├── requirements.txt
└── .gitignore
```

## Important GitHub Note
The following files are intentionally **not committed** because they are too large for a normal GitHub upload:
- `data/raw/NF-ToN-IoT-v3.csv`
- `data/processed/nf_ton_iot_v3_cleaned_sample.csv`

## Key Outputs
- Plots: `outputs/plots/`
- Saved LSTM model: `outputs/models/lstm_model.h5`
- Metrics summary: `outputs/reports/phase3_metrics.txt`
- Comparison table: `outputs/reports/comparison_table.csv`

## How To Run
From the project root:

```powershell
.\.venv310\Scripts\python.exe -u .\src\phase3_pipeline.py
```

## Demo Tip
For viva/demo, open files in this order:
1. `src/phase3_pipeline.py`
2. `outputs/reports/phase3_metrics.txt`
3. `outputs/reports/comparison_table.csv`
4. `outputs/plots/` images
5. `outputs/models/lstm_model.h5` if asked
