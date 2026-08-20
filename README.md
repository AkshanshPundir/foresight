# Project FORESIGHT — Demand & Inventory Intelligence

**Zidio Development | Data Science & Analytics Internship**

## Business problem
NorthBay Living needs a data-driven way to forecast product demand, identify stockout/overstock risk, and prioritize inventory actions.

## Dataset
The project uses the supplied synthetic retail extracts and a 100,000-transaction sample from the larger sales transaction file. The sample contains 11 fields including date, receipt, store, SKU, customer, quantity, price, revenue, channel, discount and promotion.

## Deliverables
- Reproducible data pipeline
- EDA and data-quality analysis
- Weekly SKU-level demand forecast
- Seasonal-naive baseline and ML comparison
- Stockout/overstock risk scoring
- Streamlit planning dashboard
- Executive report and presentation

## Model
A global Random Forest regression model uses lagged demand, rolling statistics, calendar features and product economics. A seasonal-naive lag-4 baseline is used for comparison. The evaluation uses a time-based holdout and WAPE; no random split is used.

## Important data limitation
The original Kaggle sales file is larger than the upload limit, so development uses a reproducible 100,000-row sample. For production-grade final results, rerun the pipeline on the full transaction file.

## Run locally
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Key results from the sample
- Transactions: 100,000
- SKUs: 5,000
- Stores: 30
- Revenue: ₹109,994,351
- Model WAPE: 81.87%
- Baseline WAPE: 103.71%
- Reorder-now SKUs: 109
- Markdown/clear SKUs: 1,170

## Structure
`data/` raw and processed data · `outputs/` model results and charts · `models/` trained model · `dashboard/` Streamlit app · `reports/` submission documents.
