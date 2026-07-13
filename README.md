# 📊 Retail Demand Forecasting & Inventory Intelligence System

## Overview

This project was developed as part of the **Xylofy AI Internship (Week 3 & Week 4)**. The objective was to build an end-to-end retail demand forecasting system capable of predicting future sales, detecting anomalies, segmenting products based on demand patterns, and presenting insights through an interactive Streamlit dashboard.

---

## Features

- 📈 Exploratory Data Analysis (EDA)
- ⏳ Time Series Analysis & Decomposition
- 🤖 Sales Forecasting using:
  - SARIMA
  - Prophet
  - XGBoost (Best Performing Model)
- 🚨 Sales Anomaly Detection
  - Isolation Forest
  - Z-Score Detection
- 📦 Product Demand Segmentation using K-Means Clustering
- 🌐 Interactive Streamlit Dashboard

---

## Dashboard Features

### 📊 Sales Overview
- KPI Cards
- Total Sales by Year
- Monthly Sales Trend
- Region & Category Filters

### 📈 Forecast Explorer
- Category/Region Selection
- Forecast Horizon (1–3 Months)
- XGBoost Forecast Visualization
- Model Performance Metrics (MAE, RMSE, MAPE)

### 🚨 Anomaly Report
- Weekly Sales Trend
- Isolation Forest Anomalies
- Z-Score Anomalies
- Detected Anomaly Tables

### 📦 Product Demand Segments
- K-Means Cluster Visualization
- PCA Scatter Plot
- Demand Segment Table
- Recommended Stocking Strategies

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Statsmodels
- Prophet
- Plotly
- Streamlit
- Google Colab

---

## Model Performance

| Model | MAE | RMSE | MAPE |
|------|------:|------:|------:|
| SARIMA | 18,652.22 | 24,824.40 | 17.85% |
| Prophet | 20,250.79 | 22,318.41 | 21.86% |
| **XGBoost** | **14,763.81** | **18,337.41** | **14.48%** |

**Best Model:** XGBoost

---

## Live Streamlit Dashboard

🔗 https://retail-demand-forecasting-owmkr58fwkszreq3xasnwf.streamlit.app/

---

## Repository Structure

```
Retail-Demand-Forecasting/
│
├── app.py
├── requirements.txt
├── Retail_Demand_Forecasting.ipynb
├── summary.pdf
│
├── train.csv
├── weekly_sales.csv
├── monthly_sales.csv
├── cluster_data.csv
├── iso_anomalies.csv
├── zscore_anomalies.csv
│
├── xgboost_model.pkl
└── kmeans_model.pkl
```

---

## Business Outcomes

- Improved demand forecasting accuracy using XGBoost.
- Identified seasonal sales patterns and unusual demand spikes.
- Segmented products into demand groups for optimized inventory planning.
- Developed an interactive dashboard to support business decision-making.

---

## Author

**Vivek Sagar Arora**

B.Tech Computer Science & Engineering (AI/ML)

Bennett University

Xylofy AI Internship Project
