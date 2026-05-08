# MIS 584 — TwinBytes: Scalable Retail Demand Forecasting and Stockout Prediction

**Course:** MIS 584: Big Data Technologies  
**Team:** Ashleigh McNamara, Purva Pandit  
**Semester:** Spring 2026

---

## Project Overview

This project applies Apache Spark and Spark MLlib to the M5 Walmart Forecasting dataset to address two retail business problems:

1. **Demand Forecasting** — predicting daily unit sales per product per store using regression models
2. **Stockout Risk Prediction** — classifying whether a given day carries high stockout risk using simulated labels and classification models

The full pipeline processes 57.4 million rows entirely in Spark, covering 3,049 Walmart products across 10 stores in California, Texas, and Wisconsin from 2011 to 2016.

---

## Repository Structure

```
MIS584-TwinBytes-Final-Project/
│
├── Notebook_1_Data_Engineering.ipynb   — Spark pipeline, feature engineering
├── Notebook_2_EDA.ipynb                — Exploratory data analysis and plots
├── Notebook_3_Modeling.ipynb           — Model training and evaluation
├── Notebook_4_Evaluation.ipynb         — Results, business impact, scalability
│
├── plots/
│   ├── plot_sales_over_time.png
│   ├── plot_sales_by_category.png
│   ├── plot_zero_rate_by_category.png
│   ├── plot_snap_effect.png
│   ├── plot_correlation_heatmap.png
│   ├── plot_regression_comparison.png
│   ├── plot_classification_comparison.png
│   ├── plot_sensitivity_analysis.png
│   ├── plot_feature_importance.png
│   ├── plot_event_effect.png
│   ├── plot_price_buckets.png 
│   ├── plot_price_vs_sales.png
│   ├── plot_sales_by_month.png
│   ├── plot_sales_by_store.png 
│   └── plot_sales_by_dow.png 
└── README.md
```

---

## Dataset

Download the following files from Kaggle and upload them to your Colab session before running Notebook 1.

**Source:** https://www.kaggle.com/competitions/m5-forecasting-accuracy

**Files needed:**
- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

> The dataset is not included in this repository as it is third-party data hosted by Kaggle. A free Kaggle account is required to download it.

---

## Requirements

- Google Colab (free tier sufficient for EDA and evaluation)
- Google Drive mounted at `/content/drive/MyDrive/MIS584_Project/`
- PySpark — pre-installed in Colab, no installation needed
- Python libraries: matplotlib, seaborn, pandas — all pre-installed in Colab

---

## How to Run

Run all four notebooks in order. Each notebook depends on the previous one having been run in the same session or having saved its outputs to Google Drive.

---

### Notebook 1 — Data Engineering

**What it does:**
- Ingests all three raw CSV files into Spark DataFrames
- Melts wide sales data (30,490 rows × 1,913 columns) into long format (57.4 million rows × 20 columns)
- Joins calendar and pricing data so every row has full date context and price information
- Engineers the following features using Spark Window functions partitioned by (item_id, store_id):
  - Lag features: sales 7, 14, and 28 days prior
  - Rolling averages: 7, 30, and 90 day windows
  - Price delta: week over week price change
  - SNAP flag: whether SNAP benefits are active in that store's state
  - Event flag: whether a holiday or calendar event falls on that day
  - Temporal features: day of week, month
- Handles missing values — zeros are kept as real observations, null prices are forward filled
- Saves final cleaned dataset to Google Drive as CSV

**Estimated runtime:** 45 minutes

---

### Notebook 2 — Exploratory Data Analysis

**What it does:**
- Loads cleaned data from Google Drive
- Computes summary statistics including zero-sales rate across all 57.4 million rows
- Produces the following visualizations:
  - Total daily sales over the full 5 year period
  - Average sales by day of week and month
  - Total sales by store and state
  - Zero-sales rate by product category (FOODS, HOUSEHOLD, HOBBIES)
  - SNAP day vs non-SNAP day sales comparison
  - Event day vs non-event day sales comparison
  - Price vs demand scatter plot
  - Feature correlation heatmap
- Saves all plots to Google Drive

**Estimated runtime:** 20 minutes

---

### Notebook 3 — Model Development

**What it does:**
- Loads cleaned data from Google Drive
- Creates simulated stockout labels at three threshold multipliers (1.2×, 1.5×, 2.0×) based on each item's historical average sales computed from training data only
- Performs a time-aware train/test split — training on all data before January 1 2016, testing on 2016 data. No random splitting was used to prevent data leakage
- Trains and evaluates three regression models for demand forecasting:
  - Linear Regression (baseline)
  - Random Forest Regressor (50 trees, max depth 5)
  - Gradient Boosted Trees (10 iterations, max depth 5)
- Trains and evaluates three classification models for stockout risk:
  - Logistic Regression (baseline)
  - Random Forest Classifier (50 trees, max depth 5)
  - Gradient Boosted Trees Classifier (10 iterations, max depth 5)
- Runs sensitivity analysis comparing model performance across all three stockout thresholds
- Extracts and plots feature importances from the Random Forest regressor
- Saves all results as CSVs and plots to Google Drive

**Estimated runtime:** 1.5 to 2 hours

> **Note on sampling:** A 30% random sample of the training data was used at the MLlib model fitting stage due to Google Colab memory constraints. All data engineering, feature engineering, and label creation ran on the full 57.4 million rows. On a proper multi-node Spark cluster no sampling would be needed and the pipeline code would require no changes.

---

### Notebook 4 — Evaluation and Business Impact

**What it does:**
- Loads saved model results from Google Drive — no models are re-trained in this notebook
- Produces report-ready bar charts comparing RMSE and MAE across all three regression models
- Produces report-ready charts comparing AUC-ROC, F1, and Accuracy across all three classifiers
- Visualizes sensitivity analysis showing how AUC and % days flagged change across the three stockout thresholds
- Computes zero-sales rate segmentation by product category to quantify the intermittent demand problem
- Runs a business impact simulation estimating revenue that could be recovered through early stockout detection
- Documents pipeline runtime and discusses how the pipeline would scale to a full Spark cluster
- Saves all outputs and charts to Google Drive

**Estimated runtime:** 20 minutes

---

## Key Results

### Demand Forecasting — Regression

| Model | RMSE | MAE |
|---|---|---|
| Linear Regression | 2.0858 | 0.9956 |
| Random Forest | 2.2147 | 1.0133 |
| GBT | 2.2063 | 0.9972 |

Linear Regression achieved the lowest RMSE. The similar performance across all three models suggests the relationship between our engineered features and sales is largely linear at this scale. On average all models predict within approximately 1 unit of actual daily sales.

---

### Stockout Risk Prediction — Classification (1.5× threshold)

| Model | AUC-ROC | F1 | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.5903 | 0.6554 | 0.7547 |
| Random Forest | 0.6225 | 0.6497 | 0.7551 |
| GBT | 0.6762 | 0.6529 | 0.7558 |

GBT achieved the highest AUC-ROC of 0.676, making it the best model for separating stockout risk days from safe days. All models outperform a random baseline of 0.50, confirming our engineered features carry real predictive signal.

---

### Sensitivity Analysis — Stockout Threshold Comparison

| Threshold | AUC-ROC | F1 | % Days Flagged |
|---|---|---|---|
| 1.2× | 0.6903 | 0.5952 | 28.8% |
| 1.5× | 0.6158 | 0.6497 | 24.5% |
| 2.0× | 0.5837 | 0.7285 | 18.7% |

Results are consistent across all three thresholds, confirming findings are not an artifact of any single threshold choice. The 1.5× threshold was selected as the primary model as it represents the best balance between catching real risk and avoiding false alarms.

---

## Important Notes

**Stockout labels are simulated**
The M5 dataset contains no real inventory or stockout data. Labels were created based on whether daily sales exceeded a multiplier of each item's historical average sales, computed from training data only to prevent data leakage.

**Zero-sales days**
Between 60% and 77% of product-store-day observations have zero sales depending on category, reflecting intermittent demand. This is why AUC-ROC and F1 were used as primary metrics rather than accuracy, which would be misleading on imbalanced classes.

**Time-aware split**
All models were trained on data before January 1 2016 and tested on 2016 data. Random splitting was never used as it would allow the model to see future dates during training and artificially inflate results.

**Sampling note**
Sampling was applied only at the model training step due to Google Colab hardware limitations. The full 57.4 million row dataset was used for all data engineering and feature engineering steps.

---

## Individual Contributions

- **Ashleigh McNamara:** Data engineering pipeline (Notebook 1), Model development and training (Notebook 3), presentation slide design and content, final report writing  
- **Purva Pandit:** Exploratory data analysis (Notebook 2),, evaluation and business impact analysis (Notebook 4), presentation slide design and content, final report writing

---

## Acknowledgements

Dataset: M5 Forecasting — Accuracy, hosted by Kaggle  
https://www.kaggle.com/competitions/m5-forecasting-accuracy
