# Customer Churn Prediction Using Machine Learning

This repository contains a CS280 / CS485 Introduction to Artificial Intelligence lab project for predicting customer churn using supervised machine learning.

The project includes a complete Jupyter notebook, a Streamlit dashboard, a Word report template, and supporting files for running and presenting the work.

## Project Objective

The goal is to predict whether a customer is likely to churn so that a business can identify at-risk customers and design proactive retention strategies.

This is a binary classification problem:

- `0`: customer did not churn
- `1`: customer churned

## Dataset

The dataset files are expected in the `dataset/` folder:

- `customer_churn_dataset-training-master.csv`
- `customer_churn_dataset-testing-master.csv`

Target variable:

- `Churn`

Predictive features:

- `Age`
- `Gender`
- `Tenure`
- `Usage Frequency`
- `Support Calls`
- `Payment Delay`
- `Subscription Type`
- `Contract Length`
- `Total Spend`
- `Last Interaction`

`CustomerID` is not used as a predictive feature because it is an identifier.

## Models Used

The notebook compares three distinct supervised machine learning models:

1. Logistic Regression
2. K-Nearest Neighbors
3. Random Forest Classifier

The workflow includes:

- dataset inspection,
- data cleaning,
- exploratory data analysis,
- preprocessing pipelines,
- train/test strategy,
- baseline model comparison,
- hyperparameter tuning,
- final held-out test evaluation,
- model selection,
- feature interpretability,
- inference on unseen examples.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

For churn prediction, Recall and F1-score are especially important because missing a customer who is likely to churn can be costly.

## Repository Files

| File | Description |
|---|---|
| `customer_churn_ml_project.ipynb` | Main executable machine learning notebook |
| `dashboard.py` | Streamlit dashboard application |
| `.streamlit/config.toml` | Streamlit theme configuration |
| `requirements.txt` | Python dependencies |
| `report_outline.md` | Short report outline |
| `customer_churn_prediction_report.docx` | Word report with screenshot placeholders |
| `.gitignore` | Git ignore rules |

The notebook can generate the following model artifacts after execution:

| Artifact | Description |
|---|---|
| `churn_prediction_model.pkl` | Saved final preprocessing + model pipeline |
| `model_metrics.json` | Saved final model metrics for dashboard use |
| `model_comparison.csv` | Model comparison table |

These generated artifacts are ignored by Git because they are reproducible outputs.

## Installation

Create and activate a virtual environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
```

If `pip` points to the wrong Python version, use:

```bash
python -m pip install -r requirements.txt
```

## How to Run the Notebook

Open and run:

```text
customer_churn_ml_project.ipynb
```

Run all cells from top to bottom. The notebook performs the full machine learning workflow and saves the final dashboard artifacts:

- `churn_prediction_model.pkl`
- `model_metrics.json`
- `model_comparison.csv`

## How to Run the Dashboard

After running the notebook and generating the model artifacts, start the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The dashboard includes:

- overview KPI cards,
- dataset explorer,
- churn insight visualizations,
- model performance comparison,
- single customer churn prediction,
- batch CSV prediction,
- downloadable prediction results.

If the model artifacts are missing, the dashboard will show a warning instead of crashing.

## Batch Prediction Format

The batch prediction CSV must include these columns:

```text
Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay,
Subscription Type, Contract Length, Total Spend, Last Interaction
```

`CustomerID` is optional. If included, it is preserved in the downloaded prediction results but ignored by the model.

The dashboard adds:

- `Predicted_Churn`
- `Churn_Probability`
- `Risk_Level`

## Report and Screenshots

The Word report is provided as:

```text
customer_churn_prediction_report.docx
```

It contains placeholder boxes showing exactly where to insert screenshots from:

- the notebook,
- the Streamlit dashboard,
- model comparison results,
- final evaluation plots,
- prediction pages.

Use the screenshot checklist at the end of the report to confirm all required visuals are included before submission.

## Academic Notes

This project avoids deep learning, AutoML, pretrained models, and hidden shortcuts. The modelling choices are intended to be explainable and defendable during a presentation.

The external testing dataset is used only for final held-out evaluation, not for training or tuning, to avoid data leakage.

## Limitations

- The dataset may not fully represent real customer behavior.
- Temporal churn effects are not explicitly modeled.
- No business-specific cost matrix is included.
- Synthetic examples are used only to demonstrate inference.

## Future Improvements

- Add cost-sensitive threshold tuning.
- Use time-based validation if temporal data becomes available.
- Add additional customer behavior features.
- Deploy the dashboard with scheduled model monitoring.
