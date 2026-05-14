from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix


st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
TRAIN_PATH = DATA_DIR / "customer_churn_dataset-training-master.csv"
TEST_PATH = DATA_DIR / "customer_churn_dataset-testing-master.csv"
MODEL_PATH = BASE_DIR / "churn_prediction_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"

TARGET_COLUMN = "Churn"
ID_COLUMN = "CustomerID"
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
]
NUMERIC_FEATURES = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]
CATEGORICAL_FEATURES = ["Gender", "Subscription Type", "Contract Length"]

COLOR_PRIMARY = "#0f766e"
COLOR_BLUE = "#1d4ed8"
COLOR_ORANGE = "#f97316"
COLOR_GREEN = "#16a34a"
COLOR_RED = "#dc2626"
PLOT_TEMPLATE = "plotly_white"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 2rem 2.2rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 100%);
            color: white;
            margin-bottom: 1.4rem;
            box-shadow: 0 18px 45px rgba(15, 118, 110, 0.18);
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(2rem, 4vw, 3.4rem);
            letter-spacing: 0;
            font-weight: 800;
        }
        .hero p {
            margin: 0.55rem 0 0;
            font-size: 1.15rem;
            opacity: 0.94;
        }
        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            margin: 1.4rem 0 0.75rem;
            color: #111827;
        }
        .kpi-card, .feature-card, .callout, .result-card {
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            background: #ffffff;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        .kpi-card:hover, .feature-card:hover {
            transform: translateY(-2px);
            transition: all 160ms ease;
            box-shadow: 0 16px 32px rgba(15, 23, 42, 0.08);
        }
        .kpi-label {
            font-size: 0.82rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 700;
        }
        .kpi-value {
            font-size: 1.9rem;
            color: #0f172a;
            font-weight: 850;
            margin-top: 0.25rem;
        }
        .feature-card h3 {
            font-size: 1.05rem;
            margin: 0 0 0.35rem;
            color: #0f172a;
        }
        .feature-card p, .callout p {
            margin: 0;
            color: #4b5563;
            line-height: 1.48;
        }
        .callout {
            background: #f8fafc;
            border-left: 5px solid #0f766e;
        }
        .result-high {
            border-color: rgba(220, 38, 38, 0.25);
            background: linear-gradient(135deg, #fff7ed 0%, #fff1f2 100%);
        }
        .result-low {
            border-color: rgba(22, 163, 74, 0.25);
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfeff 100%);
        }
        .risk-label {
            font-size: 1.6rem;
            font-weight: 850;
            margin-bottom: 0.25rem;
        }
        .sidebar-footer {
            color: #6b7280;
            font-size: 0.82rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e5e7eb;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 1rem;
            border-radius: 16px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(label: str, value: str, help_text: str | None = None) -> None:
    help_html = f"<div style='color:#6b7280;font-size:0.88rem;margin-top:0.35rem;'>{help_text}</div>" if help_text else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(text: str) -> None:
    st.markdown(f"""<div class="callout"><p>{text}</p></div>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_datasets() -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        train_df = clean_dataset(train_df)
        test_df = clean_dataset(test_df)
        return train_df, test_df, None
    except FileNotFoundError as exc:
        return None, None, f"Dataset file missing: {exc.filename}"
    except Exception as exc:
        return None, None, f"Could not load datasets: {exc}"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(how="all").copy()
    if TARGET_COLUMN in cleaned.columns:
        cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    return cleaned


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[Any | None, str | None]:
    if not MODEL_PATH.exists():
        return None, "Model artifact not found. Run the notebook first to create churn_prediction_model.pkl."
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as exc:
        return None, f"Could not load model artifact: {exc}"


@st.cache_data(show_spinner=False)
def load_metrics() -> tuple[dict[str, Any] | None, pd.DataFrame | None, str | None]:
    metrics = None
    comparison = None
    message = None

    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            if "metrics" in metrics:
                comparison = pd.DataFrame(metrics["metrics"])
        except Exception as exc:
            message = f"Could not read model_metrics.json: {exc}"

    if comparison is None and COMPARISON_PATH.exists():
        try:
            comparison = pd.read_csv(COMPARISON_PATH)
        except Exception as exc:
            message = f"Could not read model_comparison.csv: {exc}"

    if metrics is None and comparison is None and message is None:
        message = "Model metrics not found. Run the notebook first to generate dashboard artifacts."

    return metrics, comparison, message


def create_sidebar_navigation() -> str:
    st.sidebar.markdown("## Churn Intelligence")
    st.sidebar.caption("AI-powered retention analytics for supervised customer churn prediction.")

    pages = [
        "Overview",
        "Data Explorer",
        "Churn Insights",
        "Model Performance",
        "Predict Customer Churn",
        "Batch Prediction",
    ]
    page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

    train_df, _, data_error = load_datasets()
    model, model_error = load_model()
    if data_error:
        st.sidebar.warning("Dataset unavailable")
    else:
        st.sidebar.success(f"Dataset loaded: {len(train_df):,} training rows")

    if model_error:
        st.sidebar.warning("Model artifact pending")
    else:
        st.sidebar.success("Model artifact loaded")

    st.sidebar.markdown('<div class="sidebar-footer">CS280 / CS485 AI Lab Project</div>', unsafe_allow_html=True)
    return page


def model_count(comparison: pd.DataFrame | None) -> int:
    if comparison is None or "Model" not in comparison.columns:
        return 3
    return int(comparison["Model"].nunique())


def render_home_page(train_df: pd.DataFrame | None, comparison: pd.DataFrame | None) -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Customer Churn Intelligence Dashboard</h1>
            <p>Machine learning-driven customer retention analysis and prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if train_df is None:
        st.error("Training dataset is unavailable. Please confirm the dataset folder and CSV files exist.")
        return

    churn_rate = train_df[TARGET_COLUMN].mean()
    avg_tenure = train_df["Tenure"].mean()
    avg_spend = train_df["Total Spend"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        card("Training Customers", f"{len(train_df):,}")
    with c2:
        card("Overall Churn Rate", f"{churn_rate:.1%}")
    with c3:
        card("Average Tenure", f"{avg_tenure:.1f}")
    with c4:
        card("Average Spend", f"${avg_spend:,.0f}")
    with c5:
        card("Prediction Models", f"{model_count(comparison)}")

    st.markdown('<div class="section-title">What this dashboard does</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("Explore churn patterns", "Filter and inspect customer behavior across demographic and subscription segments."),
        ("Compare ML models", "Review Logistic Regression, KNN, and Random Forest performance side by side."),
        ("Predict individual risk", "Score a single customer profile with the selected trained pipeline."),
        ("Score uploaded batches", "Upload customer CSV files, generate risk scores, and download results."),
    ]
    for col, (title, body) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f'<div class="feature-card"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Project Objective</div>', unsafe_allow_html=True)
    callout(
        "The goal is to identify customers likely to churn so that businesses can design proactive retention strategies. "
        "Higher churn risk is a signal for retention attention, not a final business decision."
    )

    with st.expander("How to use the dashboard"):
        st.write(
            "Start with Data Explorer and Churn Insights to understand the dataset. "
            "Use Model Performance to review scientific evaluation. "
            "Use Predict Customer Churn or Batch Prediction when the notebook artifacts are available."
        )


def dataset_summary_cards(df: pd.DataFrame) -> None:
    missing_total = int(df.isna().sum().sum())
    numeric_count = len(df.select_dtypes(include=np.number).columns)
    categorical_count = len(df.select_dtypes(exclude=np.number).columns)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Columns", f"{df.shape[1]:,}")
    with c3:
        st.metric("Missing Values", f"{missing_total:,}")
    with c4:
        st.metric("Numeric / Categorical", f"{numeric_count} / {categorical_count}")


def render_dataset_overview(train_df: pd.DataFrame | None, test_df: pd.DataFrame | None) -> None:
    st.title("Data Explorer")
    if train_df is None or test_df is None:
        st.error("Datasets could not be loaded.")
        return

    selected_name = st.radio("Choose dataset", ["Training dataset", "Testing dataset"], horizontal=True)
    df = train_df if selected_name == "Training dataset" else test_df

    dataset_summary_cards(df)

    st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)
    preview_rows = st.slider("Rows to preview", min_value=5, max_value=100, value=10, step=5)
    st.dataframe(df.head(preview_rows), use_container_width=True)

    st.markdown('<div class="section-title">Feature Distribution</div>', unsafe_allow_html=True)
    feature = st.selectbox("Select a feature", [col for col in df.columns if col != ID_COLUMN])
    if pd.api.types.is_numeric_dtype(df[feature]):
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.histogram(df, x=feature, nbins=40, title=f"Distribution of {feature}", template=PLOT_TEMPLATE)
            fig.update_traces(marker_color=COLOR_PRIMARY)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(df[feature].describe().to_frame("value"), use_container_width=True)
    else:
        freq = df[feature].value_counts(dropna=False).reset_index()
        freq.columns = [feature, "Count"]
        fig = px.bar(freq, x=feature, y="Count", title=f"Category Frequency: {feature}", template=PLOT_TEMPLATE)
        fig.update_traces(marker_color=COLOR_BLUE)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Missing Values</div>', unsafe_allow_html=True)
    missing = df.isna().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    st.dataframe(missing, use_container_width=True)

    if selected_name == "Training dataset" and TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].map({0: "No Churn", 1: "Churn"}).value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        fig = px.pie(
            counts,
            names="Class",
            values="Count",
            hole=0.55,
            title="Training Target Distribution",
            color="Class",
            color_discrete_map={"No Churn": COLOR_GREEN, "Churn": COLOR_ORANGE},
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    c1, c2, c3 = st.columns(3)
    with c1:
        genders = st.multiselect("Gender", sorted(df["Gender"].dropna().unique()), default=sorted(df["Gender"].dropna().unique()))
    with c2:
        subscriptions = st.multiselect(
            "Subscription Type",
            sorted(df["Subscription Type"].dropna().unique()),
            default=sorted(df["Subscription Type"].dropna().unique()),
        )
    with c3:
        contracts = st.multiselect(
            "Contract Length",
            sorted(df["Contract Length"].dropna().unique()),
            default=sorted(df["Contract Length"].dropna().unique()),
        )

    return df[
        df["Gender"].isin(genders)
        & df["Subscription Type"].isin(subscriptions)
        & df["Contract Length"].isin(contracts)
    ].copy()


def churn_rate_chart(df: pd.DataFrame, column: str) -> None:
    grouped = df.groupby(column, dropna=False)[TARGET_COLUMN].mean().reset_index(name="Churn Rate")
    grouped = grouped.sort_values("Churn Rate", ascending=False)
    fig = px.bar(
        grouped,
        x=column,
        y="Churn Rate",
        text=grouped["Churn Rate"].map(lambda x: f"{x:.1%}"),
        title=f"Observed Churn Rate by {column}",
        template=PLOT_TEMPLATE,
    )
    fig.update_traces(marker_color=COLOR_PRIMARY, textposition="outside")
    fig.update_yaxes(tickformat=".0%", range=[0, min(1, max(0.05, grouped["Churn Rate"].max() * 1.2))])
    st.plotly_chart(fig, use_container_width=True)


def render_churn_insights(train_df: pd.DataFrame | None) -> None:
    st.title("Churn Insights")
    if train_df is None:
        st.error("Training dataset could not be loaded.")
        return

    st.write("Filter the training data to explore observed churn associations across segments.")
    filtered = apply_filters(train_df)
    if filtered.empty:
        st.warning("No rows match the selected filters.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Filtered Customers", f"{len(filtered):,}")
    with c2:
        st.metric("Observed Churn Rate", f"{filtered[TARGET_COLUMN].mean():.1%}")
    with c3:
        st.metric("Churned Customers", f"{int(filtered[TARGET_COLUMN].sum()):,}")

    st.markdown('<div class="section-title">Churn Rate by Segment</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        churn_rate_chart(filtered, "Gender")
    with b:
        churn_rate_chart(filtered, "Subscription Type")
    with c:
        churn_rate_chart(filtered, "Contract Length")

    st.markdown('<div class="section-title">Average Customer Behavior by Churn Status</div>', unsafe_allow_html=True)
    behavior_cols = ["Support Calls", "Payment Delay", "Tenure", "Total Spend"]
    grouped = filtered.groupby(TARGET_COLUMN)[behavior_cols].mean().reset_index()
    grouped[TARGET_COLUMN] = grouped[TARGET_COLUMN].map({0: "No Churn", 1: "Churn"})
    melted = grouped.melt(id_vars=TARGET_COLUMN, var_name="Feature", value_name="Average Value")
    fig = px.bar(
        melted,
        x="Feature",
        y="Average Value",
        color=TARGET_COLUMN,
        barmode="group",
        title="Average Numerical Features by Churn Status",
        color_discrete_map={"No Churn": COLOR_GREEN, "Churn": COLOR_ORANGE},
        template=PLOT_TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)

    support_means = filtered.groupby(TARGET_COLUMN)["Support Calls"].mean()
    if 0 in support_means and 1 in support_means:
        if support_means.loc[1] > support_means.loc[0]:
            callout("Customers with more support calls show a higher observed churn association in this filtered subset.")
        else:
            callout("In this filtered subset, average support calls are not higher for churned customers.")

    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = filtered[NUMERIC_FEATURES + [TARGET_COLUMN]].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Numeric Feature Correlations",
        template=PLOT_TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)


def standardize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "F1-score": "F1",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "Roc Auc": "ROC-AUC",
    }
    out = df.rename(columns=rename_map).copy()
    expected = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    for col in expected:
        if col not in out.columns:
            out[col] = np.nan
    return out[expected]


def render_model_performance(
    metrics: dict[str, Any] | None,
    comparison: pd.DataFrame | None,
    metrics_message: str | None,
    model: Any | None,
    test_df: pd.DataFrame | None,
) -> None:
    st.title("Model Comparison and Final Selection")

    if comparison is None:
        st.warning(metrics_message or "Model metrics are not available. Execute the notebook first to generate artifacts.")
        return

    comparison = standardize_metric_columns(comparison)
    st.dataframe(comparison.style.format({col: "{:.4f}" for col in comparison.columns if col != "Model"}), use_container_width=True)

    metric_cols = ["F1", "Recall", "ROC-AUC"]
    chart_df = comparison.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")
    fig = px.bar(
        chart_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Model Comparison: F1, Recall, and ROC-AUC",
        template=PLOT_TEMPLATE,
        color_discrete_sequence=[COLOR_PRIMARY, COLOR_ORANGE, COLOR_BLUE],
    )
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    selected_name = metrics.get("selected_model") if metrics else comparison.sort_values("F1", ascending=False).iloc[0]["Model"]
    selected_row = comparison.loc[comparison["Model"] == selected_name]
    if selected_row.empty:
        selected_row = comparison.sort_values("F1", ascending=False).head(1)
        selected_name = selected_row.iloc[0]["Model"]

    st.markdown('<div class="section-title">Final Selected Model</div>', unsafe_allow_html=True)
    callout(
        (metrics or {}).get(
            "selection_reason",
            "Selected using held-out performance with emphasis on F1, Recall, and ROC-AUC for churn detection.",
        )
    )
    r = selected_row.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Selected Model", selected_name)
    with c2:
        st.metric("F1-score", f"{r['F1']:.3f}")
    with c3:
        st.metric("Recall", f"{r['Recall']:.3f}")
    with c4:
        st.metric("ROC-AUC", f"{r['ROC-AUC']:.3f}")

    st.info(
        "For churn prediction, Recall and F1 may be more important than raw Accuracy because missing a likely churner can be costly."
    )

    if model is not None and test_df is not None and TARGET_COLUMN in test_df.columns:
        with st.expander("Confusion matrix for saved final model on the provided testing dataset"):
            try:
                X_test = test_df[FEATURE_COLUMNS]
                y_test = test_df[TARGET_COLUMN]
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["No Churn", "Churn"],
                    y=["No Churn", "Churn"],
                    color_continuous_scale="Blues",
                    title="Confusion Matrix",
                    template=PLOT_TEMPLATE,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not compute confusion matrix from saved model: {exc}")

    with st.expander("About the model"):
        st.write(
            "The saved model is expected to be a full scikit-learn pipeline containing preprocessing and the selected classifier. "
            "It accepts raw customer feature columns directly and returns churn predictions."
        )


def categorical_options(df: pd.DataFrame | None, column: str, fallback: list[str]) -> list[str]:
    if df is None or column not in df.columns:
        return fallback
    values = sorted([str(v) for v in df[column].dropna().unique()])
    return values or fallback


def prepare_prediction_input(values: dict[str, Any]) -> pd.DataFrame:
    row = {col: values[col] for col in FEATURE_COLUMNS}
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_prediction_scores(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    prediction = model.predict(X)
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)[:, 1]
    return prediction, probability


def format_prediction_result(predicted_class: int, probability: float | None) -> tuple[str, str, str, float]:
    score = float(probability) if probability is not None else float(predicted_class)
    high_risk = predicted_class == 1
    label = "High Risk of Churn" if high_risk else "Low Risk of Churn"
    css_class = "result-high" if high_risk else "result-low"
    interpretation = (
        "This customer profile is predicted to have elevated churn risk and may warrant retention attention."
        if high_risk
        else "This customer profile appears less likely to churn based on the trained model."
    )
    return label, css_class, interpretation, score


def render_prediction_result(predicted_class: int, probability: float | None) -> None:
    label, css_class, interpretation, score = format_prediction_result(predicted_class, probability)
    color = COLOR_RED if predicted_class == 1 else COLOR_GREEN
    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="risk-label" style="color:{color};">{label}</div>
            <p style="margin:0;color:#374151;">Predicted churn probability: <strong>{score:.1%}</strong></p>
            <p style="margin-top:0.65rem;color:#4b5563;">{interpretation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(score, 0.0), 1.0))


def render_single_prediction(model: Any | None, model_error: str | None, train_df: pd.DataFrame | None) -> None:
    st.title("Predict Customer Churn")
    st.write("Enter a customer profile and score it using the saved final machine learning pipeline.")

    if model is None:
        st.error(model_error or "Model artifact is missing. Run the notebook first to create churn_prediction_model.pkl.")
        return

    with st.form("single_prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=1, max_value=120, value=35)
            tenure = st.number_input("Tenure", min_value=0, max_value=120, value=12)
            support_calls = st.number_input("Support Calls", min_value=0, max_value=100, value=3)
            gender = st.selectbox("Gender", categorical_options(train_df, "Gender", ["Female", "Male"]))
        with c2:
            usage_frequency = st.number_input("Usage Frequency", min_value=0, max_value=100, value=15)
            payment_delay = st.number_input("Payment Delay", min_value=0, max_value=365, value=10)
            subscription = st.selectbox(
                "Subscription Type",
                categorical_options(train_df, "Subscription Type", ["Basic", "Standard", "Premium"]),
            )
        with c3:
            total_spend = st.number_input("Total Spend", min_value=0.0, max_value=100000.0, value=500.0, step=10.0)
            last_interaction = st.number_input("Last Interaction", min_value=0, max_value=365, value=7)
            contract = st.selectbox(
                "Contract Length",
                categorical_options(train_df, "Contract Length", ["Monthly", "Quarterly", "Annual"]),
            )

        submitted = st.form_submit_button("Predict Churn Risk", type="primary", use_container_width=True)

    if submitted:
        values = {
            "Age": age,
            "Gender": gender,
            "Tenure": tenure,
            "Usage Frequency": usage_frequency,
            "Support Calls": support_calls,
            "Payment Delay": payment_delay,
            "Subscription Type": subscription,
            "Contract Length": contract,
            "Total Spend": total_spend,
            "Last Interaction": last_interaction,
        }
        input_df = prepare_prediction_input(values)
        try:
            prediction, probability = get_prediction_scores(model, input_df)
            render_prediction_result(int(prediction[0]), None if probability is None else float(probability[0]))
            st.markdown('<div class="section-title">Input Profile</div>', unsafe_allow_html=True)
            st.dataframe(input_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")


def sample_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [24, "Female", 4, 3, 8, 25, "Basic", "Monthly", 180, 28],
            [42, "Male", 48, 27, 1, 2, "Premium", "Annual", 980, 3],
            [61, "Female", 18, 12, 4, 12, "Standard", "Quarterly", 520, 14],
        ],
        columns=FEATURE_COLUMNS,
    )


def validate_batch_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    allowed = set(FEATURE_COLUMNS + [ID_COLUMN, TARGET_COLUMN])
    unexpected = [col for col in df.columns if col not in allowed]
    return missing, unexpected


def render_batch_prediction(model: Any | None, model_error: str | None) -> None:
    st.title("Batch Prediction")
    st.write("Upload a CSV file, validate customer columns, generate churn risk scores, and download the results.")

    template = sample_template()
    st.download_button(
        "Download Sample CSV Template",
        data=template.to_csv(index=False),
        file_name="churn_prediction_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if model is None:
        st.error(model_error or "Model artifact is missing. Run the notebook first to create churn_prediction_model.pkl.")
        return

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        uploaded_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        return

    st.markdown('<div class="section-title">Uploaded Preview</div>', unsafe_allow_html=True)
    st.dataframe(uploaded_df.head(20), use_container_width=True)

    missing, unexpected = validate_batch_columns(uploaded_df)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return
    if unexpected:
        st.warning(f"Unexpected columns will be ignored for prediction: {', '.join(unexpected)}")

    try:
        prediction_input = uploaded_df[FEATURE_COLUMNS].copy()
        for col in NUMERIC_FEATURES:
            prediction_input[col] = pd.to_numeric(prediction_input[col], errors="coerce")

        predictions, probabilities = get_prediction_scores(model, prediction_input)
        results = uploaded_df.copy()
        results["Predicted_Churn"] = predictions.astype(int)
        if probabilities is not None:
            results["Churn_Probability"] = np.round(probabilities, 4)
        else:
            results["Churn_Probability"] = results["Predicted_Churn"].astype(float)
        results["Risk_Level"] = np.where(results["Predicted_Churn"] == 1, "High Risk", "Low Risk")

        churn_count = int(results["Predicted_Churn"].sum())
        churn_pct = churn_count / len(results)
        avg_probability = float(results["Churn_Probability"].mean())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Uploaded Customers", f"{len(results):,}")
        with c2:
            st.metric("Predicted to Churn", f"{churn_count:,}")
        with c3:
            st.metric("Predicted Churn %", f"{churn_pct:.1%}")
        with c4:
            st.metric("Average Probability", f"{avg_probability:.1%}")

        risk_counts = results["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Count",
            hole=0.5,
            title="Batch Risk Distribution",
            color="Risk Level",
            color_discrete_map={"High Risk": COLOR_RED, "Low Risk": COLOR_GREEN},
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
        st.dataframe(results, use_container_width=True)
        st.download_button(
            "Download Prediction Results",
            data=results.to_csv(index=False),
            file_name="customer_churn_prediction_results.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Batch prediction could not be completed: {exc}")


def main() -> None:
    inject_css()
    train_df, test_df, data_error = load_datasets()
    model, model_error = load_model()
    metrics, comparison, metrics_message = load_metrics()
    page = create_sidebar_navigation()

    if data_error:
        st.warning(data_error)

    if page == "Overview":
        render_home_page(train_df, comparison)
    elif page == "Data Explorer":
        render_dataset_overview(train_df, test_df)
    elif page == "Churn Insights":
        render_churn_insights(train_df)
    elif page == "Model Performance":
        render_model_performance(metrics, comparison, metrics_message, model, test_df)
    elif page == "Predict Customer Churn":
        render_single_prediction(model, model_error, train_df)
    elif page == "Batch Prediction":
        render_batch_prediction(model, model_error)


if __name__ == "__main__":
    main()
