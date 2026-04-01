from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "artifacts" / "models" / "best_model.pkl"
MODELS_DIR = ROOT / "artifacts" / "models"
THRESHOLD_PATH = ROOT / "reports" / "selected_threshold.json"
METRICS_PATH = ROOT / "reports" / "metrics_summary.csv"
RECOMMEND_PATH = ROOT / "reports" / "model_recommendations.json"

st.set_page_config(page_title="ChurnStrike Intelligence", page_icon="📉", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .subtle-note {
        color: #475569;
        font-size: 0.95rem;
        margin-top: -6px;
        margin-bottom: 8px;
    }
    .section-title {
        margin-top: 10px;
        margin-bottom: 8px;
        font-size: 1.15rem;
        color: #0f172a;
        font-weight: 700;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 10px 12px;
    }
    .legend-box {
        background: #f8fafc;
        border-left: 4px solid #0f766e;
        border-radius: 8px;
        padding: 10px 12px;
        margin-top: 6px;
        margin-bottom: 10px;
        color: #334155;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_EXPLANATIONS = {
    "logistic_regression": {
        "description": "Fast and interpretable linear baseline model.",
        "best_for": "Simple baseline and explainability.",
    },
    "random_forest": {
        "description": "Bagged decision trees with strong generalization.",
        "best_for": "Balanced performance and robust tabular baseline.",
    },
    "random_forest_tuned": {
        "description": "Random Forest optimized with hyperparameter tuning.",
        "best_for": "High churn recall/F1 for retention campaigns.",
    },
    "xgboost": {
        "description": "Gradient boosting model for strong ranking performance.",
        "best_for": "High accuracy and AUC on tabular data.",
    },
    "xgboost_tuned": {
        "description": "XGBoost after parameter optimization.",
        "best_for": "Accuracy/AUC-focused deployment.",
    },
    "lightgbm": {
        "description": "Fast gradient boosting framework.",
        "best_for": "Efficient training and competitive recall.",
    },
    "lightgbm_tuned": {
        "description": "Tuned LightGBM with improved recall/F1 tradeoff.",
        "best_for": "Near-best recall with faster runtime.",
    },
    "catboost": {
        "description": "Boosting method that performs well on categorical-heavy data.",
        "best_for": "Strong accuracy and ranking quality.",
    },
    "catboost_tuned": {
        "description": "Tuned CatBoost model.",
        "best_for": "Best overall accuracy in latest run.",
    },
    "soft_voting_ensemble": {
        "description": "Probability average of multiple strong models.",
        "best_for": "Stable blended prediction behavior.",
    },
}


def load_artifacts():
    if not MODEL_PATH.exists():
        st.error("Model not found. Generate artifacts by running: PYTHONPATH=src python run_pipeline.py")
        st.stop()

    recommendations = {}
    if RECOMMEND_PATH.exists():
        with open(RECOMMEND_PATH, "r", encoding="utf-8") as f:
            recommendations = json.load(f)

    metrics_df = None
    if METRICS_PATH.exists():
        metrics_df = pd.read_csv(METRICS_PATH)

    default_threshold = 0.5
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            threshold_json = json.load(f)
            default_threshold = float(threshold_json.get("threshold", 0.5))

    return recommendations, metrics_df, default_threshold


def render_overview_page(metrics_df: pd.DataFrame | None, recommendations: dict):
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #0f766e 0%, #0369a1 100%);
            padding: 16px 20px;
            border-radius: 12px;
            color: white;
            margin-bottom: 12px;
        ">
            <h1 style="margin: 0; font-size: 30px;">ChurnStrike Intelligence</h1>
            <p style="margin: 6px 0 0 0; font-size: 15px; opacity: 0.95;">
                End-to-end ML system with model comparison, objective-driven selection, and live prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="subtle-note">Project Overview</p>', unsafe_allow_html=True)

    if recommendations:
        st.markdown('<div class="section-title">Best Model By Purpose</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Accuracy", recommendations.get("best_for_accuracy", "n/a"))
        c2.metric("Best Recall", recommendations.get("best_for_recall", "n/a"))
        c3.metric("Best F1", recommendations.get("best_for_f1", "n/a"))
        c4.metric("Best AUC", recommendations.get("best_for_auc", "n/a"))

    st.markdown('<div class="section-title">Model Types and Use Cases</div>', unsafe_allow_html=True)
    model_names = list(MODEL_EXPLANATIONS.keys())
    for i in range(0, len(model_names), 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx >= len(model_names):
                continue
            model_name = model_names[idx]
            details = MODEL_EXPLANATIONS[model_name]
            with cols[j]:
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #d1d5db;
                        border-radius: 10px;
                        padding: 12px;
                        margin-bottom: 10px;
                        background-color: #f8fafc;
                    ">
                        <h4 style="margin: 0 0 6px 0; color: #0f172a;">{model_name}</h4>
                        <p style="margin: 0 0 6px 0; color: #334155; font-size: 14px;">{details['description']}</p>
                        <p style="margin: 0; color: #0f766e; font-size: 13px;"><b>Best for:</b> {details['best_for']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if metrics_df is not None and not metrics_df.empty:
        st.markdown('<div class="section-title">Model Metrics</div>', unsafe_allow_html=True)
        display_df = metrics_df.copy()
        st.dataframe(display_df, use_container_width=True)

        metric_option = st.selectbox(
            "Choose metric to compare models",
            options=["accuracy", "precision_churn", "recall_churn", "f1_churn", "roc_auc"],
            index=3,
        )
        chart_df = display_df[["model", metric_option]].set_index("model").sort_values(metric_option, ascending=False)
        st.bar_chart(chart_df)


def render_prediction_page(metrics_df: pd.DataFrame | None, recommendations: dict, default_threshold: float):
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #0ea5e9 0%, #0f766e 100%);
            padding: 14px 18px;
            border-radius: 12px;
            color: white;
            margin-bottom: 12px;
        ">
            <h2 style="margin: 0; font-size: 26px;">ChurnStrike Intelligence</h2>
            <p style="margin: 6px 0 0 0; font-size: 14px; opacity: 0.95;">Live scoring with objective-based model and threshold strategy.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_model_files = sorted([p for p in MODELS_DIR.glob("*.pkl") if p.name != "best_model.pkl"])
    available_model_names = [p.stem for p in available_model_files]

    default_model_name = recommendations.get("best_for_f1") if recommendations else ""
    if default_model_name not in available_model_names and available_model_names:
        default_model_name = available_model_names[0]

    st.sidebar.header("Prediction Strategy")
    selection_mode = st.sidebar.radio(
        "Model selection mode",
        options=["Auto by objective", "Manual model select"],
        index=0,
    )

    mode = st.sidebar.radio(
        "Choose objective",
        options=["High Recall", "Balanced", "High Precision", "Custom"],
        index=1,
    )

    objective_to_reco_key = {
        "High Recall": "best_for_recall",
        "Balanced": "best_for_f1",
        "High Precision": "best_for_accuracy",
        "Custom": "best_for_f1",
    }

    selected_model_name = default_model_name
    if selection_mode == "Auto by objective" and recommendations:
        selected_model_name = recommendations.get(objective_to_reco_key[mode], default_model_name)
    elif selection_mode == "Manual model select" and available_model_names:
        selected_model_name = st.sidebar.selectbox(
            "Select model",
            available_model_names,
            index=available_model_names.index(default_model_name),
        )

    model_path = MODELS_DIR / f"{selected_model_name}.pkl" if selected_model_name else MODEL_PATH
    if model_path.exists():
        model = joblib.load(model_path)
    else:
        model = joblib.load(MODEL_PATH)
        selected_model_name = "best_model"

    mode_thresholds = {
        "High Recall": 0.35,
        "Balanced": float(default_threshold),
        "High Precision": 0.65,
    }

    if mode == "Custom":
        threshold = st.sidebar.slider(
            "Decision threshold",
            min_value=0.1,
            max_value=0.9,
            value=float(default_threshold),
            step=0.01,
        )
    else:
        threshold = mode_thresholds[mode]
        st.sidebar.success(f"Mode threshold applied: {threshold:.2f}")

    st.sidebar.caption(
        "Lower threshold catches more churners (higher recall). "
        "Higher threshold reduces false positives (higher precision)."
    )

    st.sidebar.markdown(
        '<div class="legend-box"><b>Quick Guide</b><br/>'
        'High Recall: catch more churners<br/>'
        'Balanced: best F1 tradeoff<br/>'
        'High Precision: fewer false alarms</div>',
        unsafe_allow_html=True,
    )

    if recommendations:
        st.sidebar.markdown("### Best Models By Purpose")
        st.sidebar.write(f"Accuracy: {recommendations.get('best_for_accuracy', 'n/a')}")
        st.sidebar.write(f"Recall: {recommendations.get('best_for_recall', 'n/a')}")
        st.sidebar.write(f"F1: {recommendations.get('best_for_f1', 'n/a')}")
        st.sidebar.write(f"AUC: {recommendations.get('best_for_auc', 'n/a')}")

    if metrics_df is not None and not metrics_df.empty:
        st.markdown('<div class="section-title">Model Performance Summary</div>', unsafe_allow_html=True)
        st.dataframe(metrics_df, use_container_width=True)

    if not DATA_PATH.exists():
        st.error("Dataset not found in data/raw.")
        st.stop()

    raw_df = pd.read_csv(DATA_PATH)
    raw_df["TotalCharges"] = pd.to_numeric(raw_df["TotalCharges"], errors="coerce")
    feature_df = raw_df.drop(columns=["customerID", "Churn"]).copy()

    st.markdown('<div class="section-title">Enter Customer Details</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    input_data = {}
    for idx, col_name in enumerate(feature_df.columns):
        target_col = [col1, col2, col3][idx % 3]
        with target_col:
            if pd.api.types.is_numeric_dtype(feature_df[col_name]):
                min_v = float(feature_df[col_name].min())
                max_v = float(feature_df[col_name].max())
                median_v = float(feature_df[col_name].median())
                if col_name == "SeniorCitizen":
                    input_data[col_name] = st.selectbox(col_name, [0, 1], index=int(median_v >= 0.5))
                else:
                    input_data[col_name] = st.number_input(
                        col_name,
                        min_value=min_v,
                        max_value=max_v,
                        value=median_v,
                    )
            else:
                values = sorted(feature_df[col_name].dropna().unique().tolist())
                mode_v = feature_df[col_name].mode().iloc[0] if not feature_df[col_name].mode().empty else values[0]
                default_idx = values.index(mode_v) if mode_v in values else 0
                input_data[col_name] = st.selectbox(col_name, values, index=default_idx)

    if st.button("Predict Churn", type="primary"):
        sample = pd.DataFrame([input_data])
        churn_prob = float(model.predict_proba(sample)[:, 1][0])
        churn_pred = int(churn_prob >= threshold)

        c1, c2, c3 = st.columns(3)
        c1.metric("Selected Model", selected_model_name)
        c2.metric("Threshold", f"{threshold:.2f}")
        c3.metric("Churn Probability", f"{churn_prob:.2%}")
        st.metric("Prediction", "Churn" if churn_pred == 1 else "No Churn")

        if churn_pred == 1:
            st.warning("High retention risk. Recommend proactive intervention.")
        else:
            st.success("Customer appears relatively stable at this threshold.")


recommendations, metrics_df, default_threshold = load_artifacts()
page = st.sidebar.radio("Page", ["Project Overview", "Predict Churn"], index=0)

if page == "Project Overview":
    render_overview_page(metrics_df, recommendations)
else:
    render_prediction_page(metrics_df, recommendations, default_threshold)
