import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "loan_data.csv"
CLEAN_DATA_PATH = DATA_DIR / "loan_data_clean.csv"
MODELS_DIR = BASE_DIR / "models"
APP_VERSION = "1.2.0"

MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
}


st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(CLEAN_DATA_PATH)


@st.cache_data
def load_metadata() -> dict:
    with (MODELS_DIR / "metadata.json").open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_model(model_key: str):
    return joblib.load(MODELS_DIR / f"{model_key}.pkl")


@st.cache_resource
def load_scaler():
    return joblib.load(MODELS_DIR / "scaler.pkl")


def parse_dependents(value: str) -> int:
    return 3 if value == "3+" else int(value)


def build_features(raw_inputs: dict, feature_names: list[str]) -> pd.DataFrame:
    applicant_income = float(raw_inputs["ApplicantIncome"])
    coapplicant_income = float(raw_inputs["CoapplicantIncome"])
    loan_amount = float(raw_inputs["LoanAmount"])
    loan_term = float(raw_inputs["Loan_Amount_Term"])
    total_income = applicant_income + coapplicant_income
    monthly_installment = loan_amount / loan_term if loan_term else 0.0

    feature_values = {
        "Gender": 1 if raw_inputs["Gender"] == "Male" else 0,
        "Married": 1 if raw_inputs["Married"] == "Yes" else 0,
        "Dependents": parse_dependents(raw_inputs["Dependents"]),
        "Education": 1 if raw_inputs["Education"] == "Graduate" else 0,
        "Self_Employed": 1 if raw_inputs["Self_Employed"] == "Yes" else 0,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": int(raw_inputs["Credit_History"]),
        "TotalIncome": total_income,
        "LoanAmountToIncome": loan_amount / total_income
        if total_income
        else 0.0,
        "EMI": monthly_installment,
        "EMIToIncome": monthly_installment / total_income
        if total_income
        else 0.0,
        "Log_LoanAmount": np.log1p(loan_amount),
        "Log_TotalIncome": np.log1p(total_income),
        "Has_Coapplicant": 1 if coapplicant_income > 0 else 0,
        "Property_Area_Semiurban": 1
        if raw_inputs["Property_Area"] == "Semiurban"
        else 0,
        "Property_Area_Urban": 1
        if raw_inputs["Property_Area"] == "Urban"
        else 0,
    }

    return pd.DataFrame(
        [[feature_values[name] for name in feature_names]],
        columns=feature_names,
    )


def predict_loan_approval(
    model_key: str, features: pd.DataFrame
) -> tuple[int, np.ndarray]:
    model = load_model(model_key)

    if model_key == "logistic_regression":
        scaled_features = load_scaler().transform(features)
        prediction = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0]
    else:
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

    return prediction, probabilities


def predict_with_explanation(
    model_key: str, feature_frame: pd.DataFrame
) -> dict[str, object]:
    model = load_model(model_key)
    prediction, proba = predict_loan_approval(model_key, feature_frame)

    if model_key == "logistic_regression":
        scaled_features = load_scaler().transform(feature_frame)
        impact_values = scaled_features[0] * model.coef_[0]
        impact_df = (
            pd.DataFrame(
                {
                    "Feature": feature_frame.columns,
                    "Impact": impact_values,
                }
            )
            .sort_values(
                "Impact", key=lambda col: np.abs(col), ascending=False
            )
            .head(5)
        )
        explanation_note = (
            "Top 5 des contributions locales (valeur normalisée x "
            "coefficient du modèle)."
        )
    else:
        impact_df = (
            pd.DataFrame(
                {
                    "Feature": feature_frame.columns,
                    "Impact": model.feature_importances_,
                }
            )
            .sort_values("Impact", ascending=False)
            .head(5)
        )
        explanation_note = (
            "Top 5 des importances globales du modèle Random Forest."
        )

    return {
        "model_prediction": prediction,
        "proba_approved": float(proba[1]),
        "proba_rejected": float(proba[0]),
        "impact_df": impact_df,
        "explanation_note": explanation_note,
    }


def get_eval_proba_and_importance(
    model_key: str, X_eval: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    model = load_model(model_key)

    if model_key == "logistic_regression":
        X_for_model = load_scaler().transform(X_eval)
        y_proba = model.predict_proba(X_for_model)[:, 1]
        importances = np.abs(model.coef_[0])
    else:
        y_proba = model.predict_proba(X_eval)[:, 1]
        importances = model.feature_importances_

    return y_proba, importances


def compute_threshold_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    step: float = 0.01,
) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.0, 1.0 + step, step), 2)
    rows = []

    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "accuracy": accuracy_score(y_true, y_pred_thr),
                "precision": precision_score(
                    y_true, y_pred_thr, zero_division=0
                ),
                "recall": recall_score(y_true, y_pred_thr, zero_division=0),
                "f1": f1_score(y_true, y_pred_thr, zero_division=0),
                "approval_rate": float(np.mean(y_pred_thr)),
            }
        )

    return pd.DataFrame(rows)


def build_calibration_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    calib_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_proba": y_proba,
        }
    )
    calib_df["bin"] = pd.cut(
        calib_df["y_proba"],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
    )

    table = (
        calib_df.groupby("bin", observed=False)
        .agg(
            predicted_mean=("y_proba", "mean"),
            observed_rate=("y_true", "mean"),
            sample_count=("y_true", "count"),
        )
        .reset_index(drop=True)
        .dropna()
    )

    return table


def build_fairness_input_frame(X_eval: pd.DataFrame) -> pd.DataFrame:
    fairness_input = pd.DataFrame(index=X_eval.index)

    if "Gender" in X_eval.columns:
        fairness_input["Gender"] = np.where(
            X_eval["Gender"] == 1,
            "Male",
            "Female",
        )

    if "Education" in X_eval.columns:
        fairness_input["Education"] = np.where(
            X_eval["Education"] == 1,
            "Graduate",
            "Not Graduate",
        )

    if "Married" in X_eval.columns:
        fairness_input["Married"] = np.where(
            X_eval["Married"] == 1,
            "Yes",
            "No",
        )

    if {
        "Property_Area_Semiurban",
        "Property_Area_Urban",
    }.issubset(X_eval.columns):
        fairness_input["Property_Area"] = np.select(
            [
                X_eval["Property_Area_Semiurban"] == 1,
                X_eval["Property_Area_Urban"] == 1,
            ],
            ["Semiurban", "Urban"],
            default="Rural",
        )

    return fairness_input


def compute_group_fairness_table(
    X_eval: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    fairness_input = build_fairness_input_frame(X_eval)
    if fairness_input.shape[1] == 0:
        return pd.DataFrame()

    eval_df = fairness_input.copy()
    eval_df["y_true"] = y_true
    eval_df["y_pred"] = y_pred
    eval_df["error"] = (y_pred != y_true).astype(int)

    rows = []
    for dimension in fairness_input.columns:
        for group_name, group_df in eval_df.groupby(dimension, dropna=False):
            if len(group_df) == 0:
                continue

            negatives = (group_df["y_true"] == 0).sum()
            positives = (group_df["y_true"] == 1).sum()

            false_pos = (
                (group_df["y_pred"] == 1) & (group_df["y_true"] == 0)
            ).sum()
            false_neg = (
                (group_df["y_pred"] == 0) & (group_df["y_true"] == 1)
            ).sum()

            rows.append(
                {
                    "dimension": str(dimension),
                    "group": str(group_name),
                    "count": int(len(group_df)),
                    "predicted_approval_rate": float(
                        group_df["y_pred"].mean()
                    ),
                    "actual_approval_rate": float(group_df["y_true"].mean()),
                    "error_rate": float(group_df["error"].mean()),
                    "fpr": float(false_pos / negatives)
                    if negatives > 0
                    else np.nan,
                    "fnr": float(false_neg / positives)
                    if positives > 0
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


def loan_status_to_binary(series: pd.Series) -> pd.Series:
    status = series.astype(str).str.strip().str.upper()
    approved_values = {"Y", "1", "APPROVED", "YES", "TRUE"}
    rejected_values = {"N", "0", "REJECTED", "NO", "FALSE"}

    binary = pd.Series(np.nan, index=series.index, dtype="float64")
    binary[status.isin(approved_values)] = 1.0
    binary[status.isin(rejected_values)] = 0.0
    return binary


@st.cache_data
def build_holdout_set(
    clean_df: pd.DataFrame,
    feature_names: list[str],
    holdout_ratio: float,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, int, int]:
    eval_df = clean_df[feature_names + ["Loan_Status"]].dropna().copy()
    y_binary = (
        loan_status_to_binary(eval_df["Loan_Status"])
        if eval_df["Loan_Status"].dtype == object
        else pd.to_numeric(eval_df["Loan_Status"], errors="coerce")
    )
    valid_mask = y_binary.notna()
    X_all = eval_df.loc[valid_mask, feature_names]
    y_all = y_binary.loc[valid_mask].astype(int).to_numpy()

    if len(X_all) == 0:
        return X_all, y_all, 0, 0

    if len(y_all) < 2:
        return X_all, y_all, len(y_all), len(y_all)

    stratify = y_all if len(np.unique(y_all)) > 1 else None
    _, X_test, _, y_test = train_test_split(
        X_all,
        y_all,
        test_size=holdout_ratio,
        random_state=random_state,
        stratify=stratify,
    )

    return X_test, y_test, len(y_all), len(y_test)


if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


metadata = load_metadata()
raw_data = load_data()
clean_data = load_clean_data()

st.sidebar.title("💰 Loan Approval Predictor")
st.sidebar.caption("Application Streamlit de scoring de prêt")

selected_model_label = st.sidebar.selectbox(
    "Choisir un modèle",
    list(MODEL_OPTIONS.keys()),
)
selected_model_key = MODEL_OPTIONS[selected_model_label]

if "decision_threshold" not in st.session_state:
    st.session_state.decision_threshold = 0.5
if "pending_decision_threshold" in st.session_state:
    applied_threshold = float(
        st.session_state.pop("pending_decision_threshold")
    )
    st.session_state.decision_threshold = applied_threshold
    st.session_state.threshold_toast_value = applied_threshold

threshold = st.sidebar.slider(
    "Seuil de prédiction",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state.decision_threshold),
    step=0.01,
    key="decision_threshold",
)

if "threshold_toast_value" in st.session_state:
    st.toast(
        "Seuil mis à jour à "
        f"{st.session_state.pop('threshold_toast_value'):.2f}",
        icon="✅",
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Filtres exploration")

income_min = (
    float(raw_data["ApplicantIncome"].min())
    if "ApplicantIncome" in raw_data.columns
    else 0.0
)
income_max = (
    float(raw_data["ApplicantIncome"].max())
    if "ApplicantIncome" in raw_data.columns
    else 0.0
)
income_filter = st.sidebar.slider(
    "Revenu du demandeur (min / max)",
    min_value=income_min,
    max_value=income_max,
    value=(income_min, income_max),
)

education_options = ["Tous"]
if "Education" in raw_data.columns:
    education_values = sorted(
        raw_data["Education"].dropna().astype(str).unique().tolist()
    )
    education_options.extend(education_values)

education_filter = st.sidebar.selectbox(
    "Niveau d'éducation", education_options
)

st.title("🏛️ Prédiction d'Approbation de Prêt")
st.markdown(
    "**Analysez les données, simulez une demande et comparez les "
    "performances des modèles.**"
)
st.info(
    "💡 Utilisez la barre latérale pour choisir le modèle et ajuster "
    "le seuil de décision."
)
with st.expander("🚀 Mode d'emploi rapide (3 étapes)"):
    st.write("1. Explorez les données pour comprendre le profil global.")
    st.write("2. Simulez une demande dans l'onglet Prédiction.")
    st.write(
        "3. Analysez performance, calibration et équité dans l'onglet "
        "Performance."
    )

tab_exploration, tab_prediction, tab_performance = st.tabs(
    [
        "📊 Exploration des données",
        "🤖 Prédiction",
        "📈 Performance du modèle",
    ]
)

with tab_exploration:
    st.header("📊 Dashboard d'exploration")

    uploaded_file = st.file_uploader(
        "Charger un autre fichier CSV", type=["csv"]
    )
    display_data = raw_data
    if uploaded_file is not None:
        display_data = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        st.caption("Aperçu du fichier importé par l'utilisateur.")
    else:
        st.caption(
            "Dataset par défaut chargé avec cache depuis data/loan_data.csv."
        )

    filtered_data = display_data.copy()
    if "ApplicantIncome" in filtered_data.columns:
        applicant_income_numeric = pd.to_numeric(
            filtered_data["ApplicantIncome"], errors="coerce"
        )
        filtered_data = filtered_data.loc[
            applicant_income_numeric.between(
                income_filter[0], income_filter[1], inclusive="both"
            )
        ]

    if education_filter != "Tous" and "Education" in filtered_data.columns:
        filtered_data = filtered_data.loc[
            filtered_data["Education"].astype(str) == education_filter
        ]

    status_binary = (
        loan_status_to_binary(filtered_data["Loan_Status"])
        if "Loan_Status" in filtered_data.columns
        else pd.Series(np.nan, index=filtered_data.index, dtype="float64")
    )

    if "TotalIncome" in filtered_data.columns:
        income_series = pd.to_numeric(
            filtered_data["TotalIncome"], errors="coerce"
        )
    elif {"ApplicantIncome", "CoapplicantIncome"}.issubset(
        filtered_data.columns
    ):
        income_series = pd.to_numeric(
            filtered_data["ApplicantIncome"], errors="coerce"
        ) + pd.to_numeric(filtered_data["CoapplicantIncome"], errors="coerce")
    elif "ApplicantIncome" in filtered_data.columns:
        income_series = pd.to_numeric(
            filtered_data["ApplicantIncome"], errors="coerce"
        )
    else:
        income_series = pd.Series(
            np.nan, index=filtered_data.index, dtype="float64"
        )

    loan_amount_series = (
        pd.to_numeric(filtered_data["LoanAmount"], errors="coerce")
        if "LoanAmount" in filtered_data.columns
        else pd.Series(np.nan, index=filtered_data.index, dtype="float64")
    )

    st.subheader("Indicateurs clés")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Nombre total de demandes", f"{len(filtered_data)}")

    approval_rate = (
        float(np.nanmean(status_binary) * 100)
        if status_binary.notna().any()
        else 0.0
    )
    metric_col2.metric("Taux d'approbation global", f"{approval_rate:.1f}%")
    metric_col3.metric(
        "Montant moyen des prêts",
        f"{loan_amount_series.mean(skipna=True):.1f}",
    )
    metric_col4.metric(
        "Revenu moyen", f"{income_series.mean(skipna=True):.1f}"
    )

    st.markdown("---")
    st.subheader("Distributions")
    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        if "ApplicantIncome" in filtered_data.columns:
            fig_income = px.histogram(
                filtered_data,
                x="ApplicantIncome",
                title="Distribution des revenus",
                labels={"ApplicantIncome": "Revenu"},
                color_discrete_sequence=["#1f77b4"],
            )
            st.plotly_chart(fig_income, use_container_width=True)
        else:
            st.info(
                "La colonne ApplicantIncome est absente du dataset filtré."
            )

    with dist_col2:
        if "LoanAmount" in filtered_data.columns:
            fig_loan_box = px.box(
                filtered_data,
                y="LoanAmount",
                title="Distribution du montant du prêt",
                labels={"LoanAmount": "Montant du prêt"},
            )
            st.plotly_chart(fig_loan_box, use_container_width=True)
        else:
            st.info("La colonne LoanAmount est absente du dataset filtré.")

    st.subheader("Analyses")
    analysis_col1, analysis_col2 = st.columns(2)

    with analysis_col1:
        if {"Education", "Loan_Status"}.issubset(filtered_data.columns):
            approval_edu_df = filtered_data[
                ["Education", "Loan_Status"]
            ].copy()
            approval_edu_df["Loan_Status_Binary"] = loan_status_to_binary(
                approval_edu_df["Loan_Status"]
            )
            approval_by_education = (
                approval_edu_df.groupby("Education", dropna=False)[
                    "Loan_Status_Binary"
                ]
                .mean()
                .dropna()
                * 100
            )

            fig_approval_edu = px.bar(
                x=approval_by_education.index.astype(str),
                y=approval_by_education.values,
                title="Taux d'approbation par niveau d'éducation",
                labels={"x": "Éducation", "y": "Taux d'approbation (%)"},
                color=approval_by_education.values,
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_approval_edu, use_container_width=True)
        else:
            st.info(
                "Les colonnes Education et Loan_Status sont nécessaires "
                "pour ce graphique."
            )

    with analysis_col2:
        if "Loan_Status" in filtered_data.columns:
            status_text = (
                filtered_data["Loan_Status"]
                .astype(str)
                .str.strip()
                .str.upper()
            )
            approved_count = int(
                status_text.isin(["Y", "1", "APPROVED", "YES", "TRUE"]).sum()
            )
            rejected_count = int(
                status_text.isin(["N", "0", "REJECTED", "NO", "FALSE"]).sum()
            )

            fig_pie = px.pie(
                values=[approved_count, rejected_count],
                names=["Approved", "Rejected"],
                title="Répartition Approved / Rejected",
                color_discrete_sequence=["#00CC96", "#EF553B"],
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("La colonne Loan_Status est absente du dataset filtré.")

    st.subheader("Corrélations")
    numeric_df = filtered_data.select_dtypes(include=["number"]).copy()
    if "Loan_Status" in filtered_data.columns:
        numeric_df["Loan_Status_Binary"] = status_binary

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmid=0,
            )
        )
        fig_heatmap.update_layout(title="Matrice de corrélation")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info(
            "Pas assez de colonnes numériques pour calculer une corrélation."
        )

    st.markdown("---")
    st.subheader("Données filtrées")
    st.dataframe(filtered_data, width="stretch")

    with st.expander("Voir un aperçu structuré"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Premières lignes")
            st.table(display_data.head())
        with col2:
            st.subheader("Informations dataset")
            st.json(
                {
                    "source": uploaded_file.name
                    if uploaded_file is not None
                    else DATA_PATH.name,
                    "dimensions": {
                        "rows": int(filtered_data.shape[0]),
                        "columns": int(filtered_data.shape[1]),
                    },
                    "columns": filtered_data.columns.tolist(),
                }
            )

    with st.expander("Voir le dataset de modélisation"):
        st.caption("Version nettoyée utilisée pour entraîner les modèles.")
        st.dataframe(clean_data, width="stretch")

with tab_prediction:
    st.header("🤖 Simulation d'une demande de prêt")
    st.info(
        "🧭 Remplissez le formulaire puis cliquez sur Prédire pour "
        "obtenir la décision et son explication."
    )
    st.caption(f"Modèle actif : {selected_model_label}")

    sample_cases = {
        "Personnalisé": {
            "ApplicantIncome": 5000.0,
            "CoapplicantIncome": 1500.0,
            "LoanAmount": 140.0,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Education": "Graduate",
            "Married": "Yes",
            "Gender": "Male",
            "Dependents": "1",
            "Self_Employed": "No",
            "Property_Area": "Semiurban",
        },
        "Profil prudent": {
            "ApplicantIncome": 6500.0,
            "CoapplicantIncome": 2000.0,
            "LoanAmount": 120.0,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Education": "Graduate",
            "Married": "Yes",
            "Gender": "Female",
            "Dependents": "0",
            "Self_Employed": "No",
            "Property_Area": "Urban",
        },
        "Profil risqué": {
            "ApplicantIncome": 1800.0,
            "CoapplicantIncome": 0.0,
            "LoanAmount": 220.0,
            "Loan_Amount_Term": 120,
            "Credit_History": 0,
            "Education": "Not Graduate",
            "Married": "No",
            "Gender": "Male",
            "Dependents": "3+",
            "Self_Employed": "Yes",
            "Property_Area": "Rural",
        },
    }
    selected_case = st.selectbox(
        "Cas d'exemple",
        options=list(sample_cases.keys()),
        help="Pré-remplit le formulaire avec un profil type.",
    )
    case_defaults = sample_cases[selected_case]

    with st.form("prediction_form"):
        form_col1, form_col2 = st.columns(2)

        with form_col1:
            st.subheader("Données financières")
            applicant_income = st.number_input(
                "Revenu mensuel du demandeur",
                min_value=0.0,
                value=case_defaults["ApplicantIncome"],
                step=100.0,
                help="Revenu principal mensuel en devise locale.",
            )
            coapplicant_income = st.number_input(
                "Revenu mensuel du co-demandeur",
                min_value=0.0,
                value=case_defaults["CoapplicantIncome"],
                step=100.0,
                help="Laisser 0 si aucun co-demandeur.",
            )
            loan_amount = st.number_input(
                "Montant demandé",
                min_value=1.0,
                value=case_defaults["LoanAmount"],
                step=1.0,
                help="Montant total du prêt demandé.",
            )
            loan_term = st.slider(
                "Durée du prêt (mois)",
                min_value=12,
                max_value=480,
                value=int(case_defaults["Loan_Amount_Term"]),
                step=12,
                help="Durée de remboursement en mois.",
            )

        with form_col2:
            st.subheader("Profil emprunteur")
            credit_history = st.radio(
                "Historique de crédit",
                options=[1, 0],
                format_func=lambda x: "Positif" if x == 1 else "Négatif",
                index=0 if case_defaults["Credit_History"] == 1 else 1,
                horizontal=True,
                help="Positif = historique de remboursement satisfaisant.",
            )
            education = st.selectbox(
                "Niveau d'éducation",
                ["Graduate", "Not Graduate"],
                index=["Graduate", "Not Graduate"].index(
                    case_defaults["Education"]
                ),
            )
            married = st.selectbox(
                "Statut marital",
                ["No", "Yes"],
                index=["No", "Yes"].index(case_defaults["Married"]),
            )
            gender = st.selectbox(
                "Genre",
                ["Male", "Female"],
                index=["Male", "Female"].index(case_defaults["Gender"]),
            )
            dependents = st.selectbox(
                "Nombre de personnes à charge",
                ["0", "1", "2", "3+"],
                index=["0", "1", "2", "3+"].index(
                    case_defaults["Dependents"]
                ),
            )
            self_employed = st.selectbox(
                "Travailleur indépendant",
                ["No", "Yes"],
                index=["No", "Yes"].index(
                    case_defaults["Self_Employed"]
                ),
            )
            property_area = st.selectbox(
                "Zone du bien",
                ["Rural", "Semiurban", "Urban"],
                index=["Rural", "Semiurban", "Urban"].index(
                    case_defaults["Property_Area"]
                ),
            )

        submitted = st.form_submit_button("Prédire", use_container_width=True)

    if submitted:
        warnings = []
        total_income = applicant_income + coapplicant_income
        monthly_payment = loan_amount / loan_term if loan_term else 0

        if applicant_income < 1000:
            warnings.append("⚠️ Le revenu du demandeur semble très bas.")
        if loan_amount > max(total_income, 1) * 100:
            warnings.append(
                "⚠️ Le montant du prêt est très élevé par rapport "
                "au revenu mensuel total."
            )
        if monthly_payment > max(total_income, 1):
            warnings.append(
                "⚠️ La mensualité estimée dépasse le revenu mensuel total."
            )
        if loan_term < 60:
            warnings.append(
                "⚠️ Une durée très courte augmente fortement la mensualité."
            )

        if warnings:
            st.warning("Points d'attention détectés :")
            for warning in warnings:
                st.write(warning)

        raw_inputs = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": int(credit_history),
            "Property_Area": property_area,
        }

        try:
            feature_frame = build_features(
                raw_inputs, metadata["feature_names"]
            )
            prediction_result = predict_with_explanation(
                selected_model_key,
                feature_frame,
            )

            st.session_state.last_prediction = {
                "features": feature_frame,
                **prediction_result,
            }

        except Exception as error:
            st.error(f"❌ Erreur lors de la prédiction : {error}")
            st.info(
                "Vérifiez que toutes les données sont correctement "
                "renseignées puis réessayez."
            )

    if st.session_state.last_prediction is not None:
        result = st.session_state.last_prediction
        threshold_decision = 1 if result["proba_approved"] >= threshold else 0
        decision_label = (
            "✅ PRÊT APPROUVÉ" if threshold_decision == 1 else "❌ PRÊT REJETÉ"
        )

        st.markdown("---")
        st.subheader("📊 Résultat de la prédiction")

        if threshold_decision == 1:
            st.success(decision_label)
        else:
            st.error(decision_label)

        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        result_col1.metric(
            "Probabilité d'approbation",
            f"{result['proba_approved'] * 100:.1f}%",
        )
        result_col2.metric(
            "Probabilité de rejet", f"{result['proba_rejected'] * 100:.1f}%"
        )
        result_col3.metric(
            "Décision modèle",
            "Approved" if result["model_prediction"] == 1 else "Rejected",
        )
        result_col4.metric(
            "Décision avec seuil",
            "Approved" if threshold_decision == 1 else "Rejected",
        )

        st.progress(result["proba_approved"])
        st.caption(f"Seuil appliqué dans la sidebar : {threshold:.0%}")

        confidence_margin = abs(result["proba_approved"] - threshold)
        if confidence_margin >= 0.25:
            confidence_label = "Confiance élevée"
        elif confidence_margin >= 0.10:
            confidence_label = "Confiance moyenne"
        else:
            confidence_label = "Confiance faible"

        st.subheader("🧾 Synthèse décisionnelle")
        threshold_text = (
            "Approved" if threshold_decision == 1 else "Rejected"
        )
        st.info(
            f"Décision seuil: {threshold_text} | "
            f"Probabilité d'approbation: {result['proba_approved']:.1%} | "
            f"{confidence_label}"
        )
        st.caption(
            "La confiance dépend de l'écart entre la probabilité "
            "prédite et le seuil choisi."
        )

        st.subheader("🔍 Explication")
        st.write(result["explanation_note"])
        fig_impact = px.bar(
            result["impact_df"].sort_values("Impact", ascending=True),
            x="Impact",
            y="Feature",
            orientation="h",
            color="Impact",
            color_continuous_scale="RdYlGn",
            title="Top 5 des features influentes",
        )
        st.plotly_chart(fig_impact, use_container_width=True)

        with st.expander("Voir les variables envoyées au modèle"):
            st.dataframe(result["features"], width="stretch")

with tab_performance:
    st.header("📈 Performance du modèle")
    st.info(
        "🔎 Cette section affiche la qualité du modèle sélectionné "
        "sur un jeu de test holdout."
    )
    st.markdown(
        "**Lecture rapide:** Commencez par la synthèse exécutive, "
        "puis regardez l'optimisation du seuil, la calibration et "
        "l'équité pour valider la robustesse de la décision."
    )

    feature_names = metadata["feature_names"]
    has_required_columns = set(feature_names + ["Loan_Status"]).issubset(
        clean_data.columns
    )

    if not has_required_columns:
        st.error(
            "Les colonnes nécessaires à l'évaluation sont absentes "
            "dans data/loan_data_clean.csv."
        )
    else:
        train_size = metadata.get("train_size")
        test_size = metadata.get("test_size")
        if (
            isinstance(train_size, int)
            and isinstance(test_size, int)
            and (train_size + test_size) > 0
        ):
            holdout_ratio = test_size / (train_size + test_size)
        else:
            holdout_ratio = 0.2

        X_eval, y_true, valid_rows, holdout_rows = build_holdout_set(
            clean_data,
            feature_names,
            holdout_ratio,
        )
        st.caption(
            f"Évaluation holdout: {holdout_rows} lignes test sur "
            f"{valid_rows} lignes valides "
            f"(ratio {holdout_ratio:.0%}, random_state=42)."
        )

        if len(X_eval) == 0:
            st.warning(
                "Aucune donnée valide disponible pour évaluer le modèle."
            )
        else:
            y_proba, importances = get_eval_proba_and_importance(
                selected_model_key,
                X_eval,
            )

            y_pred = (y_proba >= threshold).astype(int)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_value = float(auc(fpr, tpr))
            else:
                fpr, tpr, auc_value = np.array([0, 1]), np.array([0, 1]), 0.5
                st.warning(
                    "La courbe ROC est moins informative car une "
                    "seule classe est présente dans les données évaluées."
                )

            threshold_table = compute_threshold_table(y_true, y_proba)
            best_f1_row = threshold_table.loc[
                threshold_table["f1"].idxmax()
            ]

            st.subheader("🧭 Synthèse exécutive")
            summary_col1, summary_col2, summary_col3, summary_col4 = (
                st.columns(4)
            )
            summary_col1.metric("Performance globale", f"AUC {auc_value:.3f}")
            summary_col2.metric("Qualité décision", f"F1 {f1:.1%}")
            summary_col3.metric(
                "Seuil actuel",
                f"{threshold:.2f}",
                delta=f"Reco F1 {float(best_f1_row['threshold']):.2f}",
            )
            summary_col4.metric(
                "Taux d'approbation prédit",
                f"{float(np.mean(y_pred)):.1%}",
            )
            st.caption(
                "Interprétation: un AUC élevé indique une bonne capacité "
                "de séparation, tandis que le F1 équilibre précision "
                "et rappel."
            )

            st.subheader("📊 Métriques de performance")
            st.caption(
                f"Seuil de décision appliqué aux métriques : {threshold:.0%}"
            )
            perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(
                5
            )
            perf_col1.metric("Accuracy", f"{acc:.1%}")
            perf_col2.metric("Precision", f"{prec:.1%}")
            perf_col3.metric("Recall", f"{rec:.1%}")
            perf_col4.metric("F1-score", f"{f1:.1%}")
            perf_col5.metric("AUC", f"{auc_value:.3f}")

            with st.expander("ℹ️ Que signifient ces métriques ?"):
                st.write("- Accuracy : part des prédictions correctes.")
                st.write(
                    "- Precision : fiabilité des prêts prédits comme "
                    "approuvés."
                )
                st.write(
                    "- Recall : capacité à détecter les prêts réellement "
                    "approuvables."
                )
                st.write("- F1-score : compromis entre précision et rappel.")
                st.write(
                    "- AUC : qualité de séparation globale entre classes."
                )

            st.subheader("🎯 Optimisation du seuil")
            tuning_col1, tuning_col2 = st.columns([2, 1])

            with tuning_col1:
                fig_tuning = go.Figure()
                fig_tuning.add_trace(
                    go.Scatter(
                        x=threshold_table["threshold"],
                        y=threshold_table["f1"],
                        name="F1",
                        line=dict(color="#1f77b4", width=2),
                    )
                )
                fig_tuning.add_trace(
                    go.Scatter(
                        x=threshold_table["threshold"],
                        y=threshold_table["precision"],
                        name="Precision",
                        line=dict(color="#2ca02c", width=2),
                    )
                )
                fig_tuning.add_trace(
                    go.Scatter(
                        x=threshold_table["threshold"],
                        y=threshold_table["recall"],
                        name="Recall",
                        line=dict(color="#d62728", width=2),
                    )
                )
                fig_tuning.add_vline(
                    x=threshold,
                    line_dash="dash",
                    line_color="#636EFA",
                    annotation_text="Seuil actuel",
                    annotation_position="top right",
                )
                fig_tuning.update_layout(
                    title="Impact du seuil sur Precision / Recall / F1",
                    xaxis_title="Seuil",
                    yaxis_title="Score",
                    yaxis_range=[0, 1],
                )
                st.plotly_chart(fig_tuning, use_container_width=True)
                st.caption(
                    "Interprétation: déplacez le seuil pour arbitrer entre "
                    "rappel (détection) et précision (fiabilité)."
                )

            with tuning_col2:
                strategy = st.selectbox(
                    "Stratégie de seuil",
                    [
                        "Maximiser F1",
                        "Priorité Recall (>= 90%)",
                        "Priorité Precision (>= 90%)",
                    ],
                )

                if strategy == "Maximiser F1":
                    best_row = threshold_table.loc[
                        threshold_table["f1"].idxmax()
                    ]
                elif strategy == "Priorité Recall (>= 90%)":
                    candidates = threshold_table[
                        threshold_table["recall"] >= 0.90
                    ]
                    if len(candidates) > 0:
                        best_row = candidates.loc[
                            candidates["precision"].idxmax()
                        ]
                    else:
                        best_row = threshold_table.loc[
                            threshold_table["recall"].idxmax()
                        ]
                else:
                    candidates = threshold_table[
                        threshold_table["precision"] >= 0.90
                    ]
                    if len(candidates) > 0:
                        best_row = candidates.loc[
                            candidates["recall"].idxmax()
                        ]
                    else:
                        best_row = threshold_table.loc[
                            threshold_table["precision"].idxmax()
                        ]

                st.metric(
                    "Seuil recommandé",
                    f"{float(best_row['threshold']):.2f}",
                )
                st.metric(
                    "F1 attendu",
                    f"{float(best_row['f1']):.1%}",
                )
                st.metric(
                    "Precision attendue",
                    f"{float(best_row['precision']):.1%}",
                )
                st.metric(
                    "Recall attendu",
                    f"{float(best_row['recall']):.1%}",
                )
                recommended_threshold = round(
                    float(best_row["threshold"]),
                    2,
                )
                if st.button(
                    "Appliquer le seuil recommande",
                    use_container_width=True,
                ):
                    st.session_state.pending_decision_threshold = (
                        recommended_threshold
                    )
                    st.rerun()
                st.caption(
                    "Le seuil conseillé est calculé sur le holdout courant."
                )

            st.subheader("🧪 Calibration des probabilités")
            brier = brier_score_loss(y_true, y_proba)
            st.caption(
                f"Brier score: {brier:.4f} "
                "(plus proche de 0 = meilleure calibration)."
            )
            calibration_table = build_calibration_table(y_true, y_proba)

            if len(calibration_table) > 0:
                fig_calib = go.Figure()
                fig_calib.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        name="Calibration parfaite",
                        line=dict(color="#636EFA", dash="dash"),
                    )
                )
                fig_calib.add_trace(
                    go.Scatter(
                        x=calibration_table["predicted_mean"],
                        y=calibration_table["observed_rate"],
                        name="Modèle",
                        mode="lines+markers",
                        marker=dict(size=8),
                        line=dict(color="#FF7F0E", width=2),
                        text=calibration_table["sample_count"].astype(str),
                        hovertemplate=(
                            "Proba moyenne: %{x:.2f}<br>"
                            "Taux observé: %{y:.2f}<br>"
                            "N: %{text}<extra></extra>"
                        ),
                    )
                )
                fig_calib.update_layout(
                    title="Courbe de calibration (reliability diagram)",
                    xaxis_title="Probabilité prédite moyenne",
                    yaxis_title="Fréquence observée",
                    xaxis_range=[0, 1],
                    yaxis_range=[0, 1],
                )
                st.plotly_chart(fig_calib, use_container_width=True)
                st.caption(
                    "Interprétation: plus la courbe suit la diagonale, "
                    "plus les probabilités sont fiables."
                )
            else:
                st.info(
                    "Pas assez de données pour construire la courbe "
                    "de calibration."
                )

            st.subheader("⚖️ Analyse d'équité par sous-groupes")
            fairness_table = compute_group_fairness_table(
                X_eval,
                y_true,
                y_pred,
            )

            if len(fairness_table) > 0:
                fairness_dim = st.selectbox(
                    "Dimension analysée",
                    sorted(fairness_table["dimension"].unique().tolist()),
                )
                fairness_dim_df = fairness_table[
                    fairness_table["dimension"] == fairness_dim
                ].copy()

                fig_fairness = go.Figure()
                fig_fairness.add_trace(
                    go.Bar(
                        x=fairness_dim_df["group"],
                        y=fairness_dim_df["predicted_approval_rate"],
                        name="Taux approuvé (prédit)",
                        marker_color="#1f77b4",
                    )
                )
                fig_fairness.add_trace(
                    go.Bar(
                        x=fairness_dim_df["group"],
                        y=fairness_dim_df["actual_approval_rate"],
                        name="Taux approuvé (réel)",
                        marker_color="#2ca02c",
                    )
                )
                fig_fairness.update_layout(
                    title=f"Comparaison par groupe: {fairness_dim}",
                    yaxis_title="Taux",
                    yaxis_range=[0, 1],
                    barmode="group",
                )
                st.plotly_chart(fig_fairness, use_container_width=True)
                st.caption(
                    "Interprétation: des écarts importants entre groupes "
                    "peuvent signaler un risque d'iniquité."
                )

                st.dataframe(
                    fairness_dim_df.assign(
                        predicted_approval_rate=fairness_dim_df[
                            "predicted_approval_rate"
                        ].map(lambda x: f"{x:.1%}"),
                        actual_approval_rate=fairness_dim_df[
                            "actual_approval_rate"
                        ].map(lambda x: f"{x:.1%}"),
                        error_rate=fairness_dim_df["error_rate"].map(
                            lambda x: f"{x:.1%}"
                        ),
                        fpr=fairness_dim_df["fpr"].map(
                            lambda x: "-" if pd.isna(x) else f"{x:.1%}"
                        ),
                        fnr=fairness_dim_df["fnr"].map(
                            lambda x: "-" if pd.isna(x) else f"{x:.1%}"
                        ),
                    ),
                    width="stretch",
                )

                approval_gap = float(
                    fairness_dim_df["predicted_approval_rate"].max()
                    - fairness_dim_df["predicted_approval_rate"].min()
                )
                error_gap = float(
                    fairness_dim_df["error_rate"].max()
                    - fairness_dim_df["error_rate"].min()
                )

                gap_col1, gap_col2 = st.columns(2)
                gap_col1.metric(
                    "Ecart taux d'approbation",
                    f"{approval_gap:.1%}",
                )
                gap_col2.metric("Ecart taux d'erreur", f"{error_gap:.1%}")

                if approval_gap >= 0.15 or error_gap >= 0.10:
                    st.warning(
                        "Alerte équité: des écarts significatifs existent "
                        "sur cette dimension."
                    )
                else:
                    st.success(
                        "Aucun écart majeur détecté avec les seuils "
                        "d'alerte actuels."
                    )

                st.caption(
                    "Seuils d'alerte: 15 points sur le taux d'approbation "
                    "prédit ou 10 points sur le taux d'erreur."
                )
            else:
                st.info(
                    "Colonnes insuffisantes pour calculer l'analyse "
                    "d'équité par groupe."
                )

            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                fig_cm = go.Figure(
                    data=go.Heatmap(
                        z=cm,
                        x=["Rejected", "Approved"],
                        y=["Rejected", "Approved"],
                        colorscale="Blues",
                        showscale=True,
                    )
                )
                fig_cm.update_layout(
                    title="Matrice de confusion",
                    xaxis_title="Prédiction",
                    yaxis_title="Vérité terrain",
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with viz_col2:
                fig_roc = go.Figure()
                fig_roc.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        name="ROC",
                        line=dict(color="#FF7F0E", width=2),
                    )
                )
                fig_roc.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        name="Random",
                        line=dict(color="#636EFA", dash="dash"),
                    )
                )
                fig_roc.update_layout(
                    title="Courbe ROC",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            st.subheader("🏆 Feature importance globale (Top 10)")
            importance_df = (
                pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Importance": importances,
                    }
                )
                .sort_values("Importance", ascending=False)
                .head(10)
            )
            fig_importance = px.bar(
                importance_df.sort_values("Importance", ascending=True),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title="Top 10 des variables les plus importantes",
            )
            st.plotly_chart(fig_importance, use_container_width=True)

            st.subheader("📉 Distribution des probabilités")
            proba_df = pd.DataFrame(
                {
                    "Probabilité d'approbation": y_proba,
                    "Classe prédite": np.where(
                        y_pred == 1, "Approved", "Rejected"
                    ),
                }
            )
            fig_proba = px.histogram(
                proba_df,
                x="Probabilité d'approbation",
                color="Classe prédite",
                nbins=25,
                barmode="overlay",
                title="Distribution des scores du modèle",
                color_discrete_map={
                    "Approved": "#00CC96",
                    "Rejected": "#EF553B",
                },
            )
            st.plotly_chart(fig_proba, use_container_width=True)

            with st.expander("ℹ️ Lecture des visualisations"):
                st.write(
                    "- Matrice de confusion : erreurs de type faux "
                    "positifs/faux négatifs."
                )
                st.write(
                    "- ROC : plus la courbe est proche du coin "
                    "haut-gauche, meilleur est le modèle."
                )
                st.write(
                    "- Importance : variables les plus structurantes "
                    "pour la décision du modèle."
                )

            st.caption(
                f"Jeu d'entraînement : {metadata['train_size']} lignes | "
                f"Jeu de test : {metadata['test_size']} lignes | "
                f"Modèles entraînés le {metadata['date']}"
            )

st.markdown("---")
st.subheader("ℹ️ À propos")
st.write(
    "Cette application démontre un pipeline complet de scoring de crédit : "
    "exploration des données, prédiction en temps réel et audit de "
    "performance."
)
with st.expander("🧪 Méthodologie, RGPD et éthique"):
    st.write(
        "- Minimisation des données : seules les variables utiles "
        "à la prédiction sont traitées."
    )
    st.write(
        "- Transparence : les performances et facteurs influents "
        "sont affichés pour auditabilité."
    )
    st.write(
        "- Vigilance biais : le modèle doit être monitoré pour "
        "éviter les discriminations indirectes."
    )
    st.write(
        "- Usage responsable : cet outil assiste une décision "
        "humaine, il ne doit pas décider seul."
    )
st.info("📬 Contact : equipe.data@nexa-edu.fr")

st.markdown("---")
st.caption(
    f"Version {APP_VERSION} | Dernière mise à jour : {metadata['date']} | "
    "Documentation : https://docs.streamlit.io"
)
