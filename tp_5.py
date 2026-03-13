import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Configuration de la page
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction de chargement des données (cachée)
@st.cache_data
def load_data():
    return pd.read_csv(DATA_DIR / "loan_data_clean.csv")

# Fonction de chargement du modèle (cachée)
@st.cache_resource
def load_model(model_name):
    if model_name == "Logistic Regression":
        return joblib.load(MODELS_DIR / "logistic_regression.pkl")
    else:
        return joblib.load(MODELS_DIR / "random_forest.pkl")

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Choisir le modèle",
    ["Logistic Regression", "Random Forest"]
)

# Charger les données et le modèle
df = load_data()
model = load_model(model_choice)

# Titre principal
st.title("🏦 Prédiction d'Approbation de Prêt")
st.markdown("Application de Machine Learning pour évaluer les demandes de prêt")
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🤖 Prédiction", "📈 Performance"])

with tab1:
    st.header("Exploration des données")
    st.dataframe(df, use_container_width=True)
    st.caption(f"Dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")

with tab2:
    st.header("Faire une prédiction")
    st.write("Section à compléter")

with tab3:
    st.header("Performance du modèle")
    st.write("Section à compléter")