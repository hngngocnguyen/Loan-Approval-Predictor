"""
TP 5 - Structure de base de l'application Streamlit

Objectifs :
- Créer la structure de l'application avec page config
- Mettre en place les onglets (tabs)
- Créer le sidebar avec sélection du modèle
- Implémenter les fonctions de chargement avec cache

Instructions :
Complétez les parties marquées TODO en suivant les indices fournis.
"""

import streamlit as st
import pandas as pd
import joblib

# ============================================================================
# TODO 1 : Configuration de la page
# ============================================================================
# Indice : Utilisez st.set_page_config() avec les paramètres suivants :
# - page_title : "Prédiction d'Approbation de Prêt"
# - page_icon : "🏦"
# - layout : "wide" (pour utiliser toute la largeur)
# - initial_sidebar_state : "expanded" (sidebar visible par défaut)

# TODO 1 : Votre code ici


# ============================================================================
# TODO 2 : Titre et description
# ============================================================================
# Indice : Ajoutez un titre avec st.title() et une ligne de séparation avec st.markdown("---")

# TODO 2 : Votre code ici


# ============================================================================
# TODO 3 : Sidebar - Sélection du modèle
# ============================================================================
# Indice : Créez un header dans la sidebar avec st.sidebar.header("⚙️ Configuration")
# Puis un selectbox pour choisir entre "Régression Logistique" et "Random Forest"
# Stockez le résultat dans une variable model_choice

# TODO 3 : Votre code ici


# ============================================================================
# TODO 4 : Sidebar - Info sur le modèle
# ============================================================================
# Indice : Utilisez st.sidebar.info() pour afficher :
# - "📊 Modèle linéaire, interprétable" si Régression Logistique
# - "🌳 Modèle ensemble, plus puissant" si Random Forest
# Utilisez une condition if/else basée sur model_choice

# TODO 4 : Votre code ici


# ============================================================================
# TODO 5 : Sidebar - Section "À propos"
# ============================================================================
# Indice : Ajoutez une section "À propos" dans la sidebar avec :
# - st.sidebar.markdown("---") pour séparer
# - st.sidebar.markdown("### 📖 À propos")
# - Une description de l'application

# TODO 5 : Votre code ici


# ============================================================================
# TODO 6 : Fonction de chargement des données avec cache
# ============================================================================
# Indice : Créez une fonction load_data() avec le décorateur @st.cache_data
# La fonction doit :
# 1. Essayer de charger "data/loan_data_clean.csv" avec pd.read_csv()
# 2. Retourner le DataFrame si succès
# 3. Si FileNotFoundError, afficher une erreur avec st.error() et retourner None
# 4. Utiliser un bloc try/except

@st.cache_data
def load_data():
    """Charge les données depuis le fichier CSV"""
    # TODO 6 : Votre code ici
    pass


# ============================================================================
# TODO 7 : Fonction de chargement du modèle avec cache
# ============================================================================
# Indice : Créez une fonction load_model(model_name) avec @st.cache_resource
# La fonction doit :
# 1. Si model_name == "Régression Logistique" :
#    - Charger "models/model_lr.pkl" et "models/scaler.pkl"
# 2. Si model_name == "Random Forest" :
#    - Charger "models/model_rf.pkl", scaler = None
# 3. Retourner (model, scaler)
# 4. Gérer FileNotFoundError avec st.error() et retourner (None, None)

@st.cache_resource
def load_model(model_name):
    """Charge le modèle sélectionné"""
    # TODO 7 : Votre code ici
    pass


# ============================================================================
# TODO 8 : Charger les données
# ============================================================================
# Indice : Appelez la fonction load_data() et stockez le résultat dans df

# TODO 8 : Votre code ici


# ============================================================================
# TODO 9 : Créer les onglets
# ============================================================================
# Indice : Utilisez st.tabs() pour créer 3 onglets :
# - "📊 Exploration"
# - "🔮 Prédiction"
# - "📈 Performance"
# Stockez dans tab1, tab2, tab3

if df is not None:
    # TODO 9 : Votre code ici
    
    
    # ========================================================================
    # TODO 10 : Contenu de l'onglet Exploration
    # ========================================================================
    # Indice : Dans le contexte "with tab1:", ajoutez :
    # - Un header "📊 Exploration des Données"
    # - Une description
    # - Un placeholder st.info() indiquant que ce sera fait au TP6
    
    # TODO 10 : Votre code ici
    
    
    # ========================================================================
    # TODO 11 : Contenu de l'onglet Prédiction
    # ========================================================================
    # Indice : Dans "with tab2:", similaire à l'onglet Exploration
    
    # TODO 11 : Votre code ici
    
    
    # ========================================================================
    # TODO 12 : Contenu de l'onglet Performance
    # ========================================================================
    # Indice : Dans "with tab3:" :
    # - Header "📈 Performance du Modèle"
    # - Description
    # - Charger le modèle avec load_model(model_choice)
    # - Si model != None, afficher st.success() et quelques métriques
    # - Utiliser st.columns(2) pour afficher Type de modèle et Scaler
    # - Utiliser st.metric() pour chaque métrique
    
    # TODO 12 : Votre code ici
    

else:
    st.error("❌ Impossible de charger les données. Vérifiez que le fichier existe.")


# ============================================================================
# TODO 13 : Footer
# ============================================================================
# Indice : Ajoutez un footer avec st.markdown() :
# - Une ligne de séparation "---"
# - Un message centré (utiliser HTML avec unsafe_allow_html=True)

# TODO 13 : Votre code ici


# ============================================================================
# AIDE MÉMOIRE - Fonctions Streamlit utiles
# ============================================================================
# st.set_page_config() : Configure la page (titre, icône, layout)
# st.title() : Titre principal
# st.header() : Titre de section
# st.markdown() : Texte formaté en Markdown
# st.sidebar : Accès à la sidebar
# st.selectbox() : Menu déroulant
# st.info() : Boîte d'information bleue
# st.error() : Boîte d'erreur rouge
# st.success() : Boîte de succès verte
# st.warning() : Boîte d'avertissement orange
# st.tabs() : Créer des onglets
# st.columns() : Créer des colonnes
# st.metric() : Afficher une métrique
# @st.cache_data : Cache pour données (DataFrames, etc.)
# @st.cache_resource : Cache pour ressources (modèles ML, connexions DB)
# ============================================================================
