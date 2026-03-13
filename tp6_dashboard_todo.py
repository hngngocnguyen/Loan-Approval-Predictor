"""
TP 6 - Dashboard d'exploration interactif

Objectifs :
- Afficher des métriques clés (KPIs)
- Créer des visualisations interactives avec Plotly
- Ajouter des filtres dynamiques
- Implémenter le téléchargement CSV

Instructions :
Complétez les parties marquées TODO en suivant les indices fournis.
Ce fichier remplace le contenu de l'onglet "Exploration" du TP5.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# Supposons que df est déjà chargé depuis le TP5
# ============================================================================


# ============================================================================
# TODO 1 : Créer la section des métriques (KPIs)
# ============================================================================
# Indice : Utilisez st.columns(4) pour créer 4 colonnes
# Pour chaque colonne, calculez et affichez avec st.metric() :
# - Colonne 1 : Nombre total de demandes (len(df))
# - Colonne 2 : Taux d'approbation (% où Loan_Status == 'Y')
# - Colonne 3 : Montant moyen de prêt (mean de LoanAmount)
# - Colonne 4 : Revenu moyen (mean de ApplicantIncome)

st.subheader("📊 Indicateurs Clés")

# TODO 1 : Votre code ici


# ============================================================================
# TODO 2 : Histogramme - Distribution des revenus
# ============================================================================
# Indice : Utilisez px.histogram() avec les paramètres :
# - x="ApplicantIncome"
# - nbins=30
# - title="Distribution des Revenus des Demandeurs"
# - labels={"ApplicantIncome": "Revenu (€)", "count": "Nombre"}
# Ajoutez une ligne verticale pour la moyenne avec fig.add_vline()
# Affichez avec st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 Distribution des Revenus")

# TODO 2 : Votre code ici


# ============================================================================
# TODO 3 : Box plot - Montants de prêt
# ============================================================================
# Indice : Utilisez px.box() avec :
# - y="LoanAmount"
# - title="Distribution des Montants de Prêt"
# - labels={"LoanAmount": "Montant du Prêt (€)"}
# Ajoutez des annotations avec fig.add_annotation() pour afficher :
# - Médiane (df["LoanAmount"].median())
# - Q1 et Q3 (quantile(0.25) et quantile(0.75))

st.subheader("📦 Montants de Prêt Demandés")

# TODO 3 : Votre code ici


# ============================================================================
# TODO 4 : Bar chart - Taux d'approbation par niveau d'éducation
# ============================================================================
# Indice : 
# 1. Créer un DataFrame groupé : df.groupby(["Education", "Loan_Status"]).size().reset_index(name="Count")
# 2. Calculer le pourcentage pour chaque groupe
# 3. Filtrer pour ne garder que Loan_Status == 'Y'
# 4. Utiliser px.bar() avec :
#    - x="Education"
#    - y="Percentage"
#    - title="Taux d'Approbation par Niveau d'Éducation"
#    - color="Education" (optionnel)

st.subheader("🎓 Approbation selon l'Éducation")

# TODO 4 : Votre code ici


# ============================================================================
# TODO 5 : Pie chart - Répartition Approuvé/Rejeté
# ============================================================================
# Indice : Utilisez px.pie() avec :
# - values : comptage de Loan_Status (df["Loan_Status"].value_counts().values)
# - names : ["Approuvé", "Rejeté"]
# - title="Répartition des Décisions"
# - color_discrete_sequence=["#2ecc71", "#e74c3c"]
# - hole=0.4 (pour un donut chart)

st.subheader("🥧 Répartition Approuvé/Rejeté")

# TODO 5 : Votre code ici


# ============================================================================
# TODO 6 : Heatmap - Matrice de corrélation
# ============================================================================
# Indice :
# 1. Sélectionner les colonnes numériques : df.select_dtypes(include=['float64', 'int64'])
# 2. Calculer la corrélation : .corr()
# 3. Utiliser go.Heatmap() avec :
#    - z=corr.values
#    - x=corr.columns
#    - y=corr.columns
#    - colorscale="RdBu"
#    - zmid=0 (centrer sur 0)
# 4. Créer la figure avec go.Figure() et afficher

st.subheader("🔥 Matrice de Corrélation")

# TODO 6 : Votre code ici


# ============================================================================
# TODO 7 : BONUS - Filtres interactifs
# ============================================================================
# Indice : Utilisez st.columns(2) pour créer 2 colonnes de filtres
# - Colonne 1 : Slider pour filtrer par revenu (st.slider)
#   min_income, max_income = st.slider("Revenu", min_value, max_value, (min_value, max_value))
# - Colonne 2 : Multiselect pour filtrer par éducation (st.multiselect)
#   selected_education = st.multiselect("Niveau d'éducation", options=df["Education"].unique())
# Appliquer les filtres sur df et réafficher les graphiques

st.markdown("---")
st.subheader("🔍 Filtres Interactifs (BONUS)")

# TODO 7 : Votre code ici


# ============================================================================
# TODO 8 : Téléchargement CSV
# ============================================================================
# Indice : Utilisez st.download_button() avec :
# - label="📥 Télécharger les données (CSV)"
# - data=df.to_csv(index=False).encode('utf-8')
# - file_name="loan_data.csv"
# - mime="text/csv"

st.markdown("---")

# TODO 8 : Votre code ici


# ============================================================================
# AIDE MÉMOIRE - Fonctions Plotly utiles
# ============================================================================
# px.histogram() : Histogramme
# px.box() : Box plot
# px.bar() : Graphique en barres
# px.pie() : Camembert
# px.scatter() : Nuage de points
# go.Heatmap() : Carte de chaleur
# fig.add_vline() : Ligne verticale
# fig.add_annotation() : Annotation sur le graphique
# fig.update_layout() : Modifier le layout (titre, axes, etc.)
# use_container_width=True : Utiliser toute la largeur du container (IMPORTANT pour responsive)
# ============================================================================

# ============================================================================
# AIDE MÉMOIRE - Fonctions Pandas utiles
# ============================================================================
# df.groupby() : Grouper par colonne(s)
# df.value_counts() : Compter les occurrences
# df.mean() : Moyenne
# df.median() : Médiane
# df.quantile() : Quantile (ex: 0.25 pour Q1)
# df.select_dtypes() : Sélectionner par type de colonne
# df.corr() : Matrice de corrélation
# df.to_csv() : Convertir en CSV
# ============================================================================
