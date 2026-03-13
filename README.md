# Loan Approval Predictor

Application Streamlit pour predire l'approbation de prets et explorer les donnees de demandes de credit.

## Fonctionnalites

- Tableau de bord d'exploration (metriques, distributions, correlations)
- Interface de prediction interactive
- Affichage de la probabilite et des facteurs influents
- Onglet de performance du modele

## Structure du projet

- `app.py` : application Streamlit principale
- `data/` : fichiers CSV utilises par l'application
- `models/` : modeles et artefacts de preprocessing
- `.streamlit/config.toml` : configuration visuelle Streamlit

Tous les chargements utilisent des chemins relatifs compatibles avec GitHub et Streamlit Cloud.

## Lancer en local

1. Installer les dependances:

   pip install -r requirements.txt

2. Lancer l'application:

   streamlit run app.py

## Deploiement Streamlit Cloud

1. Pousser ce repository sur GitHub
2. Sur Streamlit Cloud, creer une nouvelle app depuis ce repository
3. Configurer le fichier principal sur: `app.py`
4. Verifier que `requirements.txt` est detecte automatiquement
