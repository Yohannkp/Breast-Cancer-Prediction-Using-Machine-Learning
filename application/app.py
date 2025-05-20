import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pickle
st.set_page_config(
    page_title="Prédiction Cancer du Sein",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger le modèle
model_path = os.path.join(os.path.dirname(__file__), "../logistic_regression_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier {model_path} est introuvable sur Railway !")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Titre et description
st.markdown(
    "<h1 style='color:#d6336c;'>🔬 Prédiction du Cancer du Sein</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='font-size:18px;'>Cette application prédit si une tumeur est <b style='color:#228be6;'>bénigne</b> ou <b style='color:#fa5252;'>maligne</b> selon les caractéristiques saisies.</p>",
    unsafe_allow_html=True
)

# Formulaire dans la sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2913/2913466.png", width=120)
st.sidebar.header("Entrer les caractéristiques")

def user_input_features():
    col1, col2, col3 = st.sidebar.columns(3)
    features = {}
    with col1:
        features['radius_mean'] = st.number_input("Rayon moyen", min_value=0.0, step=0.1)
        features['texture_mean'] = st.number_input("Texture moyenne", min_value=0.0, step=0.1)
        features['perimeter_mean'] = st.number_input("Périmètre moyen", min_value=0.0, step=0.1)
        features['area_mean'] = st.number_input("Aire moyenne", min_value=0.0, step=0.1)
        features['smoothness_mean'] = st.number_input("Lissage moyen", min_value=0.0, step=0.001)
        features['compactness_mean'] = st.number_input("Compacité moyenne", min_value=0.0, step=0.001)
        features['concavity_mean'] = st.number_input("Concavité moyenne", min_value=0.0, step=0.001)
        features['concave_points_mean'] = st.number_input("Points concaves moyens", min_value=0.0, step=0.001)
        features['symmetry_mean'] = st.number_input("Symétrie moyenne", min_value=0.0, step=0.001)
        features['fractal_dimension_mean'] = st.number_input("Dimension fractale moyenne", min_value=0.0, step=0.001)
    with col2:
        features['radius_se'] = st.number_input("Erreur std rayon", min_value=0.0, step=0.1)
        features['texture_se'] = st.number_input("Erreur std texture", min_value=0.0, step=0.1)
        features['perimeter_se'] = st.number_input("Erreur std périmètre", min_value=0.0, step=0.1)
        features['area_se'] = st.number_input("Erreur std aire", min_value=0.0, step=0.1)
        features['smoothness_se'] = st.number_input("Erreur std lissage", min_value=0.0, step=0.001)
        features['compactness_se'] = st.number_input("Erreur std compacité", min_value=0.0, step=0.001)
        features['concavity_se'] = st.number_input("Erreur std concavité", min_value=0.0, step=0.001)
        features['concave_points_se'] = st.number_input("Erreur std points concaves", min_value=0.0, step=0.001)
        features['symmetry_se'] = st.number_input("Erreur std symétrie", min_value=0.0, step=0.001)
        features['fractal_dimension_se'] = st.number_input("Erreur std dimension fractale", min_value=0.0, step=0.001)
    with col3:
        features['radius_worst'] = st.number_input("Rayon max", min_value=0.0, step=0.1)
        features['texture_worst'] = st.number_input("Texture max", min_value=0.0, step=0.1)
        features['perimeter_worst'] = st.number_input("Périmètre max", min_value=0.0, step=0.1)
        features['area_worst'] = st.number_input("Aire max", min_value=0.0, step=0.1)
        features['smoothness_worst'] = st.number_input("Lissage max", min_value=0.0, step=0.001)
        features['compactness_worst'] = st.number_input("Compacité max", min_value=0.0, step=0.001)
        features['concavity_worst'] = st.number_input("Concavité max", min_value=0.0, step=0.001)
        features['concave_points_worst'] = st.number_input("Points concaves max", min_value=0.0, step=0.001)
        features['symmetry_worst'] = st.number_input("Symétrie max", min_value=0.0, step=0.001)
        features['fractal_dimension_worst'] = st.number_input("Dimension fractale max", min_value=0.0, step=0.001)
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

# Afficher les données entrées
st.markdown("---")
st.subheader("📝 Caractéristiques fournies")
st.dataframe(input_df.style.highlight_max(axis=1, color='#d0ebff'), height=200)

# Prédiction
if st.button("Prédire", help="Cliquez pour obtenir la prédiction"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    st.markdown("---")
    if prediction[0] == 1:
        st.warning("🔴Résultat : Maligne", icon="🔴")
    else:
        st.success("🟢 Résultat : Bénigne", icon="🟢")
    st.markdown(
        f"""
        <div style='font-size:18px;'>
            <b>Probabilité Maligne :</b> <span style='color:#fa5252;'>{prediction_proba[0][1]:.2f}</span><br>
            <b>Probabilité Bénigne :</b> <span style='color:#228be6;'>{prediction_proba[0][0]:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True
    )