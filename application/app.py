import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prédiction Cancer du Sein", page_icon=":microscope:", layout="wide")

# Charger le modèle
model = joblib.load(r"C:\Ce PC\FRANCE\Mes projets\Breast Cancer prédiction\logistic_regression_model.pkl")

# En-tête avec icône
st.markdown(
    "<h1 style='text-align: center; color: #e75480;'>🔬 Prédiction du Cancer du Sein</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #555;'>Entrez les caractéristiques de la tumeur pour prédire si elle est bénigne ou maligne.</p>",
    unsafe_allow_html=True,
)

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2965/2965567.png",
    width=120,
    caption="Cancer du Sein"
)
st.sidebar.header("Entrer les caractéristiques")

def user_input_features():
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        radius_mean = st.number_input("Rayon moyen", min_value=0.0, step=0.1)
        perimeter_mean = st.number_input("Périmètre moyen", min_value=0.0, step=0.1)
        smoothness_mean = st.number_input("Lissage moyen", min_value=0.0, step=0.001)
        concavity_mean = st.number_input("Concavité moyenne", min_value=0.0, step=0.001)
        symmetry_mean = st.number_input("Symétrie moyenne", min_value=0.0, step=0.001)
        radius_se = st.number_input("Erreur standard du rayon", min_value=0.0, step=0.1)
        perimeter_se = st.number_input("Erreur standard du périmètre", min_value=0.0, step=0.1)
        smoothness_se = st.number_input("Erreur standard du lissage", min_value=0.0, step=0.001)
        concavity_se = st.number_input("Erreur standard de la concavité", min_value=0.0, step=0.001)
        symmetry_se = st.number_input("Erreur standard de la symétrie", min_value=0.0, step=0.001)
    with col2:
        texture_mean = st.number_input("Texture moyenne", min_value=0.0, step=0.1)
        area_mean = st.number_input("Aire moyenne", min_value=0.0, step=0.1)
        compactness_mean = st.number_input("Compacité moyenne", min_value=0.0, step=0.001)
        concave_points_mean = st.number_input("Points concaves moyens", min_value=0.0, step=0.001)
        fractal_dimension_mean = st.number_input("Dimension fractale moyenne", min_value=0.0, step=0.001)
        texture_se = st.number_input("Erreur standard de la texture", min_value=0.0, step=0.1)
        area_se = st.number_input("Erreur standard de l'aire", min_value=0.0, step=0.1)
        compactness_se = st.number_input("Erreur standard de la compacité", min_value=0.0, step=0.001)
        concave_points_se = st.number_input("Erreur standard des points concaves", min_value=0.0, step=0.001)
        fractal_dimension_se = st.number_input("Erreur standard de la dimension fractale", min_value=0.0, step=0.001)
    with col3:
        radius_worst = st.number_input("Rayon maximal", min_value=0.0, step=0.1)
        texture_worst = st.number_input("Texture maximale", min_value=0.0, step=0.1)
        perimeter_worst = st.number_input("Périmètre maximal", min_value=0.0, step=0.1)
        area_worst = st.number_input("Aire maximale", min_value=0.0, step=0.1)
        smoothness_worst = st.number_input("Lissage maximal", min_value=0.0, step=0.001)
        compactness_worst = st.number_input("Compacité maximale", min_value=0.0, step=0.001)
        concavity_worst = st.number_input("Concavité maximale", min_value=0.0, step=0.001)
        concave_points_worst = st.number_input("Points concaves maximaux", min_value=0.0, step=0.001)
        symmetry_worst = st.number_input("Symétrie maximale", min_value=0.0, step=0.001)
        fractal_dimension_worst = st.number_input("Dimension fractale maximale", min_value=0.0, step=0.001)
    features = {
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave_points_mean': concave_points_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
        'radius_se': radius_se,
        'texture_se': texture_se,
        'perimeter_se': perimeter_se,
        'area_se': area_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'concave_points_se': concave_points_se,
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,
        'radius_worst': radius_worst,
        'texture_worst': texture_worst,
        'perimeter_worst': perimeter_worst,
        'area_worst': area_worst,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst,
        'concave_points_worst': concave_points_worst,
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst,
    }
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

st.markdown("---")
st.subheader("📝 Caractéristiques fournies")
st.dataframe(input_df, use_container_width=True)

if st.button("🔎 Prédire", use_container_width=True):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    st.markdown("---")
    if prediction[0] == 1:
        st.success("🌡️ **Résultat : Maligne**", icon="⚠️")
    else:
        st.info("🩺 **Résultat : Bénigne**", icon="✅")
    st.progress(prediction_proba[0][1])
    st.markdown(
        f"<div style='font-size:18px;'>"
        f"<b>Probabilité Maligne :</b> <span style='color:#e75480'>{prediction_proba[0][1]:.2f}</span><br>"
        f"<b>Probabilité Bénigne :</b> <span style='color:#4caf50'>{prediction_proba[0][0]:.2f}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )