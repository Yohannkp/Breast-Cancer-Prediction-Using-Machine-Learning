import streamlit as st
import pandas as pd
import os
import pickle
import time

st.set_page_config(
    page_title="Prédiction Cancer du Sein",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer le style
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f8f0fc 0%, #e7f5ff 100%);
    }
    .stButton>button {
        background-color: #d6336c;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        padding: 0.5em 2em;
        transition: 0.2s;
        border: none;
    }
    .stButton>button:hover {
        background-color: #fa5252;
        color: #fff;
        transform: scale(1.05);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px #d0ebff;
    }
    .stSidebar {
        background: #000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Charger le modèle
model_path = os.path.join(os.path.dirname(__file__), "../logistic_regression_model.pkl")
if not os.path.exists(model_path):
    st.error(f"Le fichier {model_path} est introuvable sur Railway !")
    st.stop()
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Titre et description
st.markdown(
    "<h1 style='color:#d6336c; font-size: 3em; text-shadow: 1px 1px 2px #fff;'>🔬 Prédiction du Cancer du Sein</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='font-size:20px; color:#495057;'>Cette application prédit si une tumeur est <b style='color:#228be6;'>bénigne</b> ou <b style='color:#fa5252;'>maligne</b> selon les caractéristiques saisies.</p>",
    unsafe_allow_html=True
)

# Sidebar design
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2913/2913466.png", width=120)
st.sidebar.markdown("<h2 style='color:#1971c2;'>📝 Caractéristiques</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border:1px solid #d0ebff;'>", unsafe_allow_html=True)

def user_input_features():
    col1, col2, col3 = st.sidebar.columns(3)
    features = {}
    with col1:
        features['radius_mean'] = st.number_input("Rayon moyen", min_value=0.0, step=0.1, format="%.2f")
        features['texture_mean'] = st.number_input("Texture moyenne", min_value=0.0, step=0.1, format="%.2f")
        features['perimeter_mean'] = st.number_input("Périmètre moyen", min_value=0.0, step=0.1, format="%.2f")
        features['area_mean'] = st.number_input("Aire moyenne", min_value=0.0, step=0.1, format="%.2f")
        features['smoothness_mean'] = st.number_input("Lissage moyen", min_value=0.0, step=0.001, format="%.3f")
        features['compactness_mean'] = st.number_input("Compacité moyenne", min_value=0.0, step=0.001, format="%.3f")
        features['concavity_mean'] = st.number_input("Concavité moyenne", min_value=0.0, step=0.001, format="%.3f")
        features['concave_points_mean'] = st.number_input("Points concaves moyens", min_value=0.0, step=0.001, format="%.3f")
        features['symmetry_mean'] = st.number_input("Symétrie moyenne", min_value=0.0, step=0.001, format="%.3f")
        features['fractal_dimension_mean'] = st.number_input("Dimension fractale moyenne", min_value=0.0, step=0.001, format="%.3f")
    with col2:
        features['radius_se'] = st.number_input("Erreur std rayon", min_value=0.0, step=0.1, format="%.2f")
        features['texture_se'] = st.number_input("Erreur std texture", min_value=0.0, step=0.1, format="%.2f")
        features['perimeter_se'] = st.number_input("Erreur std périmètre", min_value=0.0, step=0.1, format="%.2f")
        features['area_se'] = st.number_input("Erreur std aire", min_value=0.0, step=0.1, format="%.2f")
        features['smoothness_se'] = st.number_input("Erreur std lissage", min_value=0.0, step=0.001, format="%.3f")
        features['compactness_se'] = st.number_input("Erreur std compacité", min_value=0.0, step=0.001, format="%.3f")
        features['concavity_se'] = st.number_input("Erreur std concavité", min_value=0.0, step=0.001, format="%.3f")
        features['concave_points_se'] = st.number_input("Erreur std points concaves", min_value=0.0, step=0.001, format="%.3f")
        features['symmetry_se'] = st.number_input("Erreur std symétrie", min_value=0.0, step=0.001, format="%.3f")
        features['fractal_dimension_se'] = st.number_input("Erreur std dimension fractale", min_value=0.0, step=0.001, format="%.3f")
    with col3:
        features['radius_worst'] = st.number_input("Rayon max", min_value=0.0, step=0.1, format="%.2f")
        features['texture_worst'] = st.number_input("Texture max", min_value=0.0, step=0.1, format="%.2f")
        features['perimeter_worst'] = st.number_input("Périmètre max", min_value=0.0, step=0.1, format="%.2f")
        features['area_worst'] = st.number_input("Aire max", min_value=0.0, step=0.1, format="%.2f")
        features['smoothness_worst'] = st.number_input("Lissage max", min_value=0.0, step=0.001, format="%.3f")
        features['compactness_worst'] = st.number_input("Compacité max", min_value=0.0, step=0.001, format="%.3f")
        features['concavity_worst'] = st.number_input("Concavité max", min_value=0.0, step=0.001, format="%.3f")
        features['concave_points_worst'] = st.number_input("Points concaves max", min_value=0.0, step=0.001, format="%.3f")
        features['symmetry_worst'] = st.number_input("Symétrie max", min_value=0.0, step=0.001, format="%.3f")
        features['fractal_dimension_worst'] = st.number_input("Dimension fractale max", min_value=0.0, step=0.001, format="%.3f")
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

# Afficher les données entrées
st.markdown("---")
st.subheader("📝 Caractéristiques fournies")
st.dataframe(input_df.style.highlight_max(axis=1, color='#d0ebff'), height=200)

# Prédiction avec animation
if st.button("✨ Prédire", help="Cliquez pour obtenir la prédiction"):
    with st.spinner('Analyse en cours...'):
        progress = st.progress(0)
        for i in range(1, 101, 10):
            time.sleep(0.05)
            progress.progress(i)
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        progress.empty()
    st.markdown("---")
    if prediction[0] == 1:
        # Animation pour mauvaise nouvelle (maligne)
        st.markdown(
            "<div style='background:#fff0f6; border-radius:12px; padding:1em; box-shadow:0 2px 8px #fa5252; text-align:center;'>"
            "<h2 style='color:#fa5252;'>🔴 Résultat : Maligne</h2>"
            "<div style='font-size:2em;'>💔😢</div>"
            "</div>",
            unsafe_allow_html=True
        )
        st.snow()  # Effet visuel, même si blanc, c'est le plus "dramatique" disponible
    else:
        st.markdown(
            "<div style='background:#e7f5ff; border-radius:12px; padding:1em; box-shadow:0 2px 8px #228be6; text-align:center;'>"
            "<h2 style='color:#228be6;'>🟢 Résultat : Bénigne</h2>"
            "</div>",
            unsafe_allow_html=True
        )
        st.balloons()
    st.markdown(
        f"""
        <div style='font-size:20px; margin-top:1em; text-align:center;'>
            <b>Probabilité Maligne :</b> <span style='color:#fa5252; font-size:1.3em;'>{prediction_proba[0][1]:.2f}</span><br>
            <b>Probabilité Bénigne :</b> <span style='color:#228be6; font-size:1.3em;'>{prediction_proba[0][0]:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True
    )