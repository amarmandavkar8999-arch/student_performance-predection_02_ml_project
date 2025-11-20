import streamlit as st
import pickle
import numpy as np

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    with open("student_score_prediction.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -------------------------
# Page Style
# -------------------------
st.set_page_config(page_title="Students Performance Prediction by Amar M", layout="centered")

st.markdown("""
    <h1 style='text-align:center; color:#2E86C1;'>
    Students Performance Prediction<br>by <b>Amar M</b>
    </h1>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;'>Fill in the input values below to get prediction.</p>",
    unsafe_allow_html=True
)

st.write("---")

# -------------------------
# Detect number of required input features
# -------------------------
try:
    num_features = model.n_features_in_
except:
    st.error("❌ Could not detect model input size. Please ensure the model is scikit-learn compatible.")
    st.stop()

# -------------------------
# Create dynamic inputs
# -------------------------
st.subheader("Enter input values")

inputs = []
cols = st.columns(2)

for i in range(num_features):
    col = cols[i % 2]
    val = col.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

# -------------------------
# Predict
# -------------------------
if st.button("Predict"):
    try:
        input_array = np.array([inputs])
        result = model.predict(input_array)
        st.success(f"🎯 Predicted Result: **{result[0]}**")
    except Exception as e:
        st.error(f"Error while predicting: {e}")
