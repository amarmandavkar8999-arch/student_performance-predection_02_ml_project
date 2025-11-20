import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load the Model ---
# The model file name is taken directly from your prompt.
MODEL_FILE = 'student score prediction .pkl'

@st.cache_resource
def load_model(file_path):
    """Loads the pickled model object."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{file_path}' not found. "
                 "Please ensure the file is in the same directory as app.py.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model
model = load_model(MODEL_FILE)

# --- 2. Streamlit UI and Prediction Logic ---

st.title("🎓 Student Score Prediction by Amar M")
st.markdown("Predict the final score based on study and submission metrics.")

# Define the features based on the model's training data
# The features are Hours_Studied, Attendance, and Assignments_Submitted 
st.sidebar.header("Input Features")

# Input field for 'Hours_Studied' (Continuous/Float)
hours_studied = st.sidebar.number_input(
    "Hours Studied (per week):", 
    min_value=0.0, 
    max_value=100.0, 
    value=10.0, 
    step=0.5,
    help="Average number of hours the student studied."
)

# Input field for 'Attendance' (Percentage/Integer)
attendance = st.sidebar.slider(
    "Attendance (%):", 
    min_value=0, 
    max_value=100, 
    value=80, 
    step=1,
    help="Student's attendance percentage."
)

# Input field for 'Assignments_Submitted' (Count/Integer)
assignments_submitted = st.sidebar.number_input(
    "Assignments Submitted (out of 10):", 
    min_value=0, 
    max_value=10, 
    value=8, 
    step=1,
    help="Number of assignments the student submitted."
)


# --- Prediction Button and Output ---
if st.sidebar.button("Predict Score"):
    # 1. Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'Hours_Studied': [hours_studied],
        'Attendance': [attendance],
        'Assignments_Submitted': [assignments_submitted]
    })
    
    # Display the input data (optional)
    st.subheader("Input Provided:")
    st.dataframe(input_data)

    # 2. Make the prediction
    try:
        prediction = model.predict(input_data)[0]
        
        # 3. Display the result
        st.success(f"**Predicted Final Score:**")
        st.balloons()
        
        # Format the prediction to one decimal place
        st.metric(label="Predicted Score (0-100)", value=f"{prediction:.1f}")

        # Optional: Display Model Coefficients for insight
        st.markdown("---")
        st.subheader("Model Insights (Coefficients)")
        # Extract and display the coefficients and intercept from the loaded model
        coefficients = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Coefficient': model.coef_
        })
        st.dataframe(coefficients)
        st.caption(f"Intercept: {model.intercept_:.4f}")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
