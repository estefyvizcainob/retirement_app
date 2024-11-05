import streamlit as st
import joblib
import numpy as np

# Load the combined models
models = joblib.load('retirement_models.pkl')
decision_tree_model = models['decision_tree']
random_forest_model = models['random_forest']

# Title and description for the Streamlit app
st.title("Retirement Predictor")
st.write("Predict the likelihood of retirement based on age and 401K savings.")

# Option to choose the model
model_choice = st.radio("Choose a model for prediction:", ('Decision Tree', 'Random Forest'))

# Input fields for the app
age = st.number_input("Enter Age", min_value=18, max_value=100, value=50, step=1)
savings = st.number_input("Enter 401K Savings", min_value=0, step=1000, value=500000)

# Button to perform prediction
if st.button("Predict Retirement Status"):
    # Select the model based on user choice
    model = decision_tree_model if model_choice == 'Decision Tree' else random_forest_model

    # Prepare the input for prediction
    features = np.array([[age, savings]])
    prediction = model.predict(features)
    
    # Display the prediction result
    if prediction[0] == 1:
        st.success("This person is likely to be retired.")
    else:
        st.info("This person is likely not retired.")
