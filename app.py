import streamlit as st # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore

# Load trained model
model = joblib.load("knn_model.pkl")

st.title("🌼 Iris Flower Prediction - KNN Classifier")

# Input sliders
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Species: 🌸 *{species[prediction[0]]}*")
