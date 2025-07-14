
import streamlit as st # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore

# Load the trained KNN model
model = joblib.load("knn_model.joblib")

st.title("🌸 KNN Iris Classifier App")

st.markdown("""
Welcome to the Iris Flower Predictor built using *KNN algorithm* and *Streamlit*.
- Use the sidebar sliders to input flower dimensions
- Or upload a CSV file for multiple predictions
""")

# Sidebar sliders
st.sidebar.header("📊 Input Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length", 0.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width", 0.0, 5.0, 3.5)
petal_length = st.sidebar.slider("Petal Length", 0.0, 8.0, 1.4)
petal_width = st.sidebar.slider("Petal Width", 0.0, 3.0, 0.2)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Single prediction
if st.button("🔍 Predict"):
    with st.spinner("Making prediction..."):
        prediction = model.predict(input_data)
        st.success(f"Predicted Iris Species: *{prediction[0]}*")

# Bulk prediction from CSV
st.markdown("---")
st.subheader("📁 Upload CSV for Bulk Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        predictions = model.predict(df)
        df['Prediction'] = predictions
        st.write(df)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Predictions", csv_data, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠ Error processing file: {e}")

# Footer
st.markdown("---")
st.markdown("Made with ❤ by [Deepanshu Pal](https://github.com/deepanshu2628)")