import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_iris

# ------------------------------
# 🌸 Page Setup
# ------------------------------
st.set_page_config(
    page_title="Iris Flower Prediction 🌿",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------------
# 🌿 Load Model and Data
# ------------------------------
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

iris = load_iris()
class_names = iris.target_names

# ------------------------------
# 🎨 Custom CSS for styling
# ------------------------------
st.markdown("""
    <style>
    .result-card {
        background-color: #f9f9f9;
        border-radius: 15px;
        padding: 25px;
        margin-top: 30px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        text-align: center;
        transition: all 0.3s ease;
    }
    .result-card:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 15px rgba(0,0,0,0.25);
    }
    .result-text {
        font-size: 1.6rem;
        font-weight: bold;
        color: #2E8B57;
    }
    .emoji {
        font-size: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# 🌿 Sidebar Inputs
# ------------------------------
st.sidebar.title("🌿 Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.0, 10.0, 0.2)

# ------------------------------
# 🌸 Main Section
# ------------------------------
st.title("🌸 Iris Flower Species Prediction App")
st.write("""
This app predicts the **species of Iris flower** based on your input features.  
Use the sliders on the left to customize the measurements.
""")

if st.button("🔮 Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    predicted_species = class_names[prediction]

    # 🌺 Stylish Result Card
    st.markdown(f"""
        <div class='result-card'>
            <div class='emoji'>🌼</div>
            <div class='result-text'>Predicted Species: <br><b>{predicted_species.capitalize()}</b></div>
        </div>
    """, unsafe_allow_html=True)

    # 📊 Show probabilities
    st.subheader("Prediction Probabilities:")
    st.bar_chart({
        "Probability": probabilities
    })

    for name, prob in zip(class_names, probabilities):
        st.write(f"- **{name}**: {prob:.2f}")

else:
    st.info("👈 Adjust the sliders in the sidebar and click **Predict** to get started!")

# ------------------------------
# 🌺 Footer
# ------------------------------
st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Scikit-learn")
