import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Potato Crop Disease Detection",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Attractive UI ---
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1598170845058-32b9d6a5da37?q=80&w=2070&auto=format&fit=crop");
    background-size: cover;
    background-attachment: fixed;
}

.reportview-container {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 2rem;
    margin: 1rem;
}

h1, h2, h3 {
    color: #4A4A4A;
    font-family: 'Arial', sans-serif;
    text-align: center;
}

.stButton>button {
    color: #FFFFFF;
    background-color: #4CAF50;
    border: none;
    padding: 15px 32px;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 12px;
    width: 100%;
    transition: all 0.3s ease 0s;
}

.stButton>button:hover {
    background-color: #45a049;
    box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
}

.prediction-card {
    background: #f9f9f9;
    border-left: 10px solid;
    margin: 1.5em 10px;
    padding: 1em 10px;
    border-radius: 8px;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
}

.prediction-card-healthy {
    border-left-color: #4CAF50;
}

.prediction-card-disease {
    border-left-color: #f44336;
}

.prediction-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 10px;
}

.confidence-score {
    font-size: 20px;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
try:
    model = load_model('model.h5')
except Exception as e:
    st.error("Error loading 'model.h5'. Make sure it's in the same folder as app.py.", icon="🚨")
    st.error(f"Details: {e}")
    st.stop()

# --- Prediction Function ---
def predict_disease(image_data):
    image = image_data.resize((256, 256))
    image_array = np.array(image) / 255.0
    img_input = np.expand_dims(image_array, axis=0)

    predictions = model.predict(img_input)
    predicted_class_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    if predicted_class_index >= len(class_names):
        return "Unknown Class", 0.0
    return class_names[predicted_class_index], confidence

# --- Disease Info ---
DISEASE_INFO = {
    "Potato___Early_blight": {
        "title": "Early Blight",
        "description": "Early blight is a fungal disease caused by Alternaria solani...",
        "remedies": [
            "Use disease-resistant potato varieties.",
            "Practice crop rotation with non-host crops.",
            "Apply fungicides preventatively.",
            "Ensure good air circulation and avoid overhead irrigation.",
            "Remove and destroy infected plant debris."
        ]
    },
    "Potato___Late_blight": {
        "title": "Late Blight",
        "description": "Late blight, caused by Phytophthora infestans, is destructive...",
        "remedies": [
            "Plant certified disease-free seed potatoes.",
            "Implement preventative fungicide spray programs.",
            "Ensure proper spacing between plants to promote airflow.",
            "Destroy volunteer plants and cull piles.",
            "Harvest during dry weather."
        ]
    },
    "Potato___healthy": {
        "title": "Healthy",
        "description": "No significant disease symptoms detected.",
        "remedies": [
            "Monitor regularly for pests and diseases.",
            "Ensure balanced fertilization and irrigation.",
            "Maintain good field sanitation.",
            "Practice crop rotation."
        ]
    }
}

# --- Main UI ---
st.title("🥔 Potato Crop Disease Detector 🌿")
st.markdown("### Upload a potato leaf image to detect diseases.")

# --- Sidebar ---
with st.sidebar:
    st.header("About This Project")
    st.markdown("""
    Deep learning model (CNN) to detect potato leaf diseases.
    """)
    st.markdown("---")
    st.markdown("Connect with me:")
    st.markdown("[LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/) | [Portfolio](https://www.google.com/)")
    st.markdown("---")
    st.success("The model is loaded and ready!")

# --- File Uploader ---
col1, col2 = st.columns(2, gap="large")

with col1:
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Leaf Image.', use_column_width=True)
        st.markdown(
            "<div style='text-align: center; font-style: italic; color: #555;'>Image uploaded. Click 'Predict' to analyze.</div>",
            unsafe_allow_html=True
        )

with col2:
    st.markdown("## Prediction Results")
    st.markdown("The analysis of the leaf image will appear here.")
    
    if uploaded_file and st.button('Predict Disease'):
        with st.spinner('Analyzing the leaf...'):
            prediction, confidence = predict_disease(image)
            if prediction in DISEASE_INFO:
                info = DISEASE_INFO[prediction]
                card_class = "prediction-card-healthy" if "healthy" in prediction else "prediction-card-disease"
                st.markdown(f"""
                <div class="prediction-card {card_class}">
                    <div class="prediction-title">Prediction: {info['title']}</div>
                    <div class="confidence-score">Confidence: {confidence:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("---")
                st.subheader("🔎 More Information")
                st.markdown(f"**Description:** {info['description']}")
                st.subheader("💡 Recommended Actions")
                for remedy in info['remedies']:
                    st.markdown(f"- {remedy}")
            else:
                st.error("Could not classify the image. Try a different one.")
    elif not uploaded_file:
        st.info("Please upload an image to start.")


