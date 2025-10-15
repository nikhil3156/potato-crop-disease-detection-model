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

.st-emotion-cache-16txtl3 {
    padding: 2rem 1rem 1rem;
}

.st-emotion-cache-1y4p8pa {
    width: 100%;
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #FFFFFF;
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
    text-align: center;
    text-decoration: none;
    display: inline-block;
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
    border-left-color: #4CAF50; /* Green */
}

.prediction-card-disease {
    border-left-color: #f44336; /* Red */
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


# --- Model Integration ---
# Load your trained model
try:
    model = load_model('model.h5')
except Exception as e:
    st.error(f"Error: The model file 'model.h5' could not be loaded. Please ensure it's in the same directory as app.py.", icon="🚨")
    st.error(f"Details: {e}")
    st.stop()


def predict_disease(image_data):
    """
    This function takes an image, preprocesses it, and returns the predicted disease and confidence.
    """
    # 1. Preprocess the image
    #    Make sure the target_size matches the input size of your model
    image = image_data.resize((256, 256))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # 2. Make a prediction
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    # 3. Get the class name
    # IMPORTANT: Make sure this list matches the order of classes your model was trained on!
    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    
    # Safety check for class index
    if predicted_class_index >= len(class_names):
        return "Unknown Class", 0.0

    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name, confidence


# --- Disease Information ---
DISEASE_INFO = {
    "Potato___Early_blight": {
        "title": "Early Blight",
        "description": "Early blight is a fungal disease caused by *Alternaria solani*. It primarily affects leaves and tubers. Symptoms include small, dark, circular to oval lesions, often with a characteristic 'target spot' or 'bull's-eye' appearance.",
        "remedies": [
            "Use disease-resistant potato varieties.",
            "Practice crop rotation with non-host crops.",
            "Apply fungicides preventatively, especially during warm, humid conditions.",
            "Ensure good air circulation and avoid overhead irrigation.",
            "Remove and destroy infected plant debris."
        ]
    },
    "Potato___Late_blight": {
        "title": "Late Blight",
        "description": "Late blight, caused by the oomycete *Phytophthora infestans*, is one of the most destructive potato diseases. It appears as pale green, water-soaked spots, often at the tips or edges of leaves. These spots enlarge rapidly and turn dark brown or black.",
        "remedies": [
            "Plant certified disease-free seed potatoes.",
            "Implement a preventative fungicide spray program.",
            "Ensure proper spacing between plants to promote airflow.",
            "Destroy cull piles and volunteer potato plants.",
            "Harvest during dry weather and ensure tubers are dry before storage."
        ]
    },
    "Potato___healthy": {
        "title": "Healthy",
        "description": "The plant appears to be healthy. No significant disease symptoms are detected on the leaf. Continue with good agricultural practices to maintain plant health.",
        "remedies": [
            "Continue regular monitoring for pests and diseases.",
            "Ensure balanced fertilization and proper irrigation.",
            "Maintain good field sanitation.",
            "Practice crop rotation to prevent soil-borne issues."
        ]
    }
}


# --- Main Application UI ---
st.title("🥔 Potato Crop Disease Detector 🌿")
st.markdown("### Upload an image of a potato leaf to detect if it's healthy or has a disease.")

# --- Sidebar ---
with st.sidebar:
    st.header("About This Project")
    st.markdown("""
    This web application uses a deep learning model (Convolutional Neural Network) to detect common diseases in potato crops from leaf images.

    **Created by:** [Your Name Here]
    """)
    st.markdown("---")
    st.markdown("Connect with me:")
    st.markdown("[LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/) | [Portfolio](https://www.google.com/)")
    st.markdown("*(Replace the links above with your actual profiles!)*")
    st.markdown("---")
    st.success("The model is loaded and the app is ready!")


# --- File Uploader and Main Content ---
col1, col2 = st.columns(2, gap="large")

with col1:
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Leaf Image.', use_column_width=True)
        st.markdown(
            """
            <div style='text-align: center; font-style: italic; color: #555;'>
                Image successfully uploaded. Click 'Predict' to analyze.
            </div>
            """, unsafe_allow_html=True
        )

with col2:
    st.markdown("## Prediction Results")
    st.markdown("The analysis of the leaf image will be displayed here.")

    if uploaded_file is not None and st.button('Predict Disease'):
        with st.spinner('Analyzing the leaf... Please wait.'):
            # Make prediction
            prediction, confidence = predict_disease(image)
            
            # Check if the prediction is valid
            if prediction in DISEASE_INFO:
                info = DISEASE_INFO[prediction]
                
                # Display result
                st.markdown("---")
                if "healthy" in prediction:
                    card_class = "prediction-card-healthy"
                else:
                    card_class = "prediction-card-disease"

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
                st.error("Could not classify the image. Please try a different one.")


    elif uploaded_file is None:
        st.info("Please upload an image to get started.")
