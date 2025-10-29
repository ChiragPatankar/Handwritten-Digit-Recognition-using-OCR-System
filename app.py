import sys
sys.path.insert(0, r'C:\tf')  # Add custom TensorFlow installation path

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras import models
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

MODEL_PATH = "tf-cnn-model.h5"

@st.cache_resource
def load_model_cached():
    """Load the model once and cache it"""
    model = models.load_model(MODEL_PATH, compile=False)
    return model

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # Resize to 28x28
    img_resized = cv2.resize(img_gray, (28, 28))
    
    # Reshape for model input
    img_final = img_resized.reshape(1, 28, 28, 1)
    
    return img_final, img_resized

def predict_digit(model, image):
    """Predict the digit from the preprocessed image"""
    prediction = model.predict(image, verbose=0)
    predicted_digit = np.argmax(prediction, axis=-1)[0]
    confidence = np.max(prediction) * 100
    return predicted_digit, confidence, prediction[0]

# Main app
def main():
    # Title and description
    st.title("🔢 Handwritten Digit Recognition")
    st.markdown("""
    ### MNIST Digit Classifier
    Upload an image of a handwritten digit (0-9) and the model will predict what digit it is!
    
    **Model Accuracy:** 99%+ on MNIST test dataset
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model_cached()
    
    # File uploader
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image of a handwritten digit (0-9)"
    )
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Your uploaded image", use_container_width=True)
        
        # Preprocess and predict
        with st.spinner("Processing..."):
            preprocessed_img, resized_img = preprocess_image(image)
            predicted_digit, confidence, probabilities = predict_digit(model, preprocessed_img)
        
        with col2:
            st.subheader("Preprocessed (28×28)")
            st.image(resized_img, caption="Grayscale 28×28", use_container_width=True)
        
        # Display prediction results
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        # Show predicted digit in large font
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
            <h1 style='font-size: 72px; margin: 0; color: #1f77b4;'>{predicted_digit}</h1>
            <p style='font-size: 24px; color: #555;'>Confidence: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show probability distribution
        st.markdown("### Probability Distribution")
        
        # Create a bar chart of probabilities
        prob_data = {str(i): float(probabilities[i] * 100) for i in range(10)}
        st.bar_chart(prob_data)
        
        # Show detailed probabilities
        with st.expander("View detailed probabilities"):
            for digit in range(10):
                prob_percent = probabilities[digit] * 100
                st.write(f"Digit {digit}: {prob_percent:.2f}%")
    
    else:
        # Show example images when no file is uploaded
        st.info("👆 Please upload an image to get started!")
        
        st.markdown("---")
        st.subheader("📝 Tips for best results:")
        st.markdown("""
        - Use a clear, well-lit image
        - The digit should be centered in the image
        - Black digit on white background works best
        - Avoid cluttered backgrounds
        - Single digit per image
        """)
        
        # Show sample predictions
        st.markdown("---")
        st.subheader("🖼️ Try these sample images:")
        st.markdown("You can find sample images in the `assets/images/` folder!")

if __name__ == "__main__":
    main()

