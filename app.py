import streamlit as st 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image  # Import the Image module from PIL

# Set page title and icon
st.set_page_config(
    page_title="Digit Detection",
    page_icon="🔢", 
)

# Load the digit detection model
model = load_model("digits_recognition_cnn.h5")

def load_image(image_file):
    img = Image.open(image_file)
    return img

# Preprocess the image for digit detection
def preprocess_image(image):
    resized_image = image.resize((28, 28))
    grayscale_image = resized_image.convert('L')
    normalized_image = np.array(grayscale_image) / 255.0
    reshaped_image = normalized_image.reshape(1, 28, 28, 1)
    return reshaped_image

# Function to perform digit detection
def detect_digit(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    digit = np.argmax(prediction)
    return digit

def main():
    st.title("Digit Detection") 
    menu = ["Home", "About", "Contact Us"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        
        image_file = st.file_uploader("Upload Image", type=['PNG', 'JPG', 'JPEG'])
        if image_file is not None:
            img = load_image(image_file)
            st.image(img)
            
            # Perform digit detection
            digit = detect_digit(img)
            st.write("Detected Digit:", digit)
            
    elif choice == "About": 
        st.title("About Us")
        # Your about section code here

    else:
        st.title("Get In Touch With Us")
        # Your contact form code here

if __name__ == '__main__':
    main()
