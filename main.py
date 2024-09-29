import streamlit as st
import requests
from PIL import Image
import io

# Initialize the inference client
API_URL = "https://detect.roboflow.com"
API_KEY = "GqnM7y9zQj8VpTm5adfE"
MODEL_ID = "construction-safety-gsnvb/1"

def infer_image(image):
    """Send image to the API and return the inference results."""
    # Convert the image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()

    # Send image to API for inference
    response = requests.post(
        f"{API_URL}/{MODEL_ID}",
        files={"file": img_bytes},
        params={"api_key": API_KEY}
    )

    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to get a response from the API.")
        return None

# Streamlit web app
st.title("Construction Safety Detection")

# File uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the image
    image = Image.open(uploaded_image)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Call the inference API and display the results
    st.write("Processing...")

    result = infer_image(image)

    if result:
        st.write("Detection Results:")

        for prediction in result.get("predictions", []):
            class_name = prediction["class"]
            confidence = prediction["confidence"]
            st.write(f"Class: {class_name}, Confidence: {confidence:.2f}")
