import streamlit as st
import replicate
import os
from PIL import Image
import io

# Streamlit App Title
st.title("Real-ESRGAN Image Enhancement")

# Input for Replicate API Token
api_token = st.text_input("Enter your Replicate API token:", type="password")

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # User inputs for enhancement settings
    scale = st.slider("Select scale factor:", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    face_enhance = st.checkbox("Enable face enhancement", value=True)

    if st.button("Enhance Image"):
        if not api_token:
            st.error("Please enter your Replicate API token.")
        else:
            try:
                # Convert image to bytes for API
                img_byte_array = io.BytesIO()
                image.save(img_byte_array, format="PNG")  # Convert to PNG for API processing
                img_byte_array = img_byte_array.getvalue()

                # Set API token
                os.environ["REPLICATE_API_TOKEN"] = api_token

                # Call Replicate API
                with st.spinner("Enhancing image..."):
                    output = replicate.run(
                        "nightmareai/real-esrgan",
                        input={
                            "image": img_byte_array,
                            "scale": scale,
                            "face_enhance": face_enhance
                        }
                    )

                # Display enhanced image
                if output:
                    st.image(output, caption="Enhanced Image", use_column_width=True)
                else:
                    st.error("Failed to process the image. Please try again.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
