import streamlit as st
import replicate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.title("Real-ESRGAN Image Enhancement")

api_token = st.text_input("Enter your Replicate API token:", type="password")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    scale = st.slider("Select scale factor:", min_value=1.0, max_value=10.0, value=2.0, step=0.1)
    face_enhance = st.checkbox("Enable face enhancement", value=True)

    if st.button("Enhance Image"):
        if not api_token:
            st.error("Please enter your Replicate API token.")
        else:
            with st.spinner("Enhancing image..."):
                try:
                    os.environ["REPLICATE_API_TOKEN"] = api_token
                    output = replicate.run(
                        "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa",
                        input={
                            "image": uploaded_file,
                            "scale": scale,
                            "face_enhance": face_enhance
                        }
                    )
                    st.image(output, caption="Enhanced Image.", use_column_width=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
