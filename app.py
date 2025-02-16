import streamlit as st
import replicate
import os

# ชื่อแอปพลิเคชัน
st.title("Real-ESRGAN Image Enhancement")

# ฟังก์ชันสำหรับการประมวลผลรูปภาพ
def enhance_image(image, scale, face_enhance, api_token):
    # ตั้งค่า API Token
    os.environ["REPLICATE_API_TOKEN"] = api_token

    # เรียกใช้โมเดล Real-ESRGAN
    output = replicate.run(
        "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa",
        input={
            "image": image,
            "scale": scale,
            "face_enhance": face_enhance
        }
    )
    return output

# รับค่า API token จากผู้ใช้
api_token = st.text_input("Enter your Replicate API token:", type="password")

# อัพโหลดรูปภาพ
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงรูปภาพที่อัพโหลด
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # รับค่าพารามิเตอร์จากผู้ใช้
    scale = st.slider("Select scale factor (recommended: 2-4):", min_value=1.0, max_value=10.0, value=2.0, step=0.1)
    face_enhance = st.checkbox("Enable face enhancement", value=True)

    # ปุ่มสำหรับประมวลผลรูปภาพ
    if st.button("Enhance Image"):
        if not api_token:
            st.error("Please enter your Replicate API token.")
        else:
            with st.spinner("Enhancing image..."):
                try:
                    # เรียกใช้ฟังก์ชัน enhance_image
                    output = enhance_image(uploaded_file, scale, face_enhance, api_token)
                    st.image(output, caption="Enhanced Image.", use_column_width=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
