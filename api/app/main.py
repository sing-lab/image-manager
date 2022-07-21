"""Main module defining the app"""
import sys
from io import BytesIO

import streamlit as st
from PIL import Image

sys.path.append("C:\\Users\\Mathias\\Documents\\Projets_Python\\image_manager\\src\\image_manager\\super_resolution")
from super_resolution import get_prediction
from streamlit_image_comparison import image_comparison


if __name__ == "__main__":
    uploaded_file = st.file_uploader("Choose an image")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image.filename = uploaded_file.name
        st.image(image, caption='Original image', use_column_width='always')

        if st.button('Process image'):
            sr_image = get_prediction(image)
            st.image(sr_image, caption='Super resolution image', use_column_width='always')

            image_comparison(
                img1=image.convert('RGB'),
                img2=sr_image.convert('RGB'),
                label1="Original image",
                label2="Super resolution image",
                width=700,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True,
            )

            buf = BytesIO()
            sr_image.save(buf, format="PNG")
            byte_image = buf.getvalue()

            download_button = st.download_button(
                label="Download image",
                data=byte_image,
                file_name=f"{uploaded_file.name}",
                mime="image/png")
