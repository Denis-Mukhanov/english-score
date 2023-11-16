import streamlit as st

# CSS стиль для центрирования заголовка
title_style = """
    <style>
    .title {
        text-align: center;
    }
    </style>
"""
# Вставляем CSS стиль перед заголовком
st.markdown(title_style, unsafe_allow_html=True)
# Заголовок
st.markdown("<h1 class='title'>Определение уровня сложности английского</h1>", unsafe_allow_html=True)
image = "https://i.postimg.cc/GpMx0rJ0/image.png"
st.image(image, caption="", use_column_width=True)
