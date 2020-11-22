import streamlit as st
import numpy as np
import cv2
from PIL import Image

from core import Kmean, Pixel

# * Header
st.write("""
# Color Reductor / Pixel Reductor
""")

# * Sidebar
st.sidebar.header('User Input Parameters')
st.sidebar.subheader('Select image')
uploaded_file = st.sidebar.file_uploader('')

color_redu = st.sidebar.checkbox('Color Reduction')
K = st.sidebar.slider('Color Quantity', 1, 30, 8)

pixel_redu = st.sidebar.checkbox('Pixel Reduction')
PX = st.sidebar.slider('Pixel Quantity', 1, 256, 16)

# * Page

if uploaded_file is None:
    st.write("""### Please select an image from the menu on the left side""")



if uploaded_file is not None:

    PIL_img = Image.open(uploaded_file)
    img = np.array(PIL_img)

    if (color_redu==True) and (pixel_redu==False):
        output_img, color_list = Kmean.k_mean(img, K=K,  K_iter=1, criteria_iter=5, criteria_eps=1.0)

    elif (color_redu==False) and (pixel_redu==True):
        output_img = Pixel.pixeleted(img, PX)
    
    elif (color_redu==True) and (pixel_redu==True):
        output_img, color_list = Kmean.k_mean(img, K=K,  K_iter=1, criteria_iter=5, criteria_eps=1.0)
        output_img = Pixel.pixeleted(output_img, PX)
    
    else:
        output_img = img


    #* Layout
    img_col = st.beta_columns(2)

    img_col[0].write("""### **Original IMG**""")
    img_col[0].image(PIL_img, use_column_width=True)

    img_col[1].write("""### **Processed IMG**""")
    img_col[1].image(output_img, use_column_width=True)