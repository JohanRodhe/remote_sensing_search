import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import numpy as np
from PIL import Image, ImageDraw
from transformers  import CLIPProcessor, CLIPModel, FlaxCLIPModel
import utils
import matplotlib.pyplot as plt


@st.cache_data
def get_base64_of_bin_file(bin_file):
   with open(bin_file, 'rb') as f:
      data = f.read()
   return base64.b64encode(data).decode()

def get_base64_of_PIL_img(image):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode()

@st.cache_resource
def load_models():
    model = FlaxCLIPModel.from_pretrained("flax-community/clip-rsicd-v2")
    processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2")
    return model, processor

@st.cache_resource
def load_index():
    vector_file = "./vectors/test_im_vectors_rome2.tsv"
    filenames, index = utils.load_index(vector_file)
    return filenames, index

image_name = ""
image = ""
IMAGES_DIR_ROME = "./images/image_patches_rome/"
img_file = 'Earth_from_Space_Rome_Italy.jpg'
model, processor = load_models()
filenames, index = load_index()

from streamlit.components.v1 import html
def show_image(img):
    html(
        f"""
        <body>
        <div id="container">
        <img src="data:image/jpeg;base64,{img}"/>
        </div>
        </body> 
        <script>
        const container = document.getElementById("container");
        const img = container.querySelector("img");
        var zoomActive = false;
        container.addEventListener("mousedown", toggleZoom);
        function toggleZoom(e) {{
            if (zoomActive) {{
                offZoom(e, false);
                container.removeEventListener("mousemove", onZoom);
            }}
            else {{
                onZoom(e, true);
                container.addEventListener("mousemove", onZoom);
            }}
            zoomActive = !zoomActive;
        }}
        function onZoom(e, transition) {{
            const x = e.clientX - e.target.offsetLeft;
            const y = e.clientY - e.target.offsetTop;
            img.style.transformOrigin = `${{x}}px ${{y}}px`;
            img.style.transform = "scale(18.5)";
        }}   
        function offZoom(e) {{
            img.style.transformOrigin = `center center`;
            img.style.transform = "scale(1)";
        }}
        </script>
        <style>
            #container {{
                box-shadow: 3px 3px 4px rgba(0, 0, 0, 0.3);
                overflow: hidden;
                width: 650px;

            }}
            img {{
                transform-origin: center center;
                object-fit: cover;
                max-width: 100%;
                width: 650px;
            }}
            img:hover {{
            }}
            </style>
        """,
        height=825,
    )
              
def mark_results_on_img(filenames):
    marked_img = Image.open(img_file)
    org_scale_x, org_scale_y = 11447, 11019
    new_scale_x, new_scale_y = 5724, 5510
    ratio_x = org_scale_x / new_scale_x
    ratio_y = org_scale_y / new_scale_y
    for name in filenames:
        x0 = (int(name.split('_')[1])) / ratio_x
        y0 = (int(name.split('_')[2][0:-4:1])) / ratio_y
        x1 = x0 + 224
        y1 = y0 + 224
        img1 = ImageDraw.Draw(marked_img)
        img1 = img1.rectangle([(x0, y0), (x1, y1)], outline ="red", width=7)

    return marked_img

 
def find_areas_from_text(input):
    with st.spinner("Searching..."):
        inputs = processor(text=[input], images=None, return_tensors="jax", padding=True)
        query_vec = model.get_text_features(**inputs)
        query_vec = np.asarray(query_vec)
        ids, distances = index.knnQuery(query_vec, k=11)
        result_filenames = [filenames[id] for id in ids]
        return result_filenames, distances

def find_areas_from_image(input):
    with st.spinner("Searching..."):
        inputs = processor(images=input, return_tensors="jax", padding=True)
        query_vec = model.get_image_features(**inputs)
        query_vec = np.asarray(query_vec)
        ids, distances = index.knnQuery(query_vec, k=11)
        result_filenames = [filenames[id] for id in ids]
        return result_filenames, distances


img = get_base64_of_PIL_img(Image.open(img_file))
images = ["patch_2912_8064.png", "patch_2464_10528.png", "patch_3808_448.png", "patch_4256_4480.png"]
examples = []
for i in range(4):
    examples.append(os.path.join(IMAGES_DIR_ROME, images[i]))

with st.sidebar:
    st.title("Remote sensing search in Rome")
    text = st.text_input("Write a description. Can be a sentance or just a word")
    st.write("Or select an image to search for similar areas")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(examples[0], width=100)
        b1 = st.button("Search", key=5)
    with col2:
        st.image(examples[1],width=100)
        b2 = st.button("Search", key=4)
    with col3:
        st.image(examples[2], width=100)
        b3 = st.button("Search", key=3)
    with col4:
        st.image(examples[3], width=100)
        b4= st.button("Search", key=6)

if text:
    st.text("Prompt: " + text)
    result_filenames, distances = find_areas_from_text(text)
    img = get_base64_of_PIL_img(mark_results_on_img(result_filenames[:5]))
elif b1:
    result_filenames, distances = find_areas_from_image(Image.open(examples[0]))
    img = get_base64_of_PIL_img(mark_results_on_img(result_filenames[:5]))
elif b2:
    st.text("Prompt: ")
    st.image(Image.open(examples[1]), width=150)
    result_filenames, distances = find_areas_from_image(Image.open(examples[1]))
    img = get_base64_of_PIL_img(mark_results_on_img(result_filenames[:5]))
elif b3:
    st.text("Prompt: ")
    st.image(Image.open(examples[2]), width=150)
    result_filenames, distances = find_areas_from_image(Image.open(examples[2]))
    img = get_base64_of_PIL_img(mark_results_on_img(result_filenames[:5]))
elif b4:
    st.text("Prompt: ")
    st.image(Image.open(examples[3]), width=150)
    result_filenames, distances = find_areas_from_image(Image.open(examples[3]))
    img = get_base64_of_PIL_img(mark_results_on_img(result_filenames[:5]))
else:
    img = get_base64_of_PIL_img(Image.open(img_file))
show_image(img)

#from streamlit_cropper import st_cropper
#def do_cropping():
#    aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
#    aspect_dict = {
#        "1:1": (1, 1),
#        "16:9": (16, 9),
#        "4:3": (4, 3),
#        "2:3": (2, 3),
#        "Free": None
#    }
#    aspect_ratio = aspect_dict[aspect_choice]
#
#    if img_file:
#        img = Image.open(img_file)
#        # Get a cropped image from the frontend
#        cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0000',
#                                    aspect_ratio=aspect_ratio)
#
#        # Manipulate cropped image at will
#        st.write("Preview")
#        _ = cropped_img.thumbnail((150,150))
#        st.image(cropped_img)
#
#        
#with st.container():
#    cropping = st.checkbox("Crop",key=1)
#    if cropping:
#        do_cropping()
  
#plot_results(result_filenames, distances)