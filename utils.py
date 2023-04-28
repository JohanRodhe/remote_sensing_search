import json
import matplotlib.pyplot as plt
import nmslib
import numpy as np
import os
import streamlit as st

from PIL import Image
from transformers import CLIPProcessor, FlaxCLIPModel


def load_index(image_vector_file):
    filenames, image_vecs = [], []
    fvec = open(image_vector_file, "r")
    for line in fvec:
        cols = line.strip().split('\t')
        filename = cols[0]
        image_vec = np.array([float(x) for x in cols[1].split(',')])
        filenames.append(filename)
        image_vecs.append(image_vec)
    V = np.array(image_vecs)
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(V)
    index.createIndex({'post': 2}, print_progress=True)
    return filenames, index


@st.cache(allow_output_mutation=True)
def load_model(model_path, baseline_model):
    model = FlaxCLIPModel.from_pretrained(model_path)
    # processor = CLIPProcessor.from_pretrained(baseline_model)
    processor = CLIPProcessor.from_pretrained(model_path)
    return model, processor


@st.cache(allow_output_mutation=True)
def load_captions(caption_file):
    image2caption = {}
    with open(caption_file, "r") as fcap:
        for line in fcap:
            data = json.loads(line.strip())
            filename = data["filename"]
            captions = data["captions"]
            image2caption[filename] = captions
    return image2caption
