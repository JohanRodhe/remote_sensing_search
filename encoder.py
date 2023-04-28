import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def encode_image(image_file, model, processor):
    im = plt.imread(image_file)
    image = Image.fromarray((im * 255).astype(np.uint8))
    inputs = processor(images=image, return_tensors="jax")
    image_vec = model.get_image_features(**inputs)
    return np.array(image_vec).reshape(-1)

import os
from transformers  import CLIPProcessor, CLIPModel, FlaxCLIPModel
patches = os.listdir('../image_patches_rome2')
VECTORS_DIR = "vectors/"
IMAGES_DIR_ROME = "../image_patches_rome2/"
vector_file = os.path.join(VECTORS_DIR, "test_im_vectors_rome2.tsv")
model = FlaxCLIPModel.from_pretrained("flax-community/clip-rsicd-v2")
processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2")

print("Vectors written to {:s}".format(vector_file))
num_written = 0
fvec = open(vector_file, "w")
for image_file in patches:
    if num_written % 100 == 0:
        print("{:d} images processed".format(num_written))
    image_vec = encode_image(os.path.join(IMAGES_DIR_ROME, image_file), model, processor)
    image_vec_s = ",".join(["{:.7e}".format(x) for x in image_vec])
    fvec.write("{:s}\t{:s}\n".format(image_file, image_vec_s))
    num_written += 1
    
print("{:d} images processed, COMPLETE".format(num_written))
fvec.close()

