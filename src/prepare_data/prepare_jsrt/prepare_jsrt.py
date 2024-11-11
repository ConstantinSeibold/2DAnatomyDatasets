import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def process_img(input_filename):
    shape = (2048, 2048) # matrix size
    dtype = np.dtype('>u2') # big-endian unsigned integer (16bit)
    output_filename = "JPCLN001.PNG"

    # Reading.
    fid = open(input_filename, 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)
    
    image[image>2048] = 2048
    image = (image-image.min())/(image.max()-image.min())
    image = 1-image
    image = (image * 255).astype(np.uint8)
    
    return image

jsrt_images = os.getenv("JSRT_PATH")
new_path = os.getenv("NEW_JSRT_PATH")
os.makedirs(new_path, exist_ok=True)

for image in tqdm(os.listdir(jsrt_images)):
    img = process_img(os.path.join(jsrt_images, image))
    Image.fromarray(img).resize((1024,1024), 1).save(os.path.join(new_path, image.replace(".IMG",".png")))
    