import os
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path

# load images from 

# Helper fucntion
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

# Helper function
def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


root_path = "/scratch/local/ssd/janhr/data/animals_original"

data = [] # create list of dicts which will later be transformed to a pandas dataframe
pathlist = Path(root_path).glob('**/*.jpg')
num_files = len(list(pathlist))
assert num_files == 117484, "Not all images were found" # check if all image files were found
pathlist = Path(root_path).glob('**/*.jpg')

classes_path =  list(Path(root_path).glob('*'))

counter = 0
for class_path in classes_path:
     # because path is object not string
    tmp_img_list = []
    print(f"class: {class_path.name}")
    for img_file in class_path.glob('*.jpg'):
        counter = counter+1
        progress = (counter/num_files)*100
        print(f'{progress: .2f}%',flush=True,end='\r')


        im = Image.open(str(img_file))

        # Crop max square from image
        img_cropped = crop_max_square(im)

        # Resize image to 64x64
        im_resized = img_cropped.resize((64,64))

        # # print(type(img))
        # im_resized = cv2.resize(img, (64,64))
        tmp_img_list.append(np.array(im_resized))
        # print(img.shape)

    
    # np.mean(np.array(tmp_img_list), axis=0)
    avg_img = np.mean(np.array(tmp_img_list), axis=0)
    avg_img = Image.fromarray(np.uint8(avg_img)).convert('RGB')
    avg_img.save(f"/users/janhr/unsup3d_extended/data/animal_avg_class_images/{class_path.name}.jpg")