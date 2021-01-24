import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

# load images from 

root_path = "/scratch/shared/beegfs/janhr/data/unsup3d_extended/animals_original/"

data = [] # create list of dicts which will later be transformed to a pandas dataframe
pathlist = Path(root_path).glob('**/*.jpg')
num_files = len(list(pathlist))
assert num_files == 117484, "Not all images were found" # check if all image files were found
pathlist = Path(root_path).glob('**/*.jpg')

counter = 0
for path in pathlist:
     # because path is object not string
    counter = counter+1
    progress = (counter/num_files)*100
    print(f'{progress: .2f}%',flush=True,end='\r')
    path_in_str = str(path)
    class_name = path.parent.name
    image_name = path.stem
    print(class_name)
    img = cv2.imread(path_in_str)
    height, width, channels = img.shape
    img_dict = {'image_name': image_name, 'class': class_name, 'width': width, 'height': height,}
    data.append(img_dict)

df = pd.DataFrame(data, columns=['image_name', 'class', 'width', 'height'])
df.to_csv("/users/janhr/unsup3d_extended/data/animals_dataset_dataframe.csv")

print(f'Counter: {counter}')
print("Summary width:")
print(df["width"].describe())
print("Summary height:")
print(df["height"].describe())

