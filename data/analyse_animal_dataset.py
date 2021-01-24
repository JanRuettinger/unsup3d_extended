import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

# load images from 

root_path = "/scratch/shared/beegfs/janhr/data/unsup3d_extended/animals_original/"

data = [] # create list of dicts which will later transformed to a pandas dataframe
pathlist = Path(root_path).glob('**/*.JPEG')

assert len(pathlist) == 117484 # check if all image files were found

for path in pathlist:
     # because path is object not string
    path_in_str = str(path)
    class_name = path.parent.parent
    image_name = path.stem
    img = cv2.imread(path_in_str)
    height, width, channels = img.shape
    img_dict = {'image_name': img_name, 'class': class_name, 'width': width, 'height': height, 'smallest_dimension': min(width, height)}
    data.append(img_dict)

df = pd.DataFrame(data, columns=['image_name', 'class', 'width', 'height', 'smallest_dimension'])

print("Describe summary of smallest dimension column")
print(df["smallest_dimension"].describe())

