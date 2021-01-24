from PIL import Image
from pathlib import Path
import shutil
import split_folders


IMG_DIM = 64
INPUT_PATH = "/scratch/local/ssd/janhr/data/animals_original" 
OUTPUT_PATH = "/scratch/local/ssd/janhr/data/animals" 

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


# 1. Split dataset into train/val/test
split_folders.ratio(input=INPUT_PATH, output=OUTPUT_PATH, seed=1337, ratio=(.8, 0.1,0.1))

print("Images were split into train, val and test.")

# 2. Move and resize images
    # 1. Resize and crop images
    # 2. Move images from train/class_name/img.jpg to train/img.jpg
train_path = OUTPUT_PATH+"/train"
val_path = OUTPUT_PATH+"/val"
test_path = OUTPUT_PATH+"/test"


for each_file in Path(train_path).glob('**/*.jpg') # grabs all files
    trg_path = each_file.parent.parent # gets the parent of the folder 

    # Crop max square from image
    im_cropped = crop_max_square(im)

    # Resize image to 64x64
    im_resized = img_cropped.resize((IMG_DIM,IMG_DIM))

    im_resized.save(str(trg_path), quality=95)
    each_file.rename(trg_path.joinpath(each_file.name)) # moves to parent folder.

print("Images are now in the correct shape and size as well as in the correct folder structure.")

for folder in Path(train_path).glob("**")
    Path.rmdir(folder)

for folder in Path(val_path).glob("**")
    Path.rmdir(folder)

for folder in Path(test_path).glob("**")
    Path.rmdir(folder)

print("Unnecessary folders were deleted. Everything complete.")
