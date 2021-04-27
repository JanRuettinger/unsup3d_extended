from PIL import Image
from pathlib import Path
import os
import splitfolders


IMG_DIM = 128
INPUT_PATH = "/scratch/local/ssd/janhr/data/dogs_square_ratio/"
OUTPUT_PATH = "/scratch/local/ssd/janhr/data/dogs_128/" 

#### Helper fucntion ####
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

def drop_empty_folders(directory):
    """Verify that every empty folder removed in local storage."""

    for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
        if not dirnames and not filenames:
            os.rmdir(dirpath)

def preprocess_and_move_images(data_path):
    for each_file in data_path.glob('**/*.jpg'): # grabs all files
        trg_path = each_file.parent.parent # gets the parent of the folder 

        img = Image.open(str(each_file))
        # Crop max square from image
        img_cropped = crop_max_square(img)
        # Resize image to IMG_DIMxIMG_DIM
        img_resized = img_cropped.resize((IMG_DIM,IMG_DIM))

        img_resized.save(str(each_file), quality=95)
        each_file.rename(data_path.joinpath(each_file.name)) # moves to parent folder

# # 1. Split dataset into train/val/test
splitfolders.ratio(INPUT_PATH, output=OUTPUT_PATH, seed=1337, ratio=(.8, 0.1,0.1))

print("Images were split into train, val and test.")

# # 2. Move and resize images
#     # 1. Resize and crop images
#     # 2. Move images from train/class_name/img.jpg to train/img.jpg
train_path = Path(OUTPUT_PATH+"/train")
val_path = Path(OUTPUT_PATH+"/val")
test_path = Path(OUTPUT_PATH+"/test")

preprocess_and_move_images(train_path)
preprocess_and_move_images(val_path)
preprocess_and_move_images(test_path)

print("Images are now in the correct shape and size as well as in the correct folder structure.")

# sanity check
num_files = len(list(Path(OUTPUT_PATH).glob('**/*.jpg')))
# assert num_files == 117484, "Not all images were found" 


drop_empty_folders(str(train_path))
drop_empty_folders(str(val_path))
drop_empty_folders(str(test_path))

print("Unnecessary folders were deleted. Everything complete.")