import os
import shutil

# Path of the dataset
PATH = 'dataset/imagenet-dogs'

# Array for original images (uppercase)
images_alpha = os.listdir(PATH)

# For each of the images, copy it with changed filename (lowercase) and then delete it
for img in images_alpha:
    shutil.copy(os.path.join(PATH, img), os.path.join(PATH, img.lower()))
    os.remove(os.path.join(PATH, img))
