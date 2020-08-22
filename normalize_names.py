import os
import shutil
import argparse

# Parse dataset path from command argument
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

# Array for original images (uppercase)
images_alpha = os.listdir(args.path)

# For each of the images, copy it with changed filename (lowercase) and then delete it
for img in images_alpha:
    shutil.copy(os.path.join(args.path, img), os.path.join(args.path, img.lower()))
    os.remove(os.path.join(args.path, img))
