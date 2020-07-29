import os
from PIL import Image
import torch


# Class for a custom dataset - ImageNet
class CustomImageNet(torch.utils.data.Dataset):
    def __init__(self, location, transform):
        self.location = location
        self.transform = transform
        self.all_images = os.listdir(location)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image_location = os.path.join(self.location, self.all_images[idx])
        image = Image.open(image_location).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
