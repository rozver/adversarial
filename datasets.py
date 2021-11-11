import torch
import torchvision
from PIL import Image
import os


# Class for a custom dataset - ImageNet
class ImageNet(torch.utils.data.Dataset):
    def __init__(self, location, transform=torchvision.transforms.ToTensor()):
        self.location = location
        self.transform = transform
        self.all_images = os.listdir(location)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        image_location = os.path.join(self.location, self.all_images[index])
        image = Image.open(image_location).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


# Class for a custom dataset for given COCO category
class CocoCategory(torch.utils.data.Dataset):
    def __init__(self, location, category, transform=torchvision.transforms.ToTensor()):
        self.category = category

        if category == 'all':
            self.location = os.path.join(location, 'categories')
            if os.path.exists(location):
                self.images = []
                self.masks = []

                for category in os.listdir(self.location):
                    category_location = os.path.join(self.location, category)
                    self.images.append(os.listdir(os.path.join(category_location, 'images')))
                    self.masks.append(os.listdir(os.path.join(category_location, 'masks')))
        else:
            self.location = os.path.join(location, category)
            if os.path.exists(location):
                self.images = os.listdir(os.path.join(self.location, 'images'))
                self.masks = os.listdir(os.path.join(self.location, 'masks'))

        if len(self.images) != len(self.masks):
            raise ValueError('Number of images and number of masks do not match!')

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_location = os.path.join(os.path.join(self.location, 'images'), self.images[index])
        mask_location = os.path.join(os.path.join(self.location, 'masks'), self.images[index])

        image = Image.open(image_location).convert("RGB")
        mask = Image.open(mask_location).convert("RGB")

        return [self.transform(image), self.transform(mask)]

    def get_category(self):
        return self.category
