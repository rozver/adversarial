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

        self.images = []
        self.masks = []

        if category == 'all':
            self.location = os.path.join(location, 'categories')
            if os.path.exists(location):
                for category in os.listdir(self.location):
                    category_location = os.path.join(self.location, category)

                    images_directory = os.path.join(category_location, 'images')
                    masks_directory = os.path.join(category_location, 'masks')

                    for image, mask in zip(os.listdir(images_directory), os.listdir(masks_directory)):
                        self.images.append(os.path.join(images_directory, image))
                        self.masks.append(os.path.join(masks_directory, mask))

        else:
            self.location = os.path.join(location, category)
            if os.path.exists(location):
                images_directory = os.path.join(self.location, 'images')
                masks_directory = os.path.join(self.location, 'masks')

                for image, mask in zip(os.listdir(images_directory), os.listdir(masks_directory)):
                    self.images.append(os.path.join(images_directory, image))
                    self.masks.append(os.path.join(masks_directory, mask))

        if len(self.images) != len(self.masks):
            raise ValueError('Number of images and number of masks do not match!')

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("RGB")

        return [self.transform(image), self.transform(mask)]

    def get_category(self):
        return self.category
