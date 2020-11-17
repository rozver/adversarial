from abc import ABC
import torch
import shutil
from pycocotools.coco import COCO
from torchvision.transforms import transforms
from torchvision.utils import save_image
import numpy as np
import os
import datasets


def normalize_names(location):
    images_names_capitalized = os.listdir(location)

    for image in images_names_capitalized:
        if not image.islower():
            shutil.copy(os.path.join(location, image), os.path.join(location, image.lower()))
            os.remove(os.path.join(location, image))


class Normalizer(torch.nn.Module, ABC):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class ImageNetPreprocessor:
    def __init__(self, location, model, rgb=True):
        self.location = location
        self.rgb = rgb
        self.model = model
        self.dataset_images = None
        self.labels = None

    def set_dataset_images(self):
        if self.rgb:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])

        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ])

        self.dataset_images = datasets.ImageNet(location=self.location, transform=transform)

    def get_dataset_images(self):
        return self.dataset_images

    def set_labels(self):
        if self.dataset_images is not None:
            normalize = Normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.labels = [torch.argmax(self.model(normalize(x.unsqueeze(0)))) for x in self.dataset_images]
        else:
            raise ValueError('Image dataset not set!')

    def get_labels(self):
        return self.labels

    def serialize(self):
        if self.dataset_images is not None and self.labels:
            images = self.dataset_images
            labels = self.labels

            if self.rgb:
                torch.save(images, self.location+'-images.pt')
                torch.save(labels, self.location + '-labels.pt')
            else:
                torch.save(images, self.location + '-images-grayscale.pt')
                torch.save(labels, self.location + '-labels-grayscale.pt')
        else:
            raise ValueError('Images and labels not set!')

    def run(self):
        self.set_dataset_images()
        self.set_labels()
        self.serialize()


class CocoCategoryPreprocessor:
    def __init__(self, location, category):
        self.location = location
        self.category = category

    def export_images_and_masks(self):
        categories_file = open(os.path.join(self.location, 'categories_list.txt'))
        file_read = categories_file.read()
        if '\'{}\''.format(self.category) in file_read:
            category_directory = os.path.join(self.location, self.category)
            if not os.path.exists(category_directory):
                os.mkdir(category_directory)
                os.mkdir(os.path.join(category_directory, 'images'))
                os.mkdir(os.path.join(category_directory, 'masks'))

            annotations_file_location = os.path.join(self.location, 'annotations/instances_val2017.json')
            images_folder_location = os.path.join(self.location, 'images/val2017')
            coco = COCO(annotations_file_location)

            cat_ids = coco.getCatIds(catNms=self.category)
            img_ids = coco.getImgIds(catIds=cat_ids)
            images_paths = coco.loadImgs(img_ids)

            for index, image in enumerate(images_paths):
                annotations_ids = coco.getAnnIds(imgIds=image['id'], catIds=cat_ids, iscrowd=None)
                annotations = coco.loadAnns(annotations_ids)

                shutil.copy(os.path.join(images_folder_location, image['file_name']),
                            os.path.join(category_directory, 'images/{}.png'.format(index)))

                mask = coco.annToMask(annotations[0])

                for i in range(len(annotations)):
                    mask = np.maximum(coco.annToMask(annotations[i]), mask)

                mask = torch.from_numpy(mask).float()

                shutil.copy(os.path.join(images_folder_location, image['file_name']),
                            os.path.join(category_directory, 'images/{}.png'.format(index)))

                save_image(mask, os.path.join(category_directory, 'masks/{}.png'.format(index)))
        else:
            raise ValueError('Incorrect category type specified!')

    def serialize(self):
        category_location = os.path.join(self.location, self.category)
        if os.path.exists(category_location):
            dataset = datasets.CocoCategory(self.location, self.category, transform=transforms.ToTensor())
            torch.save(dataset, category_location+'.pt')
        else:
            raise ValueError('Dataset images and masks not exported!')

    def run(self):
        self.export_images_and_masks()
        self.serialize()
