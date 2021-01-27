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


def shuffle_dataset(images, labels):
    dataset = list(zip(images.all_images, labels))
    np.random.shuffle(dataset)
    images.all_images, labels = zip(*dataset)
    return images, labels


def create_data_loaders(images, labels, batch_size=10, num_workers=4, shuffle=True):
    if shuffle:
        images, labels = shuffle_dataset(images, labels)
    images_loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    labels_loader = torch.utils.data.DataLoader(labels, batch_size=batch_size, num_workers=num_workers)
    return images_loader, labels_loader


class Normalizer(torch.nn.Module, ABC):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class ImageNetPreprocessor:
    def __init__(self, location, model, resize=True, rgb=True):
        if os.path.exists(location):
            self.location = location
        else:
            raise ValueError('Invalid dataset location!')
        self.model = model
        self.resize = resize
        self.rgb = rgb
        self.dataset_images = None
        self.labels = None

    def set_dataset_images(self):
        transform_list = [transforms.ToTensor()]
        if self.resize:
            transform_list.append(transforms.Resize(256))
            transform_list.append(transforms.CenterCrop(224))
        if not self.rgb:
            transform_list.append(transforms.Grayscale(num_output_channels=3))

        transform = transforms.Compose(transform_list)
        self.dataset_images = datasets.ImageNet(location=self.location, transform=transform)

    def set_labels(self):
        if self.dataset_images is not None:
            normalize = Normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.labels = [torch.argmax(self.model(normalize(x.unsqueeze(0)))) for x in self.dataset_images]
        else:
            raise ValueError('Image dataset not set!')

    def serialize(self):
        if self.dataset_images is not None:
            suffix_location = '-grayscale.pt' if not self.rgb else '.pt'
            images = self.dataset_images
            images_location = self.location + '-images' + suffix_location

            if self.labels is not None:
                labels = self.labels
                labels_location = self.location + '-labels' + suffix_location
            else:
                labels = None
                labels_location = None

            properties_location = self.location+suffix_location
            properties_dict = {
                'images': images_location,
                'labels': labels_location,
                'length': images.__len__()
            }

            torch.save(properties_dict, properties_location)
            torch.save(images, images_location)

            if labels is not None:
                torch.save(labels, labels_location)

        else:
            raise ValueError('Images and labels not set!')

    def run(self):
        self.set_dataset_images()
        self.set_labels()
        self.serialize()


class CocoCategoryPreprocessor:
    def __init__(self, location, category):
        if os.path.exists(location):
            self.location = location
        else:
            raise ValueError('Invalid dataset location!')
        self.category = category
        self.dataset = None

    def export_images_and_masks(self):
        categories_file = open(os.path.join(self.location, 'categories_list.txt'))
        file_read = categories_file.read()
        if '{}'.format(self.category) in file_read:
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

    def set_dataset(self):
        category_location = os.path.join(self.location, self.category)
        if os.path.exists(category_location):
            dataset = datasets.CocoCategory(self.location, self.category, transform=transforms.ToTensor())
            self.dataset = dataset
        else:
            raise ValueError('Dataset images and masks for the chosen category are not exported!')

    def serialize(self):
        category_location = os.path.join(self.location, self.category)
        if self.dataset is not None:
            torch.save(self.dataset, category_location + '.pt')
        else:
            raise ValueError('Dataset not set!')

    def run(self):
        self.export_images_and_masks()
        self.set_dataset()
        self.serialize()
