import torch
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import argparse
import os
import numpy as np


def get_coco_images_and_masks(location, category):
    transform = transforms.ToTensor()

    images = []
    masks = []

    annotations_file_location = os.path.join(location, 'annotations/instances_val2017.json')
    images_folder_location = os.path.join(location, 'images/val2017')
    coco = COCO(annotations_file_location)

    cat_ids = coco.getCatIds(catNms=category)
    img_ids = coco.getImgIds(catIds=cat_ids)
    images_paths = coco.loadImgs(img_ids)

    for image in images_paths:
        annotations_ids = coco.getAnnIds(imgIds=image['id'], catIds=cat_ids, iscrowd=None)
        annotations = coco.loadAnns(annotations_ids)

        image_pil = Image.open(os.path.join(images_folder_location, image['file_name']))
        image_tensor = transform(image_pil)

        mask = coco.annToMask(annotations[0])

        for i in range(len(annotations)):
            mask = np.maximum(coco.annToMask(annotations[i]), mask)

        mask = torch.from_numpy(mask)

        images.append(image_tensor)
        masks.append(mask)

    return [images, masks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--category', type=str, required=True)
    args = parser.parse_args()

    if os.path.exists(args.location):
        categories_file = open(os.path.join(args.location, 'categories_list.txt'))
        file_read = categories_file.read()
        if '\'{}\''.format(args.category) in file_read:
            categories_file.close()
            images_and_masks = get_coco_images_and_masks(location=args.location, category=args.category)
        else:
            raise ValueError('Incorrect COCO category specified!')
    else:
        raise ValueError('Incorrect dataset location specified!')

    torch.save(images_and_masks, os.path.join(args.location, args.category + '.pt'))


if __name__ == '__main__':
    main()
