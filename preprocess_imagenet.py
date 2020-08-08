import torch
import torchvision
from torchvision import transforms
from custom_imagenet import CustomImageNet
from matplotlib import pyplot as plt
import requests
import numpy as np
from normalizer import Normalizer


def main():
    # Check whether the dataset should be inspected
    print('Do you want to inspect the dataset with its corresponding predicted labels? (yes/no)')
    whether_to_inspect = input()

    print('Use RGB version (grayscale otherwise)? (yes/no)')
    rgb_version = input()

    # Resize them to (224, 224, 3) and transform them to a PyTorch tensor
    if rgb_version == 'yes':
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

    # Normalize the dataset
    normalize = Normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Load the custom ImageNet dataset and slice it into batches
    dataset = CustomImageNet(location='./dataset/imagenet-dogs', transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)

    # Load pre-trained ResNet50 on ImageNet and enable eval mode
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()

    # For each image: get its corresponding label and save it to a list
    labels = []
    for image in data_loader:
        prediction = model(normalize(image))
        for i in prediction:
            labels.append((torch.argmax(i)))

    # Slice the labels into batches
    labels = torch.chunk(torch.from_numpy(np.array(labels)), 13)

    if whether_to_inspect == 'yes':
        # Load corresponding class name for the predicted labels
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/' \
              'raw/596b27d23537e5a1b5751d2b0481ef172f58b539/imagenet1000_clsid_to_human.txt'
        imagenet_classes = eval(requests.get(url).content)

        # Inspect the images with their corresponding lables
        for images_batch, labels_batch in zip(data_loader, labels):
            for image, label in zip(images_batch, labels_batch):
                plt.imshow(image.permute(1, 2, 0))
                plt.xlabel(imagenet_classes[label.item()])
                plt.show()

    # Serialize the images and the labels
    if rgb_version == 'yes':
        torch.save(dataset, './dataset/imagenet-dogs-images.pt')
        torch.save(labels, './dataset/imagenet-dogs-labels.pt')
    else:
        torch.save(dataset, './dataset/imagenet-dogs-images-grayscale.pt')
        torch.save(labels, './dataset/imagenet-dogs-labels-grayscale.pt')


if __name__ == '__main__':
    main()
