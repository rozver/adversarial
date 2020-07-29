import torch
import torchvision
from torchvision import transforms
from custom_imagenet import CustomImageNet
from matplotlib import pyplot as plt
import requests
import numpy as np

# Check whether the dataset should be inspected
print('Do you want to inspect the dataset with its corresponding predicted labels: (yes/no)')
whether_to_inspect = input()

# Transform the images and resize them to (224, 224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Load the custom ImageNet dataset and slice it into batches
dataset = CustomImageNet(location='./dataset/imagenet-dogs', transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)

# Load pre-trained ResNet50 on ImageNet and enable eval mode
model = torchvision.models.resnet50(pretrained=True)
model.eval()

# For each image: get its corresponding label and save it to a list
labels = []
for image in data_loader:
    prediction = model(image)
    for i in prediction:
        labels.append((torch.argmax(i)))

# Slice the labels into batches
labels = torch.chunk(torch.from_numpy(np.array(labels)), 4)

if whether_to_inspect == 'yes':
    # Load corresponding class name for the predicted labels
    url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/' \
          'raw/596b27d23537e5a1b5751d2b0481ef172f58b539/imagenet1000_clsid_to_human.txt'
    imagenet_classes = eval(requests.get(url).content)

    # Inspect the images with their corresponding lables
    for image_batch, label_batch in zip(data_loader, labels):
        for image, label in zip(image_batch, label_batch):
            plt.imshow(image.permute(1, 2, 0))
            plt.xlabel(imagenet_classes[label.item()])
            plt.show()

# Serialize the images and the labels
torch.save(data_loader, './dataset/imagenet-dogs-images.pt')
torch.save(labels, './dataset/imagenet-dogs-labels.pt')
