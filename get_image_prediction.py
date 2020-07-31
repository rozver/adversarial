import torch
import torchvision
from torchvision import transforms
from PIL import Image
import sys
import os


def predict(image_location):
    image_to_predict = Image.open(image_location)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if image_to_predict.size != (3, 224, 224):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    image_to_predict = transform(image_to_predict)
    image_to_predict = image_to_predict.view(1, 3, 224, 224)

    model = torchvision.models.resnet50(pretrained=True)
    model.eval()

    prediction = model(image_to_predict)
    return prediction


if __name__ == '__main__':
    if len(sys.argv) == 2:
        location = sys.argv[1]
        if os.path.exists(location):
            if location.endswith(('png', 'jpg', 'jpeg')):
                predicted_class = torch.argmax(predict(location)).item()
                print(predicted_class)
            else:
                print('The entered file is not an image with a format .png, .jpg or .jpeg!')
        else:
            print('Incorrect image path!')
    else:
        print('Please enter image path!')
