import torch
import torchvision
from torchvision import transforms
from PIL import Image
import sys
import os


def predict(x, model, is_tensor=True, use_gpu=False):
    if not is_tensor:
        image_to_predict = Image.open(x)

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
    else:
        image_to_predict = x
        if len(x.shape) != 4:
            image_to_predict = x.view(1, 3, 224, 224)

    if use_gpu:
        model = model.cuda()
        prediction = model(image_to_predict.cuda())
        return prediction.cpu().detach()
    else:
        prediction = model(image_to_predict)
        return prediction.detach()


def predict_multiple(images_batch, model, is_tensor=True, use_gpu=False):
    predictions = []
    for image in images_batch:
        predictions.append(predict(image, model, is_tensor, use_gpu))
    predictions = torch.cat(predictions)
    return predictions


if __name__ == '__main__':
    if len(sys.argv) == 2:
        location = sys.argv[1]
        if os.path.exists(location):
            if location.endswith(('png', 'jpg', 'jpeg')):
                model = torchvision.models.resnet50(pretrained=True).eval()
                predicted_class = torch.argmax(predict(location, model)).item()
                print(predicted_class)
            else:
                print('The entered file is not an image with a format .png, .jpg or .jpeg!')
        else:
            print('Incorrect image path!')
    else:
        print('Please enter image path!')
