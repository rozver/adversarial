import os
import torch
import argparse
import torchvision
from PIL import Image
from torchvision import transforms


def predict(x, model, is_tensor=True, use_gpu=False):
    if not is_tensor:
        image_to_predict = Image.open(x)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        image_to_predict = transform(image_to_predict)
        image_to_predict = image_to_predict.unsqueeze(0)
    else:
        image_to_predict = x
        if len(x.shape) != 4:
            image_to_predict = image_to_predict.unsqueeze(0)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    if os.path.exists(args.image):
        if args.image.endswith(('png', 'jpg', 'jpeg')):
            model = torchvision.models.resnet50(pretrained=True).eval()
            predicted_class = torch.argmax(predict(args.image, model, is_tensor=False)).item()
            print(predicted_class)
        else:
            print('The entered file is not an image with a format .png, .jpg or .jpeg!')
    else:
        print('Incorrect image path!')
