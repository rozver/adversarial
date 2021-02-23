import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
from model_utils import ARCHS_LIST, get_model


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
    parser.add_argument('--arch', type=str, choices=ARCHS_LIST, default='resnet50')
    parser.add_argument('--image', type=str, required=True)
    args_dict = vars(parser.parse_args())

    if os.path.exists(args_dict['image']):
        if args_dict['image'].endswith(('png', 'jpg', 'jpeg')):
            model = get_model(args_dict['arch'], parameters='standard').eval()
            predicted_class = torch.argmax(predict(args_dict['image'], model, is_tensor=False)).item()
            print(predicted_class)
        else:
            raise ValueError('The entered file is not an image with a format .png, .jpg or .jpeg!')
    else:
        raise ValueError('Incorrect image path!')

