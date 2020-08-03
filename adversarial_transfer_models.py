import torch
import torchvision
import argparse
import os
from predict import predict
from multiple_predictions_serializer import save_dictionary_as_csv


def get_models_dict():
    models = {
        'resnet50': torchvision.models.resnet50(pretrained=True).eval(),
        'resnet18': torchvision.models.resnet18(pretrained=True).eval(),
        'resnet152': torchvision.models.resnet152(pretrained=True).eval(),
        'alexnet': torchvision.models.alexnet(pretrained=True).eval(),
        'inception': torchvision.models.inception_v3(pretrained=True).eval(),
        'googlenet': torchvision.models.googlenet(pretrained=True).eval(),
        'vgg16': torchvision.models.vgg16(pretrained=True).eval(),
        'vgg19': torchvision.models.vgg19(pretrained=True).eval(),
        'squeezenet': torchvision.models.squeezenet1_1(pretrained=True).eval(),
    }

    return models


def get_model(model_name):
    models_dict = get_models_dict()
    return models_dict.get(model_name, models_dict.get('resnet50'))


def compare_predictions(csv_location, model):
    successful_transfers = 0
    total_images = 0
    with open(csv_location) as csv_file:
        for row in csv_file:
            image_location, label = row.split(',')
            current_model_prediction = torch.argmax(predict(image_location, model, is_tensor=False)).item()

            if current_model_prediction == int(label):
                successful_transfers += 1

            total_images += 1

    return successful_transfers/total_images


def summarize_predictions_scores(csv_location):
    scores_csv_location = csv_location[0:len(csv_location)-4]+'-scores.csv'
    scores_dict = {}
    models_dict = get_models_dict()
    for model_name in models_dict:
        model = models_dict[model_name]
        scores_dict[model_name] = compare_predictions(csv_location, model)

    save_dictionary_as_csv(scores_dict, scores_csv_location)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_location', type=str, required=True)
    parser.add_argument('--model', type=str, default='all')

    args = parser.parse_args()

    if os.path.exists(args.csv_location) and args.csv_location.endswith('.csv'):
        if args.model != 'all':
            if args.model not in get_models_dict().keys():
                print('Model not found, running for resnet50...')
                args.model = 'resnet50'

            model = get_model(args.model)
            score = compare_predictions(args.csv_location, model)
            print('Score for model ' + args.model + ' is ' + str(score))
        else:
            summarize_predictions_scores(args.csv_location)

    else:
        raise ValueError('Selected path is not a folder')


if __name__ == '__main__':
    main()
