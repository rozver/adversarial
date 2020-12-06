import torch
from torch import autograd
import os
from model_utils import get_model, get_models_dict, load_model
import argparse
from pgd import get_current_time


def get_prediction(model, image):
    prediction = model(image.cuda().unsqueeze(0))
    if type(prediction) == tuple:
        return prediction[0]
    return prediction


def get_gradient(model, image, label, criterion):
    image = autograd.Variable(image, requires_grad=True).cuda()

    prediction = get_prediction(model, image)
    loss = criterion(prediction, label)

    grad = autograd.grad(loss, image)[0]
    return grad.cpu()


def get_averages(grad, mask):
    num_values = (mask.size(0)*mask.size(1)*mask.size(2))
    num_zeros = num_values - torch.sum(mask)
    num_ones = num_values-num_zeros

    foreground_grad_sum = torch.sum(grad*mask)
    background_grad_sum = torch.sum(grad)-foreground_grad_sum

    foreground_grad_average = foreground_grad_sum/num_ones
    background_grad_average = background_grad_sum/num_zeros
    return foreground_grad_average, background_grad_average


def main():
    time = get_current_time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=get_models_dict().keys(), default='resnet50')
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--checkpoint_location', type=str, default=None)
    parser.add_argument('--from_robustness', default=False, action='store_true')
    parser.add_argument('--save_file_name', type=str, default='results/gradient/' + time + '.pt')
    args = parser.parse_args()

    if args.checkpoint_location is not None:
        model = load_model(location=args.checkpoint_location,
                           arch=args.model,
                           from_robustness=args.from_robustness).cuda().eval()
    else:
        model = get_model(args.model, pretrained=args.pretrained).cuda().eval()

    results = {}
    dataset_location = 'dataset/coco'
    for category_file in os.listdir(dataset_location):
        if category_file.endswith('.pt'):
            success = 0
            images = torch.load(os.path.join(dataset_location, category_file))
            if images.__len__() == 0:
                continue
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            for image, mask in images:
                if mask.size(0) != 3:
                    mask = mask.expand(3, mask.size(0), mask.size(1))
                prediction = get_prediction(model, image)
                label = torch.argmax(prediction, dim=1)
                grad = get_gradient(model, image, label, criterion)
                foreground_grad_average, background_grad_average = get_averages(grad, mask)
                if abs(foreground_grad_average) > abs(background_grad_average):
                    success += 1

            results[images.category] = float(success)/images.__len__()

    torch.save({'results': results, 'args': args}, args.save_file_name)


if __name__ == '__main__':
    main()
