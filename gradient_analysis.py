import torch
from torch import autograd
import os
from model_utils import MODELS_LIST, get_model, load_model
import argparse
from pgd import get_current_time


def get_prediction(model, image):
    prediction = model(image.unsqueeze(0))
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
    num_values = (mask.size(0) * mask.size(1) * mask.size(2))
    num_zeros = num_values - torch.sum(mask)
    num_ones = num_values - num_zeros

    foreground_grad_sum = torch.sum(grad * mask)
    background_grad_sum = torch.sum(grad) - foreground_grad_sum

    foreground_grad_average = foreground_grad_sum / num_ones
    background_grad_average = background_grad_sum / num_zeros
    return foreground_grad_average, background_grad_average


def compare_absolute_difference(grad, mask):
    foreground_grad_average, background_grad_average = get_averages(grad, mask)
    if abs(foreground_grad_average) > abs(background_grad_average):
        return 1
    return 0


def get_all_gradients(model, criterion, args):
    grads_dict = {}
    for category_file in os.listdir(args.dataset):
        category_grads = []
        if category_file.endswith('.pt'):
            images = torch.load(os.path.join(args.dataset, category_file))

            if images.__len__() == 0:
                continue
            for image, _ in images:
                prediction = get_prediction(model, image.cuda())
                label = torch.argmax(prediction, dim=1).cuda()

                current_grad = get_gradient(model, image, label, criterion)
                category_grads.append(current_grad)

            grads_dict[images.category] = category_grads

    return grads_dict


def normalize_grad(grad):
    mean_grad = torch.mean(grad)
    std_grad = torch.std(grad)

    normalized_grad = (grad-mean_grad)/std_grad
    return normalized_grad


def normalize_grads_dict(grads_dict):
    for key in grads_dict.keys():
        grads = grads_dict[key]
        for i in range(0, len(grads)):
            grads[i] = normalize_grad(grads[i])
        grads_dict[key] = grads
    return grads_dict


def main():
    time = get_current_time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=MODELS_LIST, default='resnet50')
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--checkpoint_location', type=str, default=None)
    parser.add_argument('--from_robustness', default=False, action='store_true')
    parser.add_argument('--dataset', type=str, default='dataset/coco')
    parser.add_argument('--save_file_name', type=str, default='results/gradient/' + time + '.pt')
    args = parser.parse_args()

    if args.checkpoint_location is not None:
        model = load_model(location=args.checkpoint_location,
                           arch=args.arch,
                           from_robustness=args.from_robustness).cuda().eval()
    else:
        model = get_model(args.arch, args.pretrained).cuda().eval()

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    grads_dict = get_all_gradients(model, criterion, args)
    grads_dict_normalized = normalize_grads_dict(grads_dict)

    torch.save({'grads': grads_dict, 'grads_normalized': grads_dict_normalized, 'args': args}, args.save_file_name)


if __name__ == '__main__':
    main()
