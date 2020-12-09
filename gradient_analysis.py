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


def get_grad_dict(model, criterion, args):
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
    mean_grad = torch.cuda.FloatTensor([[[torch.mean(grad[0])]],
                                        [[torch.mean(grad[1])]],
                                        [[torch.mean(grad[2])]]]).repeat(1, grad.size(1), grad.size(2))
    std_grad = torch.cuda.FloatTensor([[[torch.std(grad[0])]],
                                       [[torch.std(grad[1])]],
                                       [[torch.std(grad[2])]]]).repeat(1, grad.size(1), grad.size(2))

    normalized_grad = (grad.cuda() - mean_grad) / std_grad
    return normalized_grad.cpu()


def normalize_grads_dict(grads_dict):
    for key in grads_dict.keys():
        grads = grads_dict[key]
        for i in range(0, len(grads)):
            grads[i] = normalize_grad(grads[i])
        grads_dict[key] = grads
    return grads_dict


def get_category_average(grads, dataset):
    foreground_average = torch.zeros(1)
    background_average = torch.zeros(1)
    for grad, (_, mask) in zip(grads, dataset):
        foreground_grad_average, background_grad_average = get_averages(grad, mask)
        foreground_average = foreground_average.sum(foreground_grad_average)
        background_average = background_average.sum(background_grad_average)

    foreground_average = torch.mean(foreground_average)
    background_average = torch.mean(background_average)

    return foreground_average, background_average


def get_averages_by_category(grads_dict, args):
    categories_averages = {}
    for category in grads_dict.keys():
        category_dataset = torch.load(os.path.join(args.dataset, category+'.pt'))
        foreground_average, background_average = get_category_average(grads_dict[category], category_dataset)
        categories_averages[category] = [foreground_average, background_average]
    return categories_averages


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

    grads_dict = get_grad_dict(model, criterion, args)
    averages_regular = get_averages_by_category(grads_dict, args)

    grads_dict_normalized = normalize_grads_dict(grads_dict)
    averages_normalized = get_averages_by_category(grads_dict_normalized, args)

    torch.save({'averages_regular': averages_regular, 'averages_normalized': averages_normalized, 'args': args},
               args.save_file_name)


if __name__ == '__main__':
    main()
