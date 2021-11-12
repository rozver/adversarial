import torch
from torch import autograd

from file_utils import validate_save_file_location
from model_utils import ARCHS_LIST, get_model, load_model
import argparse
from pgd import get_current_time
import os


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


def get_averages(grad, mask):
    grad_abs = grad*torch.sign(grad)

    num_values = mask.size(0) * mask.size(1) * mask.size(2)
    num_ones = torch.sum(mask)
    num_zeros = num_values - num_ones

    foreground_grad_sum = torch.sum(grad_abs * mask)
    background_grad_sum = torch.sum(grad_abs) - foreground_grad_sum

    foreground_grad_average = foreground_grad_sum / num_ones
    background_grad_average = background_grad_sum / num_zeros
    return foreground_grad_average, background_grad_average


def get_category_average(grads, dataset, num_samples):
    foreground_average = 0
    background_average = 0

    for grad, (_, mask) in zip(grads, dataset):
        foreground_grad_average, background_grad_average = get_averages(grad, mask)
        foreground_average += foreground_grad_average
        background_average += background_grad_average

    foreground_average /= num_samples
    background_average /= num_samples

    return foreground_average, background_average


def get_averages_by_category(grads_dict, args_dict):
    categories_averages = {}
    for category in grads_dict.keys():
        category_dataset = torch.load(os.path.join(args_dict['dataset'], category+'.pt'))
        foreground_average, background_average = get_category_average(grads_dict[category], category_dataset)
        categories_averages[category] = [foreground_average, background_average]
    return categories_averages


def get_averages_dict(model, criterion, args_dict):
    averages_dict = {}

    for category_file in os.listdir(args_dict['dataset']):
        category_grads = []
        if category_file.endswith('.pt'):
            dataset = torch.load(os.path.join(args_dict['dataset'], category_file))

            if dataset.__len__() == 0:
                continue

            if args_dict['num_samples_per_class'] is None or dataset.__len__() < args_dict['num_samples_per_class']:
                num_samples = dataset.__len__()
            else:
                num_samples = args_dict['num_samples_per_class']

            for index, (image, _) in enumerate(dataset):
                if index == num_samples:
                    break

                prediction = get_prediction(model, image.cuda())
                label = torch.argmax(prediction, dim=1).cuda()

                current_grad = get_gradient(model, image, label, criterion)
                if args_dict['normalize_grads']:
                    current_grad = normalize_grad(current_grad)
                category_grads.append(current_grad.cpu())

            foreground_average, background_average = get_category_average(category_grads, dataset, num_samples)
            averages_dict[dataset.category] = [foreground_average, background_average]

    return averages_dict


def main():
    time = get_current_time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=ARCHS_LIST, default='resnet50')
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--checkpoint_location', type=str, default=None)
    parser.add_argument('--from_robustness', default=False, action='store_true')
    parser.add_argument('--dataset', type=str, default='dataset/coco')
    parser.add_argument('--num_samples_per_class', type=int, default=None)
    parser.add_argument('--normalize_grads', default=False, action='store_true')
    parser.add_argument('--save_file_location', type=str, default='results/gradient/' + time + '.pt')
    args_dict = vars(parser.parse_args())

    validate_save_file_location(args_dict['save_file_location'])

    if args_dict['checkpoint_location'] is not None:
        model = load_model(location=args_dict['checkpoint_location'],
                           arch=args_dict['arch'],
                           from_robustness=args_dict['from_robustness']).cuda().eval()
    else:
        model = get_model(args_dict['arch'], True if [args_dict['pretrained']] else False).cuda().eval()

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    averages = get_averages_dict(model, criterion, args_dict)
    print(averages)
    torch.save({'averages': averages, 'args': args_dict},
               args_dict['save_file_location'])


if __name__ == '__main__':
    main()
