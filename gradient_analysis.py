import torch
from torch import autograd
from model_utils import ARCHS_LIST, get_model, load_model, predict
from file_utils import validate_save_file_location
import argparse
from pgd import get_current_time
import os


def get_gradient(model, x, label, criterion, similarity_coeffs=None, mask=None):
    x = autograd.Variable(x, requires_grad=True).cuda()

    if type(model) is list:
        if similarity_coeffs is None:
            similarity_coeffs = dict(zip([i for i in range(len(model))], [1 / len(model)] * len(model)))

        loss = torch.zeros(1).cuda()
        for arch, current_model in zip(similarity_coeffs.keys(), model):
            current_model.cuda()
            prediction = predict(current_model, x)
            current_loss = criterion(prediction, label)
            loss = torch.add(loss, similarity_coeffs[arch] * current_loss)
    else:
        prediction = predict(model, x)
        loss = criterion(prediction, label)

    if mask is not None:
        x.register_hook(lambda grad: grad * mask.float())

    gradient = autograd.grad(loss, x)[0]
    return gradient.cpu()


def get_sorted_order(grad, size):
    grad = torch.flatten(grad)
    if not 0 < size < grad.size(0):
        raise ValueError('Invalid size entered!')

    order = torch.argsort(grad.cpu(), descending=True)[:size]
    return order


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


def get_average(grad):
    grad_abs = torch.abs(grad)
    average = torch.sum(grad_abs).item() / grad_abs[grad_abs != 0].size().numel()
    return average


def get_category_average(grads, dataset_length):
    foreground_average = 0
    background_average = 0

    for foreground_grad, background_grad in grads:
        foreground_grad_average = get_average(foreground_grad)
        background_grad_average = get_average(background_grad)
        foreground_average += foreground_grad_average
        background_average += background_grad_average

    foreground_average /= dataset_length
    background_average /= dataset_length

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

            if args_dict['num_samples_per_class'] is None:
                num_samples = dataset.__len__()
            else:
                num_samples = args_dict['num_samples_per_class']

            for index, (image, mask) in enumerate(dataset):
                if index == num_samples:
                    break

                prediction = predict(model, image.cuda())
                label = torch.argmax(prediction, dim=1).cuda()

                grad = get_gradient(model, image, label, criterion)
                foreground_grad = grad*mask
                background_grad = grad - foreground_grad

                if args_dict['normalize_grads']:
                    foreground_grad, background_grad = normalize_grad(foreground_grad), normalize_grad(background_grad)
                category_grads.append([foreground_grad.cpu(), background_grad.cpu()])

            foreground_average, background_average = get_category_average(category_grads, dataset.__len__())
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
    torch.save({'averages': averages, 'args': args_dict},
               args_dict['save_file_location'])


if __name__ == '__main__':
    main()
