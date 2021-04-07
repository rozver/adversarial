import torch
from torch.nn.functional import softmax
from model_utils import ARCHS_LIST, predict, get_model
from pgd import get_current_time, Attacker, PGD_DEFAULT_ARGS_DICT
from gradient_analysis import get_gradient
from transformations import Blur
from file_utils import validate_save_file_location
import random
import argparse


def get_simba_gradient(model, x, y, criterion, similarity_coeffs):
    grad = get_gradient(model, x, y, criterion, similarity_coeffs)
    grad_vector = torch.flatten(grad)
    return grad_vector


def normalize_gradient_vector(grad_vector):
    grad_normalized = softmax(torch.abs(grad_vector), dim=0)
    return grad_normalized.tolist()


def get_probabilities(model, x, y):
    with torch.no_grad():
        prediction = predict(model, x.unsqueeze(0))
        prediction_softmax = softmax(prediction, 1)
        prediction_softmax_y = prediction_softmax[0][y]
        return prediction_softmax_y


def get_tensor_coordinate_indices(coordinate, size):
    c, coordinate = divmod(coordinate, size[1] * size[2])
    w, h = divmod(coordinate, size[2])

    return c, w, h


def simba(model, x, y, args_dict, substitute_model, criterion, pgd_attacker):
    delta = torch.zeros_like(x).cuda()
    q = torch.zeros_like(x).cuda()
    available_coordinates = None
    similarity_coeffs = None
    conv = Blur()
    conv.parameters = [(9, 3)]

    if args_dict['select_ensembles'] and substitute_model is None:
        x.unsqueeze_(0)
        step = pgd_attacker.attack_step(x, 25/255.0, 1/255.0)
        substitute_model = pgd_attacker.selective_transfer(x, torch.ones_like(x), y, step)
        similarity_coeffs = pgd_attacker.similarity_coeffs
        x.squeeze_()

    p = get_probabilities(model, x, y)

    if not args_dict['gradient_masks']:
        perm = torch.randperm(x.size().numel()).tolist()
    else:
        perm = range(0, x.size().numel())
        available_coordinates = torch.ones(x.size().numel())

    for iteration in range(args_dict['num_iterations']):
        if args_dict['gradient_masks']:
            distribution = get_simba_gradient(substitute_model, x + delta, y, criterion, similarity_coeffs)
            distribution_normalized = normalize_gradient_vector(distribution*available_coordinates)
            coordinate = random.choices(perm, distribution_normalized)[0]
            available_coordinates[coordinate] = 0
        else:
            coordinate = perm[iteration]

        c, w, h = get_tensor_coordinate_indices(coordinate, x.size())

        q[c, w, h] = 1

        if args_dict['conv']:
            q = conv(q)[0]

        p_prim_left = get_probabilities(model, (x + delta + args_dict['eps'] * q).clamp(0, 1), y)

        if p_prim_left < p:
            delta = delta + args_dict['eps'] * q
            p = p_prim_left

        else:
            p_prim_right = get_probabilities(model, (x + delta - args_dict['eps'] * q).clamp(0, 1), y)
            if p_prim_right < p:
                delta = delta + args_dict['eps'] * q
                p = p_prim_left

        q.zero_()

    return delta


def nes_gradient(model, x, y, args_dict):
    sigma = args_dict['eps']
    n = args_dict['num_iterations']
    x_shape = x.size()
    g = torch.zeros(x_shape).cuda()
    mean = torch.zeros(x_shape).cuda()
    std = torch.ones(x_shape).cuda()

    for i in range(n):
        u = torch.normal(mean, std).cuda()
        pred = get_probabilities(model, (x+sigma*u).clamp(0, 1), y)
        g = g + pred*u
        pred = get_probabilities(model, (x-sigma*u).clamp(0, 1), y)
        g = g - pred*u

    return g/(2*n*sigma)


def fgsm_grad(image, grad, eps):
    adversarial_example = image + eps*grad.sign()
    return adversarial_example.detach()


def nes(model, image, label, args_dict, *args):
    return fgsm_grad(image, nes_gradient(model, image, label, args_dict), args_dict['eps'])-image


def main():
    time = get_current_time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=ARCHS_LIST, default='resnet50')
    parser.add_argument('--dataset', type=str, default='dataset/imagenet-airplanes-images.pt')
    parser.add_argument('--gradient', default=False, action='store_true')
    parser.add_argument('--attack_type', type=str, choices=['nes', 'simba'], default='simba')
    parser.add_argument('--conv', default=False, action='store_true')
    parser.add_argument('--substitute_model', type=str, choices=ARCHS_LIST, default='resnet152')
    parser.add_argument('--select_ensembles', default=False, action='store_true')
    parser.add_argument('--eps', type=float, default=10)
    parser.add_argument('--num_iterations', type=int, default=1)
    parser.add_argument('--save_file_location', type=str, default='results/blackbox/' + time + '.pt')
    args_dict = vars(parser.parse_args())

    validate_save_file_location(args_dict['save_file_location'])

    model = get_model(args_dict['model'], parameters='standard').cuda().eval()

    dataset = torch.load(args_dict['dataset'])

    adversarial_examples_list = []
    predictions_list = []
    substitute_model, criterion, pgd_attacker = None, None, None

    if args_dict['attack_type'] == 'nes':
        attack = nes
    else:
        attack = simba
        if args_dict['gradient_masks']:
            if args_dict['select_ensembles']:
                pgd_attacker = Attacker(model.cuda(), PGD_DEFAULT_ARGS_DICT)
                pgd_attacker.args_dict['label_shifts'] = 0
                pgd_attacker.available_surrogates_list = ARCHS_LIST
                pgd_attacker.available_surrogates_list.remove(args_dict['model'])
            else:
                substitute_model = get_model(args_dict['gradient_model'], parameters='standard').cuda().eval()

    for index, image in enumerate(dataset):
        with torch.no_grad():
            original_prediction = predict(model, image.cuda().unsqueeze(0))
        label = torch.argmax(original_prediction, dim=1)

        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        delta = attack(model, image.cuda(), label.cuda(), args_dict, substitute_model, criterion, pgd_attacker)
        adversarial_example = (image.cuda() + delta).clamp(0, 1)

        with torch.no_grad():
            adversarial_prediction = predict(model, adversarial_example.unsqueeze(0))

        adversarial_examples_list.append(adversarial_example.cpu())
        predictions_list.append({'original': original_prediction.cpu(),
                                 'adversarial': adversarial_prediction.cpu()})

    torch.save({'adversarial_examples': adversarial_examples_list,
                'predictions': predictions_list,
                'args_dict': args_dict},
               args_dict['save_file_location'])


if __name__ == '__main__':
    main()
