import torch
from torch.nn.functional import softmax
from adversarial_transfer_models import get_models_dict
from pgd import get_current_time
import argparse

MODELS_DICT = get_models_dict()


def get_probabilities(model, x, y):
    with torch.no_grad():
        prediction = model(x.unsqueeze(0))
    prediction_softmax = softmax(prediction, 1)
    prediction_softmax_y = prediction_softmax[0][y]

    return prediction_softmax_y


def get_tensor_pixel_indices(pixel):
    h = pixel % 224
    pixel = pixel // 224
    w = pixel % 224
    pixel = pixel // 224
    c = pixel % 3

    return c, w, h


def simba_pixels(model, x, y, args, g):
    delta = torch.zeros(x.size()).cuda()
    q = torch.zeros(x.size()).cuda()

    p = get_probabilities(model, x, y)
    perm = torch.randperm(x.size(0) * x.size(1) * x.size(1))

    for iteration, pixel in enumerate(perm):
        if iteration == args.num_iterations:
            break

        c, w, h = get_tensor_pixel_indices(pixel)

        if g[c, w, h] != 0:
            q[c, w, h] = g[c, w, h]

            p_prim_left = get_probabilities(model, (x + delta + args.eps * q).clamp(0, 1), y)

            if p_prim_left < p:
                delta = delta + args.eps * q
                p = p_prim_left

            else:
                p_prim_right = get_probabilities(model, (x + delta - args.eps * q).clamp(0, 1), y)
                if p_prim_right < p:
                    delta = delta + args.eps * q
                    p = p_prim_left

            q[c, w, h] = 0

    return delta


def nes_gradient(model, x, y, sigma, n):
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


def main():
    time = get_current_time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=MODELS_DICT.keys(), default='resnet50')
    parser.add_argument('--dataset', type=str, default='dataset/imagenet-airplanes-images.pt')
    parser.add_argument('--masks', default=False, action='store_true')
    parser.add_argument('--attack_type', type=str, choices=['nes', 'simba'], default='simba')
    parser.add_argument('--eps', type=float, default=10)
    parser.add_argument('--num_iterations', type=int, default=1)
    parser.add_argument('--save_file_name', type=str, default='results/blackbox/' + time + '.pt')
    args = parser.parse_args()

    model = MODELS_DICT.get(args.model).cuda()

    if args.masks:
        dataset = torch.load(args.dataset)
    else:
        images = torch.load(args.dataset)
        masks = [torch.ones(images[0].size())]*images.__len__()
        dataset = zip(images, masks)

    adversarial_examples_list = []
    predictions_list = []

    for image, mask in dataset:
        with torch.no_grad():
            original_prediction = model(image.cuda().unsqueeze(0))
        label = torch.argmax(original_prediction)

        if args.attack_type == 'nes':
            grad = nes_gradient(model, image.cuda(), label, args.eps, args.num_iterations)
            adversarial_example = fgsm_grad(image.cuda(), grad, args.eps)
        else:
            delta = simba_pixels(model, image.cuda(), label.cuda(), args, mask.cuda())
            adversarial_example = (image.cuda() + delta).clamp(0, 1)

        with torch.no_grad():
            adversarial_prediction = model(adversarial_example.unsqueeze(0))

        adversarial_examples_list.append(adversarial_example.cpu())
        predictions_list.append({'original': original_prediction.cpu(),
                                 'adversarial': adversarial_prediction.cpu()})

    torch.save({'adversarial_examples': adversarial_examples_list,
                'predictions': predictions_list,
                'args': args},
               args.save_file_name)


if __name__ == '__main__':
    main()
