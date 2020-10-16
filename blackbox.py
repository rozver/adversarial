import torch
from torch.nn.functional import softmax
from adversarial_transfer_models import get_models_dict
from pgd import get_current_time
import argparse

MODELS_DICT = get_models_dict()


def get_probabilities(model, x, y):
    prediction = model(x.unsqueeze(0).cuda())
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


def simba_pixels(model, x, y, args):
    eps = args.eps / 255.0
    delta = torch.zeros(x.size()).cuda()
    q = torch.zeros(x.size()).cuda()

    p = get_probabilities(model, x, y)
    perm = torch.randperm(x.size(0) * x.size(1) * x.size(1))

    for iteration, pixel in enumerate(perm):
        if iteration == args.num_iterations:
            break

        c, w, h = get_tensor_pixel_indices(pixel)
        q[c, w, h] = 1

        p_prim_left = get_probabilities(model, (x + delta + eps * q).clamp(0, 1), y)

        if p_prim_left < p:
            delta = delta + eps * q
            p = p_prim_left

        else:
            p_prim_right = get_probabilities(model, (x + delta - eps * q).clamp(0, 1), y)
            if p_prim_right < p:
                delta = delta + eps * q
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
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='dataset/imagenet-airplanes-images.pt')
    parser.add_argument('--eps', type=float, default=10)
    parser.add_argument('--num_iterations', type=int, default=1)
    parser.add_argument('--save_file_name', type=str, default='results/blackbox-' + time + '.pt')
    args = parser.parse_args()

    model = MODELS_DICT.get(args.model).cuda()
    dataset = torch.load(args.dataset)

    adversarial_examples_nes_list = []
    adversarial_examples_simba_list = []
    predictions_nes = []
    predictions_simba = []

    for image in dataset:
        original_prediction = model(image.cuda().unsqueeze(0))
        label = torch.argmax(original_prediction)

        """
        grad = nes_gradient(model, image.cuda(), label, args.eps/255.0, args.num_iterations)
        adversarial_example_nes = fgsm_grad(image.cuda(), grad, args.eps/255.0)
        adversarial_predictions_nes = torch.argmax(model(adversarial_example_nes.unsqueeze(0)))
        adversarial_examples_nes_list.append(adversarial_example_nes.cpu())
        predictions_nes.append({'original': original_prediction, 'adversarial': adversarial_predictions_nes})
        """

        delta = simba_pixels(model, image.cuda(), label.cuda(), args)
        adversarial_example_simba = (image.cuda() + delta).clamp(0, 1)

        adversarial_predictions_simba = torch.argmax(model(adversarial_example_simba.unsqueeze(0)))
        print(adversarial_predictions_simba)

        adversarial_examples_simba_list.append(adversarial_example_simba.cpu())

        predictions_simba.append({'original': original_prediction, 'adversarial': adversarial_predictions_simba})

    torch.save({'nes': zip(adversarial_examples_nes_list, predictions_nes),
                'simba': zip(adversarial_examples_simba_list, predictions_simba),
                'args': args},
               args.save_file_name)


if __name__ == '__main__':
    main()
