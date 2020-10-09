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
    eps = args.eps/255.0
    delta = torch.zeros(x.size()).cuda()
    q = torch.zeros(x.size()).cuda()

    p = get_probabilities(model, x, y)
    perm = torch.randperm(x.size(0)*x.size(1)*x.size(1))

    for iteration, pixel in enumerate(perm):
        if iteration == args.num_iterations:
            break

        c, w, h = get_tensor_pixel_indices(pixel)
        q[c, w, h] = 1

        p_prim_left = get_probabilities(model, (x + delta + eps * q).clamp(0, 1), y)

        if p_prim_left < p:
            delta = delta+eps*q
            p = p_prim_left
        
        else:
            p_prim_right = get_probabilities(model, (x + delta - eps * q).clamp(0, 1), y)
            if p_prim_right < p:
                delta = delta-eps*q
                p = p_prim_left

        q[c, w, h] = 0

    return delta


def main():
    time = get_current_time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='dataset/imagenet-airplanes-images.pt')
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--save_file_name', type=str, default='results/simba-' + time + '.pt')
    args = parser.parse_args()

    model = MODELS_DICT.get(args.model).cuda()
    dataset = torch.load(args.dataset)
    results = []

    for image in dataset:
        label = torch.argmax(model(image.cuda().unsqueeze(0)))
        delta = simba_pixels(model, image.cuda(), label.cuda(), args)
        adversarial_example = image.cuda() + delta
        adversarial_label = torch.argmax(model(adversarial_example.unsqueeze(0)))

        print('Original prediction: ' + str(label.item()))
        print('Adversarial prediction: ' + str(adversarial_label.item()))

        results.append(adversarial_example.cpu())

    torch.save({'results': results, 'args': args}, args.save_file_name)


if __name__ == '__main__':
    main()
