import torch
from pgd import TARGET_CLASS, Attacker
from model_utils import ARCHS_LIST, get_model, load_model
from file_utils import get_current_time, validate_save_file_location
import random
import argparse


def flip_values(x):
    x = x - torch.ones_like(x)
    return x*x


def get_patch_mask(mask):
    mask_width = mask.size(1)
    mask_height = mask.size(2)
    square_width = int(min(mask_width, mask_height) / 8)
    square_height = int(min(mask_width, mask_height) / 8)

    patch_mask = torch.zeros((mask_width, mask_height))
    patch_mask[:square_width, :square_height] = torch.ones(square_width, square_height)

    counter = 0
    while True:
        x_shift = random.randint(0, mask_width - square_width)
        y_shift = random.randint(0, mask_height - square_height)

        patch_mask_shifted = patch_mask.clone()
        patch_mask_shifted = torch.roll(patch_mask_shifted, x_shift, 0)
        patch_mask_shifted = torch.roll(patch_mask_shifted, y_shift, 1)

        if torch.sum(mask * patch_mask_shifted) == 0:
            return patch_mask_shifted

        if counter == 99:
            print('Could not find patch location that does not overlap with foreground, returning last mask!')
            return patch_mask_shifted

        counter += 1


def main():
    time = str(get_current_time())
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=ARCHS_LIST, default='resnet50')
    parser.add_argument('--checkpoint_location', type=str, default=None)
    parser.add_argument('--from_robustness', default=False, action='store_true')
    parser.add_argument('--dataset', type=str, default='dataset/imagenet-airplanes-images.pt')
    parser.add_argument('--masks', default=False, action='store_true')
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--norm', type=str, choices=['l2', 'linf'], default='linf')
    parser.add_argument('--step_size', type=float, default=1)
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--unadversarial', default=False, action='store_true')
    parser.add_argument('--targeted', default=False, action='store_true')
    parser.add_argument('--eot', default=False, action='store_true')
    parser.add_argument('--transfer', default=False, action='store_true')
    parser.add_argument('--selective_transfer', default=False, action='store_true')
    parser.add_argument('--num_surrogates', type=int, choices=range(0, len(ARCHS_LIST) - 1), default=5)
    parser.add_argument('--save_file_location', type=str, default='results/pgd_new_experiments/patch-' + time + '.pt')
    args_ns = parser.parse_args()

    args_dict = vars(args_ns)

    validate_save_file_location(args_dict['save_file_location'])

    args_dict['eps'], args_dict['step_size'] = args_dict['eps'] / 255.0, args_dict['step_size'] / 255.0

    print('Running PGD experiment with the following arguments:')
    print(str(args_dict)+'\n')

    if args_dict['checkpoint_location'] is None:
        model = get_model(arch=args_dict['arch'], parameters='standard').eval()
    else:
        model = load_model(location=args_dict['checkpoint_location'],
                           arch=args_dict['arch'],
                           from_robustness=args_dict['from_robustness']).eval()

    attacker = Attacker(model, args_dict)
    target = torch.zeros(1000)
    target[TARGET_CLASS] = 1

    print('Loading dataset...')
    if args_dict['masks']:
        dataset = torch.load(args_dict['dataset'])
        dataset_length = dataset.__len__()
    else:
        images = torch.load(args_dict['dataset'])
        masks = [torch.zeros_like(images[0])]*images.__len__()
        dataset = zip(images, masks)
        dataset_length = images.__len__()
    print('Finished!\n')

    adversarial_examples_list = []
    predictions_list = []

    print('Starting PGD...')
    for index, (image, mask) in enumerate(dataset):
        print('Image: ' + str(index+1) + '/' + str(dataset_length))
        original_prediction = model(image.unsqueeze(0))

        if not args_dict['targeted']:
            target = original_prediction

        patch_mask = get_patch_mask(mask)
        patch_mask = torch.cat(3 * [patch_mask]).view(image.size())

        image = image*flip_values(patch_mask)

        adversarial_example = attacker(image.cuda(), patch_mask.cuda(), target, False)
        adversarial_prediction = model(adversarial_example.unsqueeze(0))

        if args_dict['unadversarial'] or args_dict['targeted']:
            expression = torch.argmax(adversarial_prediction) == torch.argmax(target)
        else:
            expression = torch.argmax(adversarial_prediction) != torch.argmax(target)

        status = 'Success' if expression else 'Failure'
        print('Attack status: ' + status + '\n')

        adversarial_examples_list.append(adversarial_example)
        predictions_list.append({'original': original_prediction,
                                 'adversarial': adversarial_prediction})
    print('Finished!')

    print('Serializing results...')
    torch.save({'adversarial_examples': adversarial_examples_list,
                'predictions': predictions_list,
                'args_dict': args_dict},
               args_dict['save_file_location'])
    print('Finished!\n')


if __name__ == '__main__':
    main()
