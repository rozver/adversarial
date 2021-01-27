import torch
import os
from torchvision.utils import save_image
import argparse
from matplotlib import pyplot as plt

ARGS_DICT_KEYS_PGD = ['arch', 'checkpoint_location', 'from_robustness', 'dataset', 'masks', 'eps', 'norm',
                      'step_size', 'num_iterations', 'targeted', 'eot', 'transfer', 'save_file_location']
ARGS_DICT_KEYS_BLACKBOX = ['model', 'dataset', 'masks', 'gradient_masks', 'attack_type',
                           'gradient_model', 'eps', 'num_iterations', 'save_file_location']


def has_wrong_args(results, results_location):
    if 'args_dict' not in results.keys():
        print('File ' + results_location + ' does not have dictionary key args_dict')
        return True

    if 'blackbox' in results_location:
        keys = ARGS_DICT_KEYS_BLACKBOX
    else:
        keys = ARGS_DICT_KEYS_PGD

    for key in keys:
        if key not in results['args_dict'].keys():
            print('File ' + results_location + ' does not have args dictionary key ' + key)
            return True

    return False


def plot_adversarial_examples(results):
    for image in results['adversarial_examples']:
        plt.imshow(image.permute(1, 2, 0))
        plt.show()


def save_images(results, results_location, dataset):
    results_images_folder = os.path.dirname(results_location) + '/images/' + results_location.split('/')[-1][:-3]
    original_directory = results_images_folder + '/original/'
    adversarial_directory = results_images_folder + '/adversarial/'
    noises_directory = results_images_folder + '/noises/'
    masks_directory = results_images_folder + '/masks/'

    if not os.path.exists(results_images_folder):
        os.makedirs(original_directory)
        os.makedirs(adversarial_directory)
        os.makedirs(noises_directory)

        if results['args_dict']['masks']:
            os.makedirs(masks_directory)

    for index, (original_image, adversarial_example) in enumerate(zip(dataset, results['adversarial_examples'])):
        if results['args_dict']['masks']:
            original_image, mask = original_image
            save_image(mask, (masks_directory + str(index) + '.png'))

        if adversarial_example.size() != original_image.size():
            continue

        noise = original_image - adversarial_example

        save_image(original_image, (original_directory + str(index) + '.png'))
        save_image(adversarial_example, (adversarial_directory + str(index) + '.png'))
        save_image(noise, (noises_directory + str(index) + '.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--save_images', default=False, action='store_true')
    args_dict = vars(parser.parse_args())

    setups_and_results = []
    files = os.listdir(args_dict['location'])

    for file in files:
        if not file.endswith('.pt'):
            continue

        results_location = os.path.join(args_dict['location'], file)
        results = torch.load(results_location)
        successful_attacks = 0

        if has_wrong_args(results, results_location):
            continue

        if 'blackbox' in results['args_dict']['save_file_location']:
            results['args_dict']['masks'] = False

        for predictions in results['predictions']:
            original_class = torch.argmax(predictions['original']).item()
            adversarial_class = torch.argmax(predictions['adversarial']).item()

            if original_class != adversarial_class:
                successful_attacks += 1

        success_rate = round(successful_attacks / len(results['predictions']), 2)
        setups_and_results.append(str(results['args_dict']) + '\nAttack success rate: ' +
                                  str(success_rate) +
                                  '\n')

        if args_dict['save_images']:
            dataset = torch.load(results['args_dict']['dataset'])
            save_images(results, results_location, dataset)

    with open(os.path.join(args_dict['location'], 'setups_and_results.txt'), 'w') as file:
        for result in setups_and_results:
            file.write(str(result))
            file.write('\n')


if __name__ == '__main__':
    main()
