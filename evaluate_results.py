import torch
import os
from torchvision.utils import save_image
import argparse
from matplotlib import pyplot as plt


def plot_adversarial_examples(results):
    for image in results['adversarial_examples']:
        plt.imshow(image.permute(1, 2, 0))
        plt.show()


def save_images(results, results_location, dataset):
    results_images_folder = os.path.dirname(results_location)+'/images/'+results_location.split('/')[-1][:-3]
    original_directory = results_images_folder + '/original/'
    adversarial_directory = results_images_folder + '/adversarial/'
    noises_directory = results_images_folder + '/noises/'
    masks_directory = results_images_folder + '/masks/'

    if not os.path.exists(results_images_folder):
        os.makedirs(original_directory)
        os.makedirs(adversarial_directory)
        os.makedirs(noises_directory)

        if results['args'].masks:
            os.makedirs(masks_directory)

    for index, (original_image, adversarial_example) in enumerate(zip(dataset, results['adversarial_examples'],)):
        if results['args'].masks:
            original_image, mask = original_image
            save_image(mask,  (masks_directory + str(index) + '.png'), normalize=True)

        noise = original_image - adversarial_example

        save_image(original_image, (original_directory + str(index) + '.png'), normalize=True)
        save_image(adversarial_example, (adversarial_directory + str(index) + '.png'), normalize=True)
        save_image(noise, (noises_directory + str(index) + '.png'), normalize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--save_images', default=False, action='store_true')
    args = parser.parse_args()

    setups_and_results = []
    files = os.listdir(args.location)

    for file in files:
        if not file.endswith('.pt'):
            continue

        results_location = os.path.join(args.location, file)
        results = torch.load(results_location)
        successful_attacks = 0

        if 'args' not in results.keys():
            print('File ' + results_location + ' does not have dictionary key args')
            continue

        if 'blackbox' in results['args'].save_file_name:
            results['args'].masks = False

        for predictions in results['predictions']:
            original_class = torch.argmax(predictions['original']).item()
            adversarial_class = torch.argmax(predictions['adversarial']).item()

            if original_class != adversarial_class:
                successful_attacks += 1

        success_rate = round(successful_attacks/len(results['predictions']), 2)
        setups_and_results.append(str(results['args']) + '\nAttack success rate: ' +
                                  str(success_rate) +
                                  '\n')

        if args.save_images:
            dataset = torch.load(results['args'].dataset)
            save_images(results, results_location, dataset)

    with open(os.path.join(args.location, 'setups_and_results.txt'), 'w') as file:
        for result in setups_and_results:
            file.write(str(result))
            file.write('\n')


if __name__ == '__main__':
    main()
