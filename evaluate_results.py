import torch
from matplotlib import pyplot as plt
import os
from torchvision.utils import save_image
import argparse


def plot_adversarial_examples(results):
    for image in results['adversarial_examples']:
        plt.imshow(image.permute(1, 2, 0))
        plt.show()


def save_images_and_noises(results, results_location, dataset):
    results_images_folder = os.path.dirname(results_location)+'/images/'+results_location.split('/')[-1][:-3]

    if not os.path.exists(results_images_folder):
        os.makedirs(results_images_folder)

    for (image_index, adversarial_example) in enumerate(results['adversarial_examples']):
        original_image = dataset[image_index]
        noise = original_image - adversarial_example

        images_save_path = results_images_folder + '/' + str(image_index)
        save_image(original_image,
                   images_save_path + '_original.png', normalize=True)
        save_image(adversarial_example,
                   images_save_path + '_adversarial.png', normalize=True)
        save_image(noise,
                   images_save_path + '_noise.png', normalize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='dataset/imagenet-airplanes-images.pt')
    parser.add_argument('--save_images', default=False, action='store_true')
    args = parser.parse_args()

    successful_attacks = 0
    all_attacks = 0
    setups_and_results = []
    dataset = None

    files = os.listdir(args.location)

    if args.save_images:
        dataset = torch.load(args.dataset)

    for file in files:
        if not file.endswith('.pt'):
            continue

        results_location = os.path.join(args.location, file)
        results = torch.load(results_location)

        for predictions in results['predictions']:
            original_class = torch.argmax(predictions['original']).item()
            adversarial_class = torch.argmax(predictions['adversarial']).item()

            if original_class != adversarial_class:
                successful_attacks += 1

            all_attacks += 1

        setups_and_results.append(str(results['args']) + '\nAttack success rate: ' +
                                  str(successful_attacks/all_attacks*100) +
                                  '\n')

        if args.save_images:
            save_images_and_noises(results, results_location, dataset)

    with open(os.path.join(args.location, 'setups_and_results.txt'), 'w') as file:
        for result in setups_and_results:
            file.write(str(result))
            file.write('\n')


if __name__ == '__main__':
    main()
