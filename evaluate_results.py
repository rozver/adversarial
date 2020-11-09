import torch
from matplotlib import pyplot as plt
import os
from torchvision.utils import save_image


def plot_adversarial_examples(results):
    for batch in results['adversarial_examples']:
        for image in batch:
            plt.imshow(image.permute(1, 2, 0))
            plt.show()


def save_images_and_noises(results, results_location, dataset):
    results_images_folder = os.path.dirname(results_location)+'/images/'+results_location.split('/')[-1][:-3]

    if not os.path.exists(results_images_folder):
        os.makedirs(results_images_folder)

    batch_size = len(results['adversarial_examples'][0])

    for (batch_index, adversarial_batch) in enumerate(results['adversarial_examples']):
        for (image_index, adversarial_example) in enumerate(adversarial_batch):
            original_image = dataset[batch_index*batch_size+image_index]
            noise = original_image - adversarial_example

            images_save_path = results_images_folder + '/' + str(batch_index) + '_' + str(image_index)
            save_image(original_image,
                       images_save_path + '_original.png', normalize=True)
            save_image(adversarial_example,
                       images_save_path + '_adversarial.png', normalize=True)
            save_image(noise,
                       images_save_path + '_noise.png', normalize=True)


def main():
    dataset_location = 'dataset/imagenet-airplanes-images.pt'
    dataset = torch.load(dataset_location)

    files_folder_location = 'results/supercloud/pgd_new_experiments'
    files = os.listdir(files_folder_location)

    successful_attacks = 0
    all_attacks = 0
    setups_and_results = []

    for file in files:
        if not file.endswith('.pt'):
            continue

        results_location = os.path.join(files_folder_location, file)
        results = torch.load(results_location)

        save_images_and_noises(results, results_location, dataset)

        for predictions in results['predictions']:
            for original, adversarial in zip(predictions['original'], predictions['adversarial']):
                original_class = torch.argmax(original).item()
                adversarial_class = torch.argmax(adversarial).item()

                if original_class != adversarial_class:
                    successful_attacks += 1

                all_attacks += 1

        setups_and_results.append(str(results['args']) + '\nAttack success rate: ' +
                                  str(successful_attacks/all_attacks*100) +
                                  '\n')

        save_images_and_noises(results, results_location, dataset)

    with open(os.path.join(files_folder_location, 'setups_and_results.txt'), 'w') as file:
        for result in setups_and_results:
            file.write(str(result))
            file.write('\n')


if __name__ == '__main__':
    main()
