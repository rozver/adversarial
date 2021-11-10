import torch
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import json
import os
import argparse


ARGS_DICT_KEYS_PGD = ['arch', 'checkpoint_location', 'from_robustness', 'dataset', 'masks', 'eps', 'norm',
                      'step_size', 'num_iterations', 'targeted', 'eot', 'transfer', 'save_file_location']
ARGS_DICT_KEYS_BLACKBOX = ['model', 'dataset', 'attack_type', 'eps', 'num_iterations', 'save_file_location']


def has_wrong_args(results, results_location):
    if 'args' in results.keys():
        results['args_dict'] = results['args']

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


def save_images(results, results_location, dataset, save_original):
    results_images_folder = os.path.dirname(results_location) + '/images/' + results_location.split('/')[-1][:-3]
    original_directory = results_images_folder + '/original/'
    adversarial_directory = results_images_folder + '/adversarial/'

    if not os.path.exists(results_images_folder):
        os.makedirs(adversarial_directory)

        if save_original:
            os.makedirs(original_directory)

    for batch_index, (adversarial_batch, original_batch) in enumerate(zip(results['adversarial_examples'], dataset)):
        if len(adversarial_batch.size()) == 3:
            adversarial_batch = adversarial_batch.unsqueeze(0)

        for index in range(adversarial_batch.size(0)):
            save_image(adversarial_batch[index], (adversarial_directory + str(batch_index) + '_' + str(index) + '.png'))

            if save_original:
                save_image(original_batch[index], (original_directory + str(batch_index) + '_' + str(index) + '.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--model_similarity', default=False, action='store_true')
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
            targets = predictions['original']
            if len(targets.size()) == 2:
                targets = torch.argmax(targets, dim=1)

            adversarial_classes = torch.argmax(predictions['adversarial'], dim=1)

            attack_success = torch.eq(adversarial_classes, targets)

            if not results['args_dict'].get('targeted', False):
                attack_success = ~attack_success

            successful_attacks += torch.sum(attack_success).item()

        if 'num_samples' in results['args_dict'].keys():
            num_samples = results['args_dict']['num_samples']
        else:
            num_samples = len(results['predictions'])

        success_rate = round(successful_attacks / num_samples, 2)
        setups_and_results.append(str(results['args_dict']) + '\n')

        if args_dict['model_similarity']:
            if 'similarity' in results.keys():
                similarity_str = ''

                for similarity_data in results['similarity']:
                    similarity_str = similarity_str + + json.dumps(similarity_data) + ',\n'

                setups_and_results.append(similarity_str[:-2])

        setups_and_results.append('Attack success rate: ' +
                                  str(success_rate) + '\n')

        if args_dict['save_images']:
            save_original = False
            if os.path.exists(results['args_dict']['dataset']):
                if 'batch_size' not in results['args_dict']:
                    results['args_dict']['batch_size'] = 1
                    
                dataset = torch.load(results['args_dict']['dataset'])
                dataset = torch.utils.data.DataLoader(dataset,
                                                      batch_size=results['args_dict']['batch_size'],
                                                      num_workers=4)
                save_original = True
            else:
                dataset = len(results['adversarial_examples']) * [0]

            save_images(results, results_location, dataset, save_original)

    with open(os.path.join(args_dict['location'], 'setups_and_results.txt'), 'w') as file:
        for result in setups_and_results:
            file.write(str(result))
            file.write('\n')


if __name__ == '__main__':
    main()
