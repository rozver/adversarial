import torch
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
from robustness.tools.vis_tools import show_image_row
from robustness.tools.label_maps import CLASS_DICT
from torchvision.utils import save_image
import os
import datetime


def show_row(images_batch, images_adversarial, labels_batch, labels_adversarial):
    show_image_row([images_batch.cpu(), images_adversarial.cpu()],
                   tlist=[[CLASS_DICT['ImageNet'][int(t)] for t in l] for l in [labels_batch, labels_adversarial]],
                   fontsize=15,
                   filename='./adversarial_example_ImageNet.png')


def save_images(original_batch, adversarial_batch, batch_index, location):
    os.mkdir(os.path.join(location, str(batch_index)))

    for image_index in range(len(original_batch)):
        os.mkdir(os.path.join(location, str(batch_index)) + '/' + str(image_index))
        save_image(original_batch[image_index],
                   str(os.path.join(location, str(batch_index)) + '/' + str(image_index) + '/original.jpg'),
                   normalize=True)

        save_image(adversarial_batch[image_index].cpu(),
                   str(os.path.join(location, str(batch_index)) + '/' + str(image_index) + '/adversarial.jpg'),
                   normalize=True)


def main():
    IMAGES_LOCATION = 'results/images/'
    CURRENT_TIME = str(datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))
    SAVE_LOCATION = IMAGES_LOCATION + CURRENT_TIME
    os.mkdir(SAVE_LOCATION)

    print('Enter mode type: rgb or grayscale')
    mode = input()

    if mode == 'grayscale':
        imageset = (torch.load('./dataset/imagenet-dogs-images-grayscale-single.pt'))
        image_loader = torch.utils.data.DataLoader(imageset, batch_size=4, num_workers=2)
        labels = (torch.load('./dataset/imagenet-dogs-labels.pt'))
    else:
        image_loader = (torch.load('./dataset/imagenet-dogs-images.pt'))
        labels = (torch.load('./dataset/imagenet-dogs-labels.pt'))

    kwargs = {
        'constraint': 'inf',
        'eps': 8.0/255.0,
        'step_size': 1.0/255.0,
        'iterations': 40,
        'do_tqdm': True,
        #'est_grad': (1, 150)
    }

    dataset = ImageNet('dataset/imagenet-dogs')

    model, _ = make_and_restore_model(arch='resnet50', dataset=dataset,
                                      pytorch_pretrained=True)
    model = model.cuda()

    for batch_index, (images_batch, labels_batch) in enumerate(zip(image_loader, labels)):
        _, images_adversarial = model(images_batch.cuda(), labels_batch.cuda(), make_adv=True, **kwargs)
        predictions, _ = model(images_adversarial)
        save_images(images_batch, images_adversarial, batch_index, SAVE_LOCATION)

    print('Finished!')


if __name__ == '__main__':
    main()
