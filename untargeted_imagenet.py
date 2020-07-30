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


def main():
    IMAGES_LOCATION = 'results/images/'
    CURRENT_TIME = str(datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))
    SAVE_LOCATION = IMAGES_LOCATION + CURRENT_TIME

    kwargs = {
        'constraint': 'inf',
        'eps': 0.04,
        'step_size': 0.3,
        'iterations': 15,
        'do_tqdm': True,
    }

    dataset = ImageNet('dataset/imagenet-dogs')

    model, _ = make_and_restore_model(arch='resnet50', dataset=dataset,
                                      pytorch_pretrained=True, parallel=True)
    model = model.cuda()

    image_loader = (torch.load('./dataset/imagenet-dogs-images.pt'))
    labels = (torch.load('./dataset/imagenet-dogs-labels.pt'))

    os.mkdir(SAVE_LOCATION)

    for batch_index, (images_batch, labels_batch) in enumerate(zip(image_loader, labels)):
        _, images_adversarial = model(images_batch.cuda(), labels_batch.cuda(), make_adv=True, **kwargs)
        predictions, _ = model(images_adversarial)
        labels_adversarial = torch.argmax(predictions, dim=1)

        os.mkdir(os.path.join(SAVE_LOCATION, str(batch_index)))

        for image_index in range(len(images_batch)):
            os.mkdir(os.path.join(SAVE_LOCATION, str(batch_index))+'/'+str(image_index))
            save_image(images_batch[image_index],
                       str(os.path.join(SAVE_LOCATION, str(batch_index))+'/'+str(image_index)+'/original.jpg'),
                       normalize=True)
            print(labels_batch[image_index].item())

            save_image(images_adversarial[image_index].cpu(),
                       str(os.path.join(SAVE_LOCATION, str(batch_index))+'/'+str(image_index)+'/adversarial.jpg'),
                       normalize=True)
            print(labels_adversarial[image_index].item())
    print('Finished!')


if __name__ == '__main__':
    main()
