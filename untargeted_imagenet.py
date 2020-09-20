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
                   filename='./adversarial_example_imagenet.png')


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
    images_location = 'results/images/'
    current_time = str(datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))
    save_location = images_location + current_time
    os.mkdir(save_location)

    print('Enter mode type: rgb or grayscale')
    mode = input()

    if mode == 'grayscale':
        imageset = (torch.load('./dataset/imagenet-airplanes-images-grayscale.pt'))
        image_loader = torch.utils.data.DataLoader(imageset, batch_size=4, num_workers=2)
        labels = (torch.load('./dataset/imagenet-airplanes-labels.pt'))
    else:
        imageset = (torch.load('./dataset/imagenet-airplanes-images.pt'))
        image_loader = torch.utils.data.DataLoader(imageset, batch_size=4, num_workers=2)
        labels = (torch.load('./dataset/imagenet-airplanes-labels.pt'))

    kwargs = {
        'constraint': 'inf',
        'eps': 64/255.0,
        'step_size': 1/255.0,
        'iterations': 500,
        'do_tqdm': True,
    }

    dataset = ImageNet('dataset/imagenet-airplanes')

    model, _ = make_and_restore_model(arch='resnet50', dataset=dataset,
                                      pytorch_pretrained=True)
    model = model.cuda()

    for batch_index, (images_batch, labels_batch) in enumerate(zip(image_loader, labels)):
        images_batch = images_batch[:2]
        labels_batch = labels_batch[:2]
                
        label = torch.LongTensor(2)
        label[0] = 101
        label[1] = 101

        print(label)

        print(images_batch.shape)
        print(labels_batch.shape)

        _, images_adversarial = model(images_batch.cuda(), labels_batch.cuda(), make_adv=True, **kwargs)
        predictions, _ = model(images_adversarial)
        save_images(images_batch, images_adversarial, batch_index, save_location)

    print('Finished!')


if __name__ == '__main__':
    main()
