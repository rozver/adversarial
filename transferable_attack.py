import torch
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
from adversarial_transfer_models import get_models_dict
from untargeted_imagenet import save_images
import datetime
import os
from math import sqrt


def custom_transfer_loss(model, x, target):
    criterion = torch.nn.CrossEntropyLoss()
    models_dict = get_models_dict()
    loss = torch.zeros([1]).cuda()

    for key in models_dict.keys():
        current_model = models_dict[key].cuda()
        prediction = current_model(x.cuda())

        original_class = target.item()
        adv_class = torch.argmax(prediction[0]).item()

        print(original_class)
        print(adv_class)

        weight = sqrt((adv_class-original_class)**2)+0.01

        loss = loss - weight*criterion(prediction.cuda(), target.cuda())

    loss = loss/(len(models_dict.keys()))

    return loss, None


def custom_transfer_loss_targeted(model, x, target):
    criterion = torch.nn.CrossEntropyLoss()
    models_dict = get_models_dict()
    loss = torch.zeros([1]).cuda()

    for key in models_dict.keys():
        current_model = models_dict[key].cuda()
        prediction = current_model(x.cuda())

        print(torch.argmax(prediction[0]).item())

        loss = loss + criterion(prediction.cuda(), target.cuda())

    loss = loss/(len(models_dict.keys()))

    return loss, None


def main():
    images_location = 'results/images/'
    current_time = str(datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))
    save_location = images_location + current_time
    os.mkdir(save_location)

    kwargs = {
        'custom_loss': custom_transfer_loss,
        'constraint': 'inf',
        'eps': 16.0 / 255.0,
        'step_size': 1.0 / 255.0,
        'iterations': 50,
        'do_tqdm': True,
        'targeted': True,
    }

    dataset = ImageNet('dataset/imagenet-dogs')

    model, _ = make_and_restore_model(arch='resnet50', dataset=dataset,
                                      pytorch_pretrained=True)
    model = model.cuda()

    dataset = torch.load('./dataset/imagenet-dogs-images-grayscale.pt')
    image_loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)
    labels = torch.load('./dataset/imagenet-dogs-labels.pt')

    for batch_index, (images_batch, labels_batch) in enumerate(zip(image_loader, labels)):
        images_batch = images_batch[:1]
        labels_batch = labels_batch[:1]

        targ = torch.LongTensor([labels_batch])

        _, images_adversarial = model(images_batch.cuda(), targ.cuda(), make_adv=True, **kwargs)

        eval_model = model.eval()

        original_pred = eval_model(images_batch.cuda())
        original_pred = torch.argmax(original_pred[0]).item()
        adversarial_pred = eval_model(images_adversarial.cuda())
        adversarial_pred = torch.argmax(adversarial_pred[0]).item()

        print('Original prediction: ' + str(original_pred))
        print('Adversarial prediction: ' + str(adversarial_pred))
        print('')

        save_images(images_batch, images_adversarial, batch_index, save_location)

    print('Finished!')


if __name__ == '__main__':
    main()
