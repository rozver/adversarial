import torch
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
from adversarial_transfer_models import get_models_dict
from untargeted_imagenet import save_images
import datetime
import os


def custom_transfer_loss(model, x, target):
    criterion = torch.nn.CrossEntropyLoss()
    models_dict = get_models_dict()
    loss = torch.zeros([1]).cuda()

    for key in models_dict.keys():
        current_model = models_dict[key].cuda()
        prediction = current_model(x.cuda())

        current_loss = -criterion(prediction.cuda(), target.cuda())
        loss = loss + current_loss

    loss = loss/(len(models_dict.keys()))

    return loss, None


def main():
    IMAGES_LOCATION = 'results/images/'
    CURRENT_TIME = str(datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))
    SAVE_LOCATION = IMAGES_LOCATION + CURRENT_TIME
    os.mkdir(SAVE_LOCATION)

    kwargs = {
        'custom_loss': custom_transfer_loss,
        'constraint': 'inf',
        'eps': 16.0 / 255.0,
        'step_size': 1.0 / 255.0,
        'iterations': 40,
        'do_tqdm': True,
        'targeted': True,
    }

    dataset = ImageNet('dataset/imagenet-dogs')

    model, _ = make_and_restore_model(arch='resnet50', dataset=dataset,
                                      pytorch_pretrained=True)
    model = model.cuda()

    dataset = torch.load('./dataset/imagenet-dogs-images.pt')
    image_loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)
    labels = torch.load('./dataset/imagenet-dogs-labels.pt')

    for batch_index, (images_batch, labels_batch) in enumerate(zip(image_loader, labels)):
        images_batch = images_batch[:1]
        labels_batch = labels_batch[:1]

        targ = torch.LongTensor([labels_batch])

        _, images_adversarial = model(images_batch.cuda(), targ.cuda(), make_adv=True, **kwargs)

        test_model = model.eval()

        original_pred = test_model(images_batch.cuda())
        original_pred = torch.argmax(original_pred[0]).item()
        adversarial_pred = test_model(images_adversarial.cuda())
        adversarial_pred = torch.argmax(adversarial_pred[0]).item()

        print('Original prediction: ' + str(original_pred))
        print('Adversarial prediction: ' + str(adversarial_pred))
        print('')

        save_images(images_batch, images_adversarial, batch_index, SAVE_LOCATION)

    print('Finished!')


if __name__ == '__main__':
    main()
