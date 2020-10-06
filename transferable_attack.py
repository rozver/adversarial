import torch
import torchvision
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
from adversarial_transfer_models import get_models_dict
import datetime
from math import sqrt


def custom_transfer_loss(model, x, target):
    criterion = torch.nn.CrossEntropyLoss()
    models_dict = get_models_dict()
    loss = torch.zeros([1]).cuda()

    for key in models_dict.keys():
        current_model = models_dict[key].cuda()
        prediction = current_model(x.cuda())

        original_class = torch.argmax(target[0]).item()
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
    images_location = 'results/transferable-'
    current_time = str(datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))
    save_location = images_location + current_time

    kwargs = {
        'custom_loss': custom_transfer_loss,
        'constraint': 'inf',
        'eps': 8.0 / 255.0,
        'step_size': 1.0 / 255.0,
        'iterations': 1,
        'do_tqdm': True,
        'targeted': True,
    }

    dataset = ImageNet('dataset/imagenet-dogs')

    model, _ = make_and_restore_model(arch='resnet50', dataset=dataset,
                                      pytorch_pretrained=True)
    model = model.cuda()
    eval_model = torchvision.models.resnet50(pretrained=True).cuda().eval()

    dataset = torch.load('./dataset/imagenet-dogs-images.pt')
    image_loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)
    results = []

    for batch_index, images_batch in enumerate(image_loader):
        targ = eval_model(images_batch.cuda())

        _, images_adversarial = model(images_batch.cuda(), targ.cuda(), make_adv=True, **kwargs)
        results.append(images_adversarial.cpu())

    torch.save({'results': results, 'args': kwargs}, save_location)


if __name__ == '__main__':
    main()
