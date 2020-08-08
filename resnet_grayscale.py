import torch
import torchvision
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet


def main():
    dataset = ImageNet('./dataset/imagenet-dogs')

    resnet50 = torchvision.models.resnet50(pretrained=True).cuda().eval()
    resnet50.conv1.weight.data = resnet50.conv1.weight.data.sum(axis=1).reshape(64, 1, 7, 7)
    model, _ = make_and_restore_model(arch='resnet50', dataset=dataset, pytorch_pretrained=True)

    model = model.cuda().eval()

    images = torch.load('dataset/imagenet-dogs-images-grayscale-single.pt')
    labels_loader = torch.load('dataset/imagenet-dogs-labels.pt')
    images_loader = torch.utils.data.DataLoader(images, batch_size=4, num_workers=2)

    for images_batch, labels_batch in zip(images_loader, labels_loader):
        for image, label in zip(images_batch, labels_batch):
            original_pred = resnet50(image.view(1, 1, 224, 224).cuda())
            pred = model(image.view(1, 1, 224, 224).cuda())
            print('Ground truth: ' + str(label.item()))
            print('Robustness prediction: 3' + str(torch.argmax(pred[0]).item()))
            print('ResNet prediction: ' + str(torch.argmax(original_pred[0]).item()))

            print('')


if __name__ == '__main__':
    main()
