import torch
import torchvision
from torch import autograd


def fgsm(model, images_batch, labels_batch, epsilon):
    images = autograd.Variable(images_batch, requires_grad=True)
    criterion = torch.nn.CrossEntropyLoss()
    predictions = model(images)

    loss = criterion(predictions, labels_batch)
    grads_batch = autograd.grad(loss, images)

    adversarial_examples = []

    for grad in grads_batch:
        adversarial_examples.append(images + epsilon*grad.sign().detach())

    return torch.cat(adversarial_examples).detach()


def main():
    epsilon = 0.05

    images_loader = torch.load('dataset/imagenet-dogs-images.pt')
    labels_loader = torch.load('dataset/imagenet-dogs-labels.pt')

    model = torchvision.models.resnet50(pretrained=True).eval()

    for images_batch, labels_batch in zip(images_loader, labels_loader):
        adversarial_batch = fgsm(model, images_batch, labels_batch, epsilon)

        original_predictions = model(images_batch)
        adversarial_predictions = model(adversarial_batch)

        for original_prediction, adversarial_prediction in zip(original_predictions, adversarial_predictions):
            original_prediction = torch.argmax(original_prediction).item()
            adversarial_prediction = torch.argmax(adversarial_prediction).item()

            print('Original prediction: ' + str(original_prediction))
            print('Adversarial prediction: ' + str(adversarial_prediction))


if __name__ == '__main__':
    main()
