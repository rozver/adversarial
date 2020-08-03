import torch
import torchvision
from torch import autograd
from predict import predict, predict_multiple


def fsgm(model, images_batch, labels_batch, epsilon):
    images = autograd.Variable(images_batch, requires_grad=True)
    criterion = torch.nn.CrossEntropyLoss()
    predictions = model(images)

    loss = criterion(predictions, labels_batch)
    grads_batch = autograd.grad(loss, images)

    adversarial_examples = []

    for grad in grads_batch:
        adversarial_examples.append(images + epsilon*torch.sign(grad))

    return torch.cat(adversarial_examples).detach()


def main():
    epsilon = 0.05

    images_loader = torch.load('dataset/imagenet-dogs-images.pt')
    labels_loader = torch.load('dataset/imagenet-dogs-labels.pt')

    model = torchvision.models.resnet50(pretrained=True).eval()

    for images_batch, labels_batch in zip(images_loader, labels_loader):
        adversarial_batch = fsgm(model, images_batch, labels_batch, epsilon)

        original_predictions = predict_multiple(images_batch, model, is_tensor=True)
        adversarial_predictions = predict_multiple(adversarial_batch, model, is_tensor=True)

        for original_prediction, adversarial_prediction in zip(original_predictions, adversarial_predictions):
            original_prediction = torch.argmax(original_prediction).item()
            adversarial_prediction = torch.argmax(adversarial_prediction).item()

            print('Original prediction: ' + str(original_prediction))
            print('Adversarial prediction: ' + str(adversarial_prediction))


if __name__ == '__main__':
    main()
