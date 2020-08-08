import torch
import torchvision


def nes_gradient(model, x, y, sigma, n, N):
    g = torch.zeros(N)
    mean = torch.zeros(N)
    std = torch.ones(N)
    for i in range(n):
        u = torch.normal(mean, std)
        pred = model((x+sigma*u).view(1, 3, 224, 224))
        pred = (pred[0, y])
        g = g + pred*u
        pred = model((x-sigma*u).view(1, 3, 224, 224))
        pred = pred[0, y]
        g = g - pred*u

    return g/(2*n*sigma)


def fgsm(image, grad, epsilon):
    adversarial_example = image + epsilon*grad.sign()
    return adversarial_example.detach()


images_loader = torch.load('dataset/imagenet-dogs-images.pt')
labels_loader = torch.load('dataset/imagenet-dogs-labels.pt')

model = torchvision.models.resnet50(pretrained=True).eval()

for images_batch, labels_batch in zip(images_loader, labels_loader):
    for image, label in zip(images_batch, labels_batch):
        gradient = nes_gradient(model, image, label, 0.05, 80, (3, 224, 224))
        adversarial_example = fgsm(image, gradient, 0.08)
        print(adversarial_example)

        prediction = torch.argmax(model(adversarial_example.view(1, 3, 224, 224))).item()
        print('Original prediction: ' + str(label.item()))
        print('Adversarial prediction: ' + str(prediction))
