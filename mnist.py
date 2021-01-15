import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from pgd import Attacker


def get_examples(model, images_batch, labels_batch, args_dict, masks_batch=None):
    attacker = Attacker(model, args_dict)
    attacker.model = attacker.model.cpu().eval()

    adversarial_batch = None

    if masks_batch is None:
        masks_batch = torch.ones_like(images_batch).cuda()

    for (image, mask), label in zip((images_batch, masks_batch), labels_batch):
        if adversarial_batch is None:
            adversarial_batch = attacker(image=image.cuda(), mask=mask, target=label.cpu(), random_start=True).unsqueeze(0)
            continue

        adversarial_example = attacker(image=image.cuda(), mask=mask, target=label.cpu(), random_start=True).unsqueeze(0)
        adversarial_batch = torch.cat((adversarial_batch, adversarial_example), 0)

    return adversarial_batch


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

args_dict = {
            'arch': 'resnet50',
            'dataset': None,
            'masks': False,
            'eps': 8/255.0,
            'norm': 'linf',
            'step_size': 1/255.0,
            'num_iterations': 50,
            'targeted': False,
            'eot': False,
            'transfer': False,
        }

transform = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.MNIST(root='dataset/mnist', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, num_workers=4)

test_set = torchvision.datasets.MNIST(root='dataset/mnist', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, num_workers=4)

model = MnistNet().cuda().train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                            momentum=0.5)

zeros = torch.zeros(1, 28, 28).cuda()
ones = torch.ones(1, 28, 28).cuda()
for epoch in range(10):
    for images_batch, labels_batch in train_loader:
        images_batch, labels_batch = images_batch.cuda(), labels_batch.cuda()
        masks_batch = torch.where(images_batch == 0, zeros, ones)

        for mask in masks_batch:
            plt.imshow(mask.permute(1, 2, 0))
            plt.show()

        break
        adversarial_examples = get_examples(model, images_batch, labels_batch, args_dict, masks_batch)
        optimizer.zero_grad()

        predictions = model.train().cuda()(adversarial_examples.cuda())
        loss = criterion(predictions, labels_batch)

        loss.backward()
        optimizer.step()

correct = 0
total = 0
model = model.cpu().eval()

for images, labels in train_loader:
    output = model(images)
    _, predicted = torch.max(output.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

torch.save(model.state_dict(), 'models/mnist_cnn_foreground_robust.pt')
