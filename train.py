import torch
from pgd import Attacker, MODELS_DICT
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=MODELS_DICT.keys(), default='resnet50')
    parser.add_argument('--dataset', type=str, default='dataset/imagenet-airplanes-images.pt')
    parser.add_argument('--masks', default=False, action='store_true')
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--norm', type=str, choices=['l2', 'linf'], default='linf')
    parser.add_argument('--step_size', type=float, default=1)
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--targeted', default=False, action='store_true')
    parser.add_argument('--eot', default=False, action='store_true')
    parser.add_argument('--transfer', default=False, action='store_true')
    args = parser.parse_args()

    images = torch.load('dataset/imagenet-airplanes-images.pt')
    labels = torch.load('dataset/imagenet-airplanes-labels.pt')

    images_loader = torch.utils.data.DataLoader(images, batch_size=10, num_workers=4)
    labels_loader = torch.utils.data.DataLoader(labels, batch_size=10, num_workers=4)

    model = MODELS_DICT.get(args.model).train()
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=1e-1)

    attacker = Attacker(model, args)

    for images_batch, labels_batch in zip(images_loader, labels_loader):
        for image, label in zip(images_batch, labels_batch):
            delta = attacker(image, torch.ones(image.size()), label) - image
            adversarial_prediction = model.cuda()((image+delta).cuda().unsqueeze(0))

            loss = criterion(adversarial_prediction, torch.LongTensor([label]).cuda())
            print(loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save(model.state_dict(), 'models/' + args.model + '_robust.pt')


if __name__ == '__main__':
    main()
