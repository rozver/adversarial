import torch
from pgd import Attacker, MODELS_DICT
import argparse


class Trainer:
    def __init__(self, model, training_args_dict, pgd_args_dict,
                 criterion=torch.nn.CrossEntropyLoss(),
                 optimizer=torch.optim.SGD):
        self.model = model
        self.training_args_dict = training_args_dict
        self.pgd_args_dict = pgd_args_dict
        self.adversarial = training_args_dict['adversarial']
        self.attacker = None
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=1e-1)

    def fit(self, data_loader, epochs):
        self.model = self.model.cuda().train()
        for epoch in range(epochs):
            for images_batch, labels_batch in data_loader:
                if self.adversarial:
                    images_batch = self.get_adversarial_examples(images_batch, labels_batch)

                self.optimizer.zero_grad()

                predictions = self.model.cuda()(images_batch.cuda())
                loss = self.criterion(predictions, labels_batch.cuda())
                print(loss)

                loss.backward()
                self.optimizer.step()

    def switch_to_normal(self):
        self.adversarial = False

    def switch_to_adversarial(self):
        self.adversarial = True

    def get_model(self):
        return self.model

    def get_adversarial_examples(self, images_batch, labels_batch):
        mask = None
        if self.attacker is None:
            self.attacker = Attacker(self.model.cpu().eval(), self.pgd_args_dict)

        adversarial_batch = None

        for image, label in zip(images_batch, labels_batch):
            if mask is None:
                mask = torch.ones(image.size())

            if adversarial_batch is None:
                adversarial_batch = (self.attacker(image=image, mask=mask, target=label)).unsqueeze(0)
                continue

            adversarial_example = self.attacker(image=image, mask=mask, target=label)
            adversarial_batch = torch.cat((adversarial_batch, adversarial_example.unsqueeze(0)), 0)

        return adversarial_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=MODELS_DICT.keys(), default='resnet50')
    parser.add_argument('--dataset', type=str, default='dataset/imagenet-airplanes-images.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--adversarial', default=False, action='store_true')
    args_dict = vars(parser.parse_args())

    pgd_args_dict = {
        'model': 'resnet50',
        'dataset': 'dataset/imagenet-airplanes-images.pt',
        'masks': False,
        'eps': 8/255.0,
        'norm': 'linf',
        'step_size': 1/255.0,
        'num_iterations': 40,
        'targeted': False,
        'eot': False,
        'transfer': False,
    }

    images = torch.load('dataset/imagenet-airplanes-images.pt')
    labels = torch.load('dataset/imagenet-airplanes-labels.pt')

    images_loader = torch.utils.data.DataLoader(images, batch_size=10, num_workers=4)
    labels_loader = torch.utils.data.DataLoader(labels, batch_size=10, num_workers=4)
    data_loader = zip(images_loader, labels_loader)

    model = MODELS_DICT.get(args_dict['model']).cuda().train()

    trainer = Trainer(model, args_dict, pgd_args_dict)
    trainer.switch_to_adversarial()
    trainer.fit(data_loader, args_dict['epochs'])

    torch.save({'state_dict': model.state_dict(), 'training_args': args_dict, 'pgd_args': pgd_args_dict},
               'models/' + args_dict['model'] + '_robust.pt')


if __name__ == '__main__':
    main()
