import torch
from pgd import Attacker
from dataset_utils import create_data_loaders, Normalizer
from model_utils import ARCHS_LIST, get_model, load_model
from file_utils import validate_save_file_location
import argparse
import os


class Trainer:
    def __init__(self, training_args_dict, pgd_args_dict,
                 criterion=torch.nn.CrossEntropyLoss(),
                 optimizer=torch.optim.Adam):

        if training_args_dict['checkpoint_location'] is not None:
            self.model = load_model(location=training_args_dict['checkpoint_location'])
            training_args_dict['arch'] = self.model.arch
        else:
            self.model = get_model(arch=training_args_dict['arch'],
                                   parameters=('standard' if training_args_dict['pretrained'] else None))

        self.normalize = Normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.training_args_dict = training_args_dict
        self.pgd_args_dict = pgd_args_dict
        self.adversarial = training_args_dict['adversarial']
        self.attacker = None
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=training_args_dict['learning_rate'])
        self.losses = []

    def fit(self, images, labels):
        for epoch in range(self.training_args_dict['epochs']):
            current_loss = 0.0
            images_loader, labels_loader = create_data_loaders(images, labels, shuffle=True)

            for images_batch, labels_batch in zip(images_loader, labels_loader):
                if self.adversarial:
                    images_batch = self.create_adversarial_examples(images_batch, labels_batch)

                self.model = self.model.cuda().train()
                predictions = self.model(self.normalize(images_batch.cuda()))

                self.optimizer.zero_grad()
                loss = self.criterion(predictions, labels_batch.cuda())
                loss.backward()
                self.optimizer.step()

                current_loss += loss.item() * images_batch.size(0)

            epoch_loss = current_loss / len(images)
            print('Epoch: {}/{} - Loss: {}'.format(str(epoch+1),
                                                   str(self.training_args_dict['epochs']),
                                                   str(epoch_loss)))

            self.losses.append(epoch)

    def create_adversarial_examples(self, images_batch, labels_batch):
        if self.attacker is None:
            self.attacker = Attacker(self.model.cpu().eval(), self.pgd_args_dict)

        self.attacker.model = self.model.cpu().eval()

        mask = None
        adversarial_batch = None

        for image, label in zip(images_batch, labels_batch):
            if mask is None:
                mask = torch.ones(image.size())

            if adversarial_batch is None:
                adversarial_batch = self.attacker(image=image, mask=mask, target=label, random_start=True).unsqueeze(0)
                continue

            adversarial_example = self.attacker(image=image, mask=mask, target=label, random_start=True).unsqueeze(0)
            adversarial_batch = torch.cat((adversarial_batch, adversarial_example), 0)

        return adversarial_batch

    def serialize(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'training_args': self.training_args_dict,
                    'pgd_args': self.pgd_args_dict,
                    'losses': self.losses},
                   self.training_args_dict['save_file_location'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=ARCHS_LIST, default='resnet50')
    parser.add_argument('--dataset', type=str, default='dataset/imagenet-airplanes.pt')
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--checkpoint_location', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--adversarial', default=False, action='store_true')
    parser.add_argument('--save_file_location', type=str, default='models/resnet50_robust.pt')
    args_dict = vars(parser.parse_args())

    validate_save_file_location(args_dict['save_file_location'])

    if os.path.exists(args_dict['dataset']):
        dataset_properties = torch.load(args_dict['dataset'])

        pgd_args_dict = {
            'arch': args_dict['arch'],
            'dataset': dataset_properties['images'],
            'masks': False,
            'eps': 32/255.0,
            'norm': 'linf',
            'step_size': 16/255.0,
            'num_iterations': 1,
            'targeted': False,
            'eot': False,
            'transfer': False,
        }

        images = torch.load(dataset_properties['images'])

        if dataset_properties['labels'] is None:
            eval_model = get_model(arch=args_dict['arch'],  parameters='standard')
            normalize = Normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            labels = [torch.argmax(eval_model(normalize(x.unsqueeze(0)))) for x in images]
        else:
            labels = torch.load(dataset_properties['labels'])

        trainer = Trainer(args_dict, pgd_args_dict)
        trainer.fit(images, labels)
        trainer.serialize()
    else:
        raise ValueError('Specified dataset location is incorrect!')


if __name__ == '__main__':
    main()
