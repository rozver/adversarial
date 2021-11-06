import torch
from pgd import Attacker, PGD_DEFAULT_ARGS_DICT
from dataset_utils import create_data_loaders, Normalizer
from model_utils import ARCHS_LIST, predict, get_model, load_model
from file_utils import validate_save_file_location, get_current_time
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
                                   pretrained=(True if training_args_dict['pretrained'] else None))

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

            for image_batch, label_batch in zip(images_loader, labels_loader):
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

                if self.adversarial:
                    image_batch = self.create_adversarial_examples(image_batch, label_batch)

                self.model = self.model.cuda().train()
                predictions = predict(self.model, self.normalize(image_batch.cuda()))

                self.optimizer.zero_grad()
                loss = self.criterion(predictions, label_batch)
                loss.backward()
                self.optimizer.step()

                if self.training_args_dict['weight_averaging']:
                    with torch.no_grad():
                        old_parameters = self.model.parameters()
                        for (name, parameter), old_parameter in zip(self.model.named_parameters(), old_parameters):
                            if 'weight' in name:
                                parameter.copy_((parameter + old_parameter) / 2)

                        predictions = predict(self.model, self.normalize(image_batch))
                        loss = self.criterion(predictions, label_batch)

                current_loss += loss.item() * image_batch.size(0)

            epoch_loss = current_loss / len(images)
            print('Epoch: {}/{} - Loss: {}'.format(str(epoch + 1),
                                                   str(self.training_args_dict['epochs']),
                                                   str(epoch_loss)))

            self.losses.append(epoch)

    def create_adversarial_examples(self, image_batch, label_batch):
        if self.attacker is None:
            self.attacker = Attacker(self.model.eval(), self.pgd_args_dict)

        self.attacker.model = self.model.cuda().eval()

        mask_batch = None

        if mask_batch is None:
            mask_batch = torch.ones_like(image_batch)

        adversarial_examples = self.attacker(image_batch,
                                             mask_batch=mask_batch,
                                             targets=label_batch,
                                             random_start=True)

        return adversarial_examples

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
    parser.add_argument('--weight_averaging', default=False, action='store_true')
    parser.add_argument('--adversarial', default=False, action='store_true')
    parser.add_argument('--save_file_location', type=str, default='models/' + str(get_current_time()) + '.pt')
    args_dict = vars(parser.parse_args())

    validate_save_file_location(args_dict['save_file_location'])

    if os.path.exists(args_dict['dataset']):
        dataset_properties = torch.load(args_dict['dataset'])

        pgd_args_dict = PGD_DEFAULT_ARGS_DICT
        pgd_args_dict['arch'] = args_dict['arch']
        pgd_args_dict['dataset'] = dataset_properties['images']
        pgd_args_dict['eps'] = 32 / 255.0
        pgd_args_dict['step_size'] = 32 / 255.0

        images = torch.load(dataset_properties['images'])

        if dataset_properties['labels'] is None:
            eval_model = get_model(arch=args_dict['arch'], pretrained=True)
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
