import torch
from pgd_attack_steps import LinfStep, L2Step
import random
from adversarial_transfer_models import get_models_dict
from transformations import get_transformation
import argparse
import datetime

TARGETED_CLASS = 934
MODELS_DICT = get_models_dict()


def get_current_time():
    return str(datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))


def get_random_transformation():
    transformation_types_list = ['rotation', 'noise', 'light', 'translation']
    transformation_type = random.choice(transformation_types_list)

    t = get_transformation(transformation_type)
    t.set_random_parameter()

    return t


class Attacker:
    def __init__(self, model, args, attack_step=LinfStep, masks_batch=None):
        self.model = model
        self.args = args

        if args.transfer:
            self.loss = self.transfer_loss
            self.surrogate_models = [MODELS_DICT[model_key].eval()
                                     for model_key in MODELS_DICT.keys()
                                     if model_key != args.model
                                     ]
        else:
            self.loss = self.normal_loss

        if args.norm == 'l2':
            attack_step = L2Step

        self.masks_batch = masks_batch
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.attack_step = attack_step

    def __call__(self, image, mask, target, random_start=False):
        best_loss = None
        best_x = None

        step = self.attack_step(image, self.args.eps, self.args.step_size)

        if random_start:
            image = step.random_perturb(image, mask)

        label = torch.argmax(target).view(1)
        if self.args.targeted:
            label[0] = target

        x = image.clone().detach().requires_grad_(True)

        for iteration in range(self.args.num_iterations):
            t = get_random_transformation()
            x = x.clone().detach().requires_grad_(True)

            if self.args.eot:
                loss = self.loss(t(x.cuda()).cpu(), label)
            else:
                loss = self.loss(x.cpu(), label)

            loss.backward()

            grads = x.grad.detach().clone()
            x.grad.zero_()

            grads_foreground = grads*mask

            if best_loss is not None:
                if best_loss < loss:
                    best_loss = loss
                    best_x = x.clone().detach()
            else:
                best_loss = loss.clone().detach()
                best_x = x.clone().detach()

            x = step.step(x, grads_foreground)
            x = step.project(x)

        return best_x.cpu()

    def normal_loss(self, x, label):
        prediction = self.model(x.unsqueeze(0))

        optimization_direction = 1

        if self.args.targeted:
            optimization_direction = -1

        loss = optimization_direction * self.criterion(prediction, label)
        return loss.cuda()

    def transfer_loss(self, x, label):
        optimization_direction = 1

        if self.args.targeted:
            optimization_direction = -1

        loss = torch.zeros([1])

        for current_model in self.surrogate_models:
            prediction = current_model(x.unsqueeze(0))
            current_loss = self.criterion(prediction, label)

            loss = torch.add(loss, optimization_direction * current_loss)

        loss = loss / len(self.surrogate_models)
        return loss.cuda()


def main():
    time = get_current_time()

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
    parser.add_argument('--save_file_name', type=str, default='results/pgd_new_experiments/pgd-' + time + '.pt')
    args = parser.parse_args()

    args.eps, args.step_size = args.eps / 255.0, args.step_size / 255.0

    print('Running PGD experiment with the following arguments:')
    print(str(args)+'\n')

    model = MODELS_DICT.get(args.model)

    attacker = Attacker(model, args)
    target = torch.FloatTensor([TARGETED_CLASS])

    print('Loading dataset...')
    if args.masks:
        dataset = torch.load(args.dataset)
        dataset_length = dataset.__len__()
    else:
        images = torch.load(args.dataset)
        masks = [torch.ones(images[0].size())]*images.__len__()
        dataset = zip(images, masks)
        dataset_length = images.__len__()
    print('Finished!\n')

    adversarial_examples_list = []
    predictions_list = []

    print('Starting PGD...')
    for index, (image, mask) in enumerate(dataset):
        print('Image: ' + str(index+1) + '/' + str(dataset_length))
        original_prediction = model(image.unsqueeze(0))

        if not args.targeted:
            target = original_prediction

        adversarial_example = attacker(image.cuda(), mask[0].cuda(), target, False)
        adversarial_prediction = model(adversarial_example.unsqueeze(0))

        status = 'Success' if (torch.argmax(adversarial_prediction) != torch.argmax(target)) else 'Failure'
        print('Attack status: ' + status + '\n')

        adversarial_examples_list.append(adversarial_example)
        predictions_list.append({'original': original_prediction,
                                 'adversarial': adversarial_prediction})

    print('Finished!')

    print('Serializing results...')
    torch.save({'adversarial_examples': adversarial_examples_list,
                'predictions': predictions_list,
                'args': args},
               args.save_file_name)
    print('Finished!\n')


if __name__ == '__main__':
    main()
