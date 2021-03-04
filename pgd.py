import torch
import robustness
from pgd_attack_steps import LinfStep, L2Step
from model_utils import ARCHS_LIST, get_model, load_model, predict
from transformations import get_random_transformation
from file_utils import get_current_time, validate_save_file_location
from collections import defaultdict
import argparse
import random
import copy

TARGET_CLASS = 934
SURROGATES_LIST_ALL = []


class Attacker:
    def __init__(self, model, args_dict, attack_step=LinfStep, masks_batch=None):
        self.model = model
        self.args_dict = args_dict
        self.surrogates_list = []

        if args_dict['transfer'] or args_dict['selective_transfer']:
            self.loss = self.transfer_loss
            self.available_surrogates_list = copy.copy(ARCHS_LIST)
            self.available_surrogates_list.remove(args_dict['arch'])

            if args_dict['transfer']:
                surrogates_list = random.sample(self.available_surrogates_list, args_dict['num_surrogates'])
                SURROGATES_LIST_ALL.append(surrogates_list)
                self.surrogate_models = [get_model(arch, parameters='standard', freeze=True).eval()
                                         for arch in surrogates_list]
            else:
                self.args_dict['label_shift_fails'] = 0
        else:
            self.loss = self.normal_loss

        if args_dict['norm'] == 'l2':
            attack_step = L2Step

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimization_direction = -1 if args_dict['unadversarial'] or args_dict['targeted'] else 1
        self.masks_batch = masks_batch
        self.attack_step = attack_step

    def __call__(self, images_batch, masks_batch, targets, random_start=False):
        best_loss = None
        best_x = None

        step = self.attack_step(images_batch, self.args_dict['eps'], self.args_dict['step_size'])

        if random_start:
            images_batch = step.random_perturb(images_batch, masks_batch)

        x = images_batch.clone().detach().requires_grad_(True)

        if self.args_dict['selective_transfer']:
            self.surrogate_models = self.selective_transfer(images_batch,
                                                            masks_batch,
                                                            targets,
                                                            step)
            step.eps = self.args_dict['eps']

        iterations_without_updates = 0

        for iteration in range(self.args_dict['num_iterations']):
            t = get_random_transformation()

            if iterations_without_updates == 10:
                x = step.random_perturb(images_batch, masks_batch)

            x = x.clone().detach().requires_grad_(True)

            if self.args_dict['eot']:
                loss = self.loss(t(x.cuda()), targets)
            else:
                loss = self.loss(x.cuda(), targets)

            x.register_hook(lambda grad: grad * masks_batch.float())
            loss.backward()

            grads = x.grad.detach().clone()
            x.grad.zero_()

            if best_loss is not None:
                if best_loss < loss:
                    best_loss = loss
                    best_x = x.clone().detach()
                    iterations_without_updates = -1
            else:
                best_loss = loss.clone().detach()
                best_x = x.clone().detach()
                iterations_without_updates = -1

            iterations_without_updates += 1

            x = step.step(x, grads)
            x = step.project(x)

        return best_x.cuda()

    def selective_transfer(self, images_batch, masks_batch, original_labels, step):
        model_scores = {}
        model_scores = defaultdict(lambda: 0, model_scores)
        mse_criterion = torch.nn.MSELoss(reduction='mean')
        batch_indices = torch.arange(images_batch.size(0))

        step.eps = 5*step.eps

        for iteration in range(self.args_dict['num_iterations']):
            x = images_batch.clone().detach().requires_grad_(False)
            x = step.random_perturb(x, masks_batch)

            predictions = predict(self.model, x)
            labels = torch.argmax(predictions, dim=1)

            self.args_dict['label_shift_fails'] += torch.sum(torch.eq(labels, original_labels)).item()

            for arch in self.available_surrogates_list:
                current_model = get_model(arch, 'standard', freeze=True).cuda().eval()
                current_predictions = predict(current_model, x)

                current_loss = mse_criterion(current_predictions[batch_indices, labels],
                                             predictions[batch_indices, labels])
                model_scores[arch] += current_loss

        surrogates_list = [arch
                           for arch in sorted(model_scores, key=model_scores.get)
                           [:self.args_dict['num_surrogates']]]
        SURROGATES_LIST_ALL.append(surrogates_list)

        surrogate_models = [get_model(arch, parameters='standard', freeze=True).eval()
                            for arch in surrogates_list]
        return surrogate_models

    def normal_loss(self, x, label):
        prediction = predict(self.model, x)
        loss = self.optimization_direction * self.criterion(prediction, label)
        return loss

    def transfer_loss(self, x, label):
        loss = torch.zeros([1]).cuda()

        for current_model in self.surrogate_models:
            current_model.cuda()
            prediction = predict(current_model, x)

            current_loss = self.criterion(prediction, label)
            loss = torch.add(loss, self.optimization_direction * current_loss)

        loss = loss / len(self.surrogate_models)
        return loss


def main():
    time = str(get_current_time())
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=ARCHS_LIST, default='resnet50')
    parser.add_argument('--checkpoint_location', type=str, default=None)
    parser.add_argument('--from_robustness', default=False, action='store_true')
    parser.add_argument('--dataset', type=str, default='dataset/imagenet-airplanes-images.pt')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--masks', default=False, action='store_true')
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--norm', type=str, choices=['l2', 'linf'], default='linf')
    parser.add_argument('--step_size', type=float, default=1)
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--unadversarial', default=False, action='store_true')
    parser.add_argument('--targeted', default=False, action='store_true')
    parser.add_argument('--eot', default=False, action='store_true')
    parser.add_argument('--transfer', default=False, action='store_true')
    parser.add_argument('--selective_transfer', default=False, action='store_true')
    parser.add_argument('--num_surrogates', type=int, choices=range(1, len(ARCHS_LIST)), default=5)
    parser.add_argument('--save_file_location', type=str, default='results/pgd_new_experiments/pgd-' + time + '.pt')
    args_ns = parser.parse_args()

    args_dict = vars(args_ns)

    validate_save_file_location(args_dict['save_file_location'])

    args_dict['eps'], args_dict['step_size'] = args_dict['eps'] / 255.0, args_dict['step_size'] / 255.0

    print('Running PGD experiment with the following arguments:')
    print(str(args_dict) + '\n')

    if args_dict['checkpoint_location'] is None:
        model = get_model(arch=args_dict['arch'], parameters='standard', freeze=True).cuda().eval()
    else:
        model = load_model(location=args_dict['checkpoint_location'],
                           arch=args_dict['arch'],
                           from_robustness=args_dict['from_robustness']).cuda().eval()

    attacker = Attacker(model, args_dict)

    targets = torch.zeros(1000).cuda()
    targets[TARGET_CLASS] = 1

    print('Loading dataset...')
    if args_dict['masks']:
        loader = torch.load(args_dict['dataset'])
    else:
        dataset = robustness.datasets.ImageNet(args_dict['dataset'])
        loader, _ = dataset.make_loaders(workers=10, batch_size=args_dict['batch_size'])

    print('Finished!\n')
    adversarial_examples_list = []
    predictions_list = []

    print('Starting PGD...')
    for index, batch in enumerate(loader):
        if args_dict['masks']:
            images_batch, masks_batch = batch
            labels_batch = torch.argmax(predict(model, images_batch.cuda()), dim=1)
            if masks_batch.size != images_batch.size():
                masks_batch = torch.ones_like(images_batch)
        else:
            images_batch, labels_batch = batch
            masks_batch = torch.ones_like(images_batch)

        if not args_dict['targeted']:
            targets = labels_batch

        adversarial_examples = attacker(images_batch.cuda(), masks_batch.cuda(), targets.cuda(), False)
        adversarial_predictions = predict(model, adversarial_examples)

        adversarial_examples_list.append(adversarial_examples.cpu())
        predictions_list.append({'original': labels_batch.cpu(),
                                 'adversarial': adversarial_predictions.cpu()})

        if (index+2)*images_batch.size(0) > args_dict['num_samples']:
            break

    print('Finished!')

    print('Serializing results...')
    torch.save({'adversarial_examples': adversarial_examples_list,
                'predictions': predictions_list,
                'surrogates_list': SURROGATES_LIST_ALL,
                'args_dict': args_dict},
               args_dict['save_file_location'])
    print('Finished!\n')


if __name__ == '__main__':
    main()
