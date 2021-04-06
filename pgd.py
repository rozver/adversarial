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
SURROGATES_COEFFS = {}


PARSER_ARGS = [
                {'name': '--arch', 'type': str, 'choices': ARCHS_LIST, 'default': 'resnet50', 'action': None},
                {'name': '--checkpoint_location', 'type': str, 'choices': None, 'default': None, 'action': None},
                {'name': '--from_robustness', 'default': False, 'action': 'store_true'},
                {'name': '--dataset', 'type': str, 'choices': None, 'default': 'dataset/imagenet', 'action': None},
                {'name': '--num_samples', 'type': int, 'choices': None, 'default': 500, 'action': None},
                {'name': '--sigma', 'type': int, 'choices': None, 'default': 8, 'action': None},
                {'name': '--num_transformations', 'type': int, 'choices': None, 'default': 50, 'action': None},
                {'name': '--batch_size', 'type': int, 'choices': None, 'default': 2, 'action': None},
                {'name': '--masks', 'default': False, 'action': 'store_true'},
                {'name': '--eps', 'type': float, 'choices': None, 'default': 8, 'action': None},
                {'name': '--norm', 'type': str, 'choices': ['l2', 'linf'], 'default': 'linf', 'action': None},
                {'name': '--step_size', 'type': float, 'choices': None, 'default': 1, 'action': None},
                {'name': '--num_iterations', 'type': int, 'choices': None, 'default': 10, 'action': None},
                {'name': '--unadversarial', 'default': False, 'action': 'store_true'},
                {'name': '--targeted', 'default': False, 'action': 'store_true'},
                {'name': '--eot', 'default': False, 'action': 'store_true'},
                {'name': '--transfer', 'default': False, 'action': 'store_true'},
                {'name': '--selective', 'default': False, 'action': 'store_true'},
                {'name': '--surrogates_coeffs', 'default': False, 'action': 'store_true'},
                {'name': '--num_surrogates', 'type': int, 'choices': None, 'default': 5, 'action': None},
                {'name': '--save_file_location', 'type': int, 'choices': None, 'default': None, 'action': None},
            ]


def get_args_dict():
    parser = argparse.ArgumentParser()
    for arg_dict in PARSER_ARGS:
        if arg_dict['action'] is None:
            parser.add_argument(arg_dict['name'],
                                type=arg_dict['type'],
                                choices=arg_dict['choices'],
                                default=arg_dict['default'],
                                action=arg_dict['action'])
        else:
            parser.add_argument(arg_dict['name'],
                                default=arg_dict['default'],
                                action=arg_dict['action'])

    args_ns = parser.parse_args()
    args_dict = vars(args_ns)
    return args_dict


def normalize_args_dict(args_dict):
    time = str(get_current_time())
    if args_dict['save_file_location'] is None:
        args_dict['save_file_location'] = 'results/pgd_new_experiments/' + time + '.pt'
    validate_save_file_location(args_dict['save_file_location'])

    if args_dict['norm'] == 'linf':
        args_dict['eps'] = args_dict['eps'] / 255.0
    args_dict['step_size'] = args_dict['step_size'] / 255.0
    args_dict['sigma'] = args_dict['sigma'] / 255.0

    if args_dict['norm'] == 'linf':
        args_dict['restart_iterations'] = int((args_dict['eps'] / args_dict['step_size'])*1.25)
    else:
        args_dict['restart_iterations'] = 10

    return args_dict


class Attacker:
    def __init__(self, model, args_dict, attack_step=LinfStep, masks_batch=None):
        self.model = model
        self.args_dict = args_dict
        self.surrogates_list = []

        if args_dict['transfer']:
            self.loss = self.transfer_loss
            self.available_surrogates_list = copy.copy(ARCHS_LIST)
            self.available_surrogates_list.remove(args_dict['arch'])

            if not args_dict['selective']:
                surrogates_list = random.sample(self.available_surrogates_list, args_dict['num_surrogates'])
                coeffs = [1/len(surrogates_list)]*len(surrogates_list)
                SURROGATES_COEFFS.update(dict(zip(surrogates_list, coeffs)))
                self.surrogate_models = [get_model(arch, parameters='standard', freeze=True).eval()
                                         for arch in surrogates_list]
            else:
                self.args_dict['label_shifts'] = 0
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

        if self.args_dict['transfer'] and self.args_dict['selective']:
            self.surrogate_models = self.selective_transfer(images_batch,
                                                            masks_batch,
                                                            targets,
                                                            step)
            step.eps = self.args_dict['eps']

        iterations_without_updates = 0

        for iteration in range(self.args_dict['num_iterations']):
            if iterations_without_updates == self.args_dict['restart_iterations']:
                x = step.random_perturb(images_batch, masks_batch)

            x = x.clone().detach().requires_grad_(True)

            if self.args_dict['eot']:
                t = get_random_transformation()
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

        step.eps = self.args_dict['sigma']

        for iteration in range(self.args_dict['num_transformations']):
            x = images_batch.clone().detach().requires_grad_(False)
            x = step.random_perturb(x, masks_batch)

            predictions = predict(self.model.cuda(), x.cuda())
            labels = torch.argmax(predictions, dim=1)

            self.args_dict['label_shifts'] += (len(labels) - torch.sum(torch.eq(labels, original_labels)).item())

            for arch in self.available_surrogates_list:
                current_model = get_model(arch, 'standard', freeze=True).cuda().eval()
                current_predictions = predict(current_model, x)

                current_loss = mse_criterion(current_predictions[batch_indices, labels],
                                             predictions[batch_indices, labels])
                model_scores[arch] += current_loss

        surrogates_list = [arch
                           for arch in sorted(model_scores, key=model_scores.get)
                           [:self.args_dict['num_surrogates']]]

        if self.args_dict['surrogate_coeffs']:
            scores = torch.FloatTensor([model_scores[arch] for arch in model_scores.keys()])
            coeffs = torch.nn.functional.softmax(scores, dim=0).tolist()
        else:
            coeffs = [1/len(surrogates_list)]*len(surrogates_list)

        SURROGATES_COEFFS.update(dict(zip(surrogates_list, coeffs)))

        surrogate_models = [get_model(arch, parameters='standard', freeze=True).eval()
                            for arch in surrogates_list]
        return surrogate_models

    def normal_loss(self, x, labels):
        predictions = predict(self.model, x)
        loss = self.optimization_direction * self.criterion(predictions, labels)
        return loss

    def transfer_loss(self, x, labels):
        loss = torch.zeros([1]).cuda()

        for current_model in self.surrogate_models:
            current_model.cuda()
            predictions = predict(current_model, x)

            current_loss = self.criterion(predictions, labels)
            loss = torch.add(loss, self.optimization_direction * current_loss)

        loss = loss / len(self.surrogate_models)
        return loss


def main():
    args_dict = normalize_args_dict(get_args_dict())

    print('Running PGD experiment with the following arguments:')
    print(str(args_dict) + '\n')

    if args_dict['checkpoint_location'] is None:
        model = get_model(arch=args_dict['arch'], parameters='standard', freeze=True).cuda().eval()
    else:
        model = load_model(location=args_dict['checkpoint_location'],
                           arch=args_dict['arch'],
                           from_robustness=args_dict['from_robustness']).cuda().eval()

    attacker = Attacker(model, args_dict)

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
        else:
            targets = TARGET_CLASS*torch.ones_like(labels_batch)

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
                'surrogates_data': SURROGATES_COEFFS,
                'args_dict': args_dict},
               args_dict['save_file_location'])
    print('Finished!\n')


if __name__ == '__main__':
    main()
