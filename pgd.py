import torch
from pgd_attack_steps import LinfStep, L2Step
from model_utils import ARCHS_LIST, get_model, load_model, to_device, predict
from dataset_utils import load_imagenet
from transformations import get_random_transformation
from file_utils import get_current_time, validate_save_file_location
import argparse
import random
import copy

TARGET_CLASS = 934
ALL_SIMILARITY_COEFFS = []

PARSER_ARGS = [
    {'name': '--arch', 'type': str, 'choices': ARCHS_LIST + ['random'], 'default': 'resnet50', 'action': None},
    {'name': '--checkpoint_location', 'type': str, 'choices': None, 'default': None, 'action': None},
    {'name': '--from_robustness', 'default': False, 'action': 'store_true'},
    {'name': '--dataset', 'type': str, 'choices': None, 'default': 'dataset/imagenet', 'action': None},
    {'name': '--num_samples', 'type': int, 'choices': None, 'default': 50, 'action': None},
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
    {'name': '--logits_ensemble', 'default': False, 'action': 'store_true'},
    {'name': '--similarity_coeffs', 'default': False, 'action': 'store_true'},
    {'name': '--num_surrogates', 'type': int, 'choices': None, 'default': 5, 'action': None},
    {'name': '--device', 'type': str, 'choices': ['cpu', 'cuda'], 'default': 'cpu', 'action': None},
    {'name': '--seed', 'type': int, 'choices': None, 'default': None, 'action': None},
    {'name': '--save_file_location', 'type': int, 'choices': None, 'default': None, 'action': None}
]

PGD_DEFAULT_ARGS_DICT = {arg['name'][2:]: arg['default'] for arg in PARSER_ARGS}


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

    if args_dict['arch'] == 'random':
        args_dict['arch'] = random.choice(ARCHS_LIST)

    if args_dict['norm'] == 'linf':
        args_dict['eps'] = args_dict['eps'] / 255.0
    args_dict['step_size'] = args_dict['step_size'] / 255.0
    args_dict['sigma'] = args_dict['sigma'] / 255.0

    if args_dict['norm'] == 'linf':
        args_dict['restart_iterations'] = int((args_dict['eps'] / args_dict['step_size']) * 2)
    else:
        args_dict['restart_iterations'] = 10

    return args_dict


class Attacker:
    def __init__(self, model, args_dict, attack_step=LinfStep, mask_batch=None):
        self.model = model
        self.args_dict = args_dict
        self.similarity_coeffs = {}

        if args_dict['transfer']:
            self.loss = self.transfer_loss
            self.available_surrogates_list = copy.copy(ARCHS_LIST)
            self.available_surrogates_list.remove(args_dict['arch'])

            if not args_dict['selective']:
                surrogates_list = random.sample(self.available_surrogates_list, args_dict['num_surrogates'])
                coeffs = [1 / len(surrogates_list)] * len(surrogates_list)
                self.similarity_coeffs = (dict(zip(surrogates_list, coeffs)))
                ALL_SIMILARITY_COEFFS.append(self.similarity_coeffs)
                self.surrogate_models = [get_model(arch=arch, pretrained=True, freeze=True).eval()
                                         for arch in surrogates_list]
            else:
                self.args_dict['label_shifts'] = 0
        else:
            self.loss = self.normal_loss

        if args_dict['norm'] == 'l2':
            attack_step = L2Step

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimization_direction = -1 if args_dict['unadversarial'] or args_dict['targeted'] else 1
        self.mask_batch = mask_batch
        self.attack_step = attack_step

    def __call__(self, image_batch, mask_batch, targets, random_start=False):
        best_loss = None
        best_x = None

        step = self.attack_step(image_batch, self.args_dict['eps'], self.args_dict['step_size'])

        if random_start:
            image_batch = step.random_perturb(image_batch, mask_batch)

        x = image_batch.clone().detach().requires_grad_(True)

        if self.args_dict['transfer'] and self.args_dict['selective']:
            self.surrogate_models = self.selective_transfer(image_batch,
                                                            mask_batch,
                                                            targets,
                                                            step)
            step.eps = self.args_dict['eps']

        iterations_without_updates = 0

        for iteration in range(self.args_dict['num_iterations']):
            if iterations_without_updates == self.args_dict['restart_iterations']:
                x = step.random_perturb(image_batch, mask_batch)

            x = x.clone().detach().requires_grad_(True)

            if self.args_dict['eot']:
                t = get_random_transformation()
                loss = self.loss(t(x), targets)
            else:
                loss = self.loss(x, targets)

            if mask_batch != 1:
                x.register_hook(lambda grad: grad * mask_batch.float())

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

        return best_x

    def selective_transfer(self, image_batch, mask_batch, targets, step):
        model_scores = dict(zip(self.available_surrogates_list, [0] * len(self.available_surrogates_list)))
        mse_criterion = torch.nn.MSELoss(reduction='mean')
        batch_indices = torch.arange(image_batch.size(0))

        step.eps = self.args_dict['sigma']

        x = image_batch.clone().detach().requires_grad_(False)
        if self.args_dict['targeted']:
            original_labels = torch.argmax(predict(self.model, x), dim=1)
        else:
            original_labels = targets

        x = torch.cat([x.unsqueeze(0)] * self.args_dict['num_transformations'])
        x = step.random_perturb(x, mask_batch)

        predictions = []
        labels = []
        for current_x in x:
            predictions.append(predict(self.model, current_x))
            current_labels = torch.argmax(predictions[-1], dim=1)
            labels.append(current_labels)

            self.args_dict['label_shifts'] += torch.sum(~torch.eq(current_labels, original_labels)).item()

        for arch in self.available_surrogates_list:
            current_model = get_model(arch=arch, pretrained=True, freeze=True, device=self.args_dict['device']).eval()

            for index, current_x in enumerate(x):
                current_predictions = predict(current_model, current_x)
                current_loss = mse_criterion(current_predictions[batch_indices, labels[index]],
                                             predictions[index][batch_indices, labels[index]])
                model_scores[arch] += current_loss.item()

            to_device(current_model, 'cpu')

        surrogates_list = [arch
                           for arch in sorted(model_scores, key=model_scores.get)
                           [:self.args_dict['num_surrogates']]]

        if self.args_dict['similarity_coeffs']:
            scores_reversed = torch.FloatTensor([model_scores[arch] for arch in surrogates_list][::-1])
            coeffs = (scores_reversed / torch.sum(scores_reversed)).tolist()
        else:
            coeffs = [1 / len(surrogates_list)] * len(surrogates_list)

        self.similarity_coeffs = (dict(zip(surrogates_list, coeffs)))
        ALL_SIMILARITY_COEFFS.append(self.similarity_coeffs)

        surrogate_models = [get_model(arch, pretrained=True, freeze=True).eval()
                            for arch in surrogates_list]
        return surrogate_models

    def normal_loss(self, x, labels):
        predictions = predict(self.model, x)
        loss = self.optimization_direction * self.criterion(predictions, labels)
        return loss

    def transfer_loss(self, x, labels):
        if self.args_dict['logits_ensemble']:
            predictions = torch.zeros((x.size(0), 1000), device=self.args_dict['device'])

            for arch, current_model in zip(self.similarity_coeffs.keys(), self.surrogate_models):
                current_predictions = predict(to_device(current_model, self.args_dict['device']), x)

                to_device(current_model, 'cpu')

                predictions = torch.add(predictions, self.similarity_coeffs[arch] * current_predictions)

            loss = self.optimization_direction * self.criterion(predictions, labels)

        else:
            loss = torch.zeros([1], device=self.args_dict['device'])

            for arch, current_model in zip(self.similarity_coeffs.keys(), self.surrogate_models):
                current_predictions = predict(to_device(current_model, self.args_dict['device']), x)
                to_device(current_model, 'cpu')

                current_loss = self.criterion(current_predictions, labels)
                loss = torch.add(loss, self.optimization_direction * self.similarity_coeffs[arch] * current_loss)

        return loss


def main():
    args_dict = normalize_args_dict(get_args_dict())

    print('Running PGD experiment with the following arguments:')
    print(str(args_dict) + '\n')

    if args_dict['seed'] is not None:
        torch.manual_seed(args_dict['seed'])

    if args_dict['checkpoint_location'] is None:
        model = get_model(arch=args_dict['arch'], pretrained=True, freeze=True, device=args_dict['device']).eval()
    else:
        model = to_device(load_model(location=args_dict['checkpoint_location'],
                                     arch=args_dict['arch'],
                                     from_robustness=args_dict['from_robustness']).eval(),
                          'cuda')

    attacker = Attacker(model, args_dict)

    print('Loading dataset...')
    if args_dict['masks']:
        loader = torch.load(args_dict['dataset'])
    else:
        dataset = load_imagenet(args_dict['dataset'])
        loader, _ = dataset.make_loaders(workers=10, batch_size=args_dict['batch_size'])
    print('Finished!\n')

    mask_batch = 1
    total_num_samples = 0
    adversarial_examples_list = []
    predictions_list = []

    print('Starting PGD...')
    for index, batch in enumerate(loader):
        if args_dict['masks']:
            image_batch, mask_batch = batch
            image_batch.unsqueeze_(0)
            mask_batch.unsqueeze_(0)

            label_batch = torch.argmax(predict(model, to_device(image_batch, args_dict['device'])), dim=1)
            if mask_batch.size != image_batch.size():
                mask_batch = 1
            else:
                mask_batch = to_device(mask_batch, device=args_dict['device'])
        else:
            image_batch, label_batch = batch

        image_batch = to_device(image_batch, device=args_dict['device'])
        label_batch = to_device(label_batch, device=args_dict['device'])

        if not args_dict['targeted'] and not args_dict['masks']:
            predicted_label_batch = torch.argmax(predict(model, image_batch), dim=1)
            matching_labels = torch.eq(label_batch, predicted_label_batch)
            num_matching_labels = torch.sum(matching_labels)
            if num_matching_labels == 0:
                continue

            image_batch, label_batch = (image_batch[matching_labels],
                                        label_batch[matching_labels])

            if mask_batch != 1:
                mask_batch = mask_batch[matching_labels]

            targets = label_batch
        else:
            targets = TARGET_CLASS * torch.ones_like(label_batch)

        adversarial_examples = attacker(image_batch, mask_batch, targets, False)
        adversarial_predictions = predict(model, adversarial_examples)

        adversarial_examples_list.append(to_device(adversarial_examples, device='cpu'))
        predictions_list.append({'original': to_device(targets, device='cpu'),
                                 'adversarial': to_device(adversarial_predictions, device='cpu')})

        total_num_samples += image_batch.size(0)
        if total_num_samples >= args_dict['num_samples']:
            break

    args_dict['num_samples'] = total_num_samples
    print('Finished!')

    print('Serializing results...')
    torch.save({'adversarial_examples': adversarial_examples_list,
                'predictions': predictions_list,
                'similarity': ALL_SIMILARITY_COEFFS,
                'args_dict': args_dict},
               args_dict['save_file_location'])
    print('Finished!\n')


if __name__ == '__main__':
    main()
