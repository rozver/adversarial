import torch
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
                surrogates_list = random.choices(self.available_surrogates_list, k=args_dict['num_surrogates'])
                SURROGATES_LIST_ALL.append(surrogates_list)
                self.surrogate_models = [get_model(arch, parameters='standard').eval()
                                         for arch in surrogates_list]
        else:
            self.loss = self.normal_loss

        if args_dict['norm'] == 'l2':
            attack_step = L2Step

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.optimization_direction = -1 if args_dict['unadversarial'] or args_dict['targeted'] else 1
        self.masks_batch = masks_batch
        self.attack_step = attack_step

    def __call__(self, image, mask, target, random_start=False):
        best_loss = None
        best_x = None

        step = self.attack_step(image, self.args_dict['eps'], self.args_dict['step_size'])

        if random_start:
            image = step.random_perturb(image, mask)

        label = torch.argmax(target).view(1)

        x = image.clone().detach().requires_grad_(True)

        if self.args_dict['selective_transfer']:
            self.surrogate_models = self.selective_transfer(image.cpu(),
                                                            mask.cpu(),
                                                            label,
                                                            step,
                                                            self.args_dict['num_iterations']//10+1)

        self.model = self.model.cpu()
        iterations_without_updates = 0

        for iteration in range(self.args_dict['num_iterations']):
            t = get_random_transformation()

            if iterations_without_updates == 10:
                x = step.random_perturb(image, mask)

            x = x.clone().detach().requires_grad_(True)

            if self.args_dict['eot']:
                loss = self.loss(t(x.cuda()).cpu(), label)
            else:
                loss = self.loss(x.cpu(), label)

            x.register_hook(lambda grad: grad * mask.float())
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

        return best_x.cpu()

    def selective_transfer(self, image, mask, label, step, num_queries):
        model_scores = {}
        model_scores = defaultdict(lambda: 0, model_scores)

        for iteration in range(num_queries):
            x = image.clone().detach().requires_grad_(True)
            x = step.random_perturb(x, mask)
            for arch in self.available_surrogates_list:
                current_model = get_model(arch, 'standard').eval()
                prediction = predict(current_model, x)
                current_loss = self.criterion(prediction, label).item()
                model_scores[arch] += current_loss

        surrogates_list = [arch
                           for arch in sorted(model_scores, key=model_scores.get)
                           [:self.args_dict['num_surrogates']]]
        SURROGATES_LIST_ALL.append(surrogates_list)

        surrogate_models = [get_model(arch, parameters='standard').eval()
                            for arch in surrogates_list]
        return surrogate_models

    def normal_loss(self, x, label):
        prediction = predict(self.model, x)
        loss = self.optimization_direction * self.criterion(prediction, label)
        return loss.cuda()

    def transfer_loss(self, x, label):
        loss = torch.zeros([1])

        for current_model in self.surrogate_models:
            prediction = predict(current_model, x)
            current_loss = self.criterion(prediction, label)
            loss = torch.add(loss, self.optimization_direction * current_loss)

        loss = loss / len(self.surrogate_models)
        return loss.cuda()


def main():
    time = str(get_current_time())

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=ARCHS_LIST, default='resnet50')
    parser.add_argument('--checkpoint_location', type=str, default=None)
    parser.add_argument('--from_robustness', default=False, action='store_true')
    parser.add_argument('--dataset', type=str, default='dataset/imagenet-airplanes-images.pt')
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
    parser.add_argument('--num_surrogates', type=int, choices=range(0, len(ARCHS_LIST)-1), default=5)
    parser.add_argument('--save_file_location', type=str, default='results/pgd_new_experiments/pgd-' + time + '.pt')
    args_ns = parser.parse_args()

    args_dict = vars(args_ns)

    validate_save_file_location(args_dict['save_file_location'])

    args_dict['eps'], args_dict['step_size'] = args_dict['eps'] / 255.0, args_dict['step_size'] / 255.0

    print('Running PGD experiment with the following arguments:')
    print(str(args_dict) + '\n')

    if args_dict['checkpoint_location'] is None:
        model = get_model(arch=args_dict['arch'], parameters='standard').eval()
    else:
        model = load_model(location=args_dict['checkpoint_location'],
                           arch=args_dict['arch'],
                           from_robustness=args_dict['from_robustness']).eval()

    attacker = Attacker(model, args_dict)

    target = torch.zeros(1000)
    target[TARGET_CLASS] = 1

    print('Loading dataset...')
    if args_dict['masks']:
        dataset = torch.load(args_dict['dataset'])
        dataset_length = dataset.__len__()
    else:
        images = torch.load(args_dict['dataset'])
        masks = [torch.ones_like(images[0])] * images.__len__()
        dataset = zip(images, masks)
        dataset_length = images.__len__()
    print('Finished!\n')

    adversarial_examples_list = []
    predictions_list = []

    print('Starting PGD...')
    for index, (image, mask) in enumerate(dataset):
        print('Image: ' + str(index + 1) + '/' + str(dataset_length))
        original_prediction = predict(model, image)

        if not args_dict['targeted']:
            target = original_prediction

        if mask.size != image.size():
            mask = torch.ones_like(image)
        
        adversarial_example = attacker(image.cuda(), mask[0].cuda(), target, False)
        adversarial_prediction = predict(model, adversarial_example)

        if args_dict['unadversarial'] or args_dict['targeted']:
            expression = torch.argmax(adversarial_prediction) == torch.argmax(target)
        else:
            expression = torch.argmax(adversarial_prediction) != torch.argmax(target)

        status = 'Success' if expression else 'Failure'
        print('Attack status: ' + status + '\n')

        adversarial_examples_list.append(adversarial_example)
        predictions_list.append({'original': original_prediction,
                                 'adversarial': adversarial_prediction})
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
