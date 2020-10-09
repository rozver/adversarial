import torch
from pgd_attack_steps import LinfStep
from transformations import Transformation
import random
from adversarial_transfer_models import get_models_dict
import argparse
import datetime

TARGETED_CLASS = 934
MODELS_DICT = get_models_dict()


def get_current_time():
    return str(datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))


def get_random_transformation():
    transformation_types_list = ['rotation', 'noise', 'light']
    transformation_type = random.choice(transformation_types_list)

    t = Transformation(transformation_type)
    t.set_random_parameter()

    return t


def normal_loss(model, criterion, x, label, targeted=False):
    prediction = model(x.view(1, 3, 224, 224))

    optimization_direction = 1

    if targeted:
        optimization_direction = -1

    return optimization_direction*criterion(prediction, label)


def transfer_loss(model, criterion,  x, label, targeted=False):
    optimization_direction = 1

    if targeted:
        optimization_direction = -1

    losses = torch.LongTensor([]).cuda()

    for model_key in MODELS_DICT.keys():
        current_model = MODELS_DICT.get(model_key).cuda().eval()
        prediction = current_model(x.view(1, 3, 224, 224))
        current_loss = criterion(prediction, label)

        losses = torch.cat((losses, optimization_direction*current_loss))

    loss = torch.mean(losses)
    return loss


class Attacker:
    def __init__(self, images_batch, model, loss=normal_loss):
        self.images_batch = images_batch
        self.model = model
        self.loss = loss

    def get_adversarial_examples(self, target, args, random_start=False):
        adversarial_images = torch.FloatTensor(self.images_batch.size()).cuda()
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        for (index, current_image) in enumerate(self.images_batch):
            label = torch.argmax(target[index]).view(1)

            if args.targeted:
                label[0] = target[index]

            step = LinfStep(current_image, args.eps, args.step_size)
            if random_start:
                current_image = step.random_perturb(current_image)

            best_loss = None
            best_x = None
            x = current_image.clone().detach().requires_grad_(True)

            for _ in range(args.num_iterations):
                t = get_random_transformation()
                x = x.clone().detach().requires_grad_(True)

                if args.eot:
                    loss = self.loss(self.model, criterion, t(x.cuda()).cuda(), label, args.targeted)
                else:
                    loss = self.loss(self.model, criterion, x.cuda(), label, args.targeted)

                loss.backward()

                grads = x.grad.detach().clone()
                x.grad.zero_()

                if best_loss is not None:
                    if best_loss < loss:
                        best_loss = loss
                        best_x = x.clone().detach()
                else:
                    best_loss = loss.clone().detach()
                    best_x = x.clone().detach()

                x = step.step(x, grads)
                x = step.project(x)

            adversarial_images[index] = best_x

        return adversarial_images.detach()


def main():
    time = get_current_time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='dataset/imagenet-airplanes-images.pt')
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--step_size', type=float, default=1)
    parser.add_argument('--num_iterations', type=int, default=500)
    parser.add_argument('--targeted', type=bool, default=False)
    parser.add_argument('--eot', type=bool, default=False)
    parser.add_argument('--save_file_name', type=str, default='results/pgd-' + time + '.pt')
    args = parser.parse_args()

    args.eps, args.step_size = args.eps / 255.0, args.step_size / 255.0

    model = MODELS_DICT.get(args.model).cuda()

    attacker = Attacker(None, model, normal_loss)

    dataset = torch.load(args.dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=2)
    results = []

    for (batch_index, images_batch) in enumerate(data_loader):
        images_batch = images_batch.cuda()
        attacker.images_batch = images_batch

        original_predictions = model(images_batch)

        if not args.targeted:
            target = original_predictions
        else:
            target = torch.ones(images_batch.size(0)).cuda() * TARGETED_CLASS

        adversarial_examples = attacker.get_adversarial_examples(target,
                                                                 args,
                                                                 False)
        adversarial_predictions = model(adversarial_examples.cuda())

        results.append(adversarial_examples.cpu())

        for index in range(len(target)):
            print('Original prediction: ' + str(torch.argmax(original_predictions[index].cuda())))
            print('Adversarial prediction: ' + str(torch.argmax(adversarial_predictions[index].cuda())))

    torch.save({'results': results, 'args': args}, args.save_file_name)


if __name__ == '__main__':
    main()
