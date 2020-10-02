import torch
from pgd_attack_steps import LinfStep
from transformations import Transformation
import random
import argparse
from adversarial_transfer_models import get_model


def get_random_transformation():
    transformation_types_list = ['rotation', 'noise', 'light']
    transformation_type = random.choice(transformation_types_list)

    t = Transformation(transformation_type)
    t.set_random_parameter()

    return t


class Attacker:
    def __init__(self, images_batch, model):
        self.images_batch = images_batch
        self.model = model

    def get_adversarial_examples(self, target, eps, step_size, iterations, random_start=True):
        adversarial_images = torch.FloatTensor(self.images_batch.size()).cuda()
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        for (index, current_image) in enumerate(self.images_batch):
            label = torch.argmax(target[index]).view(1)
            step = LinfStep(current_image, eps, step_size)
            if random_start:
                current_image = step.random_perturb(current_image)

            best_loss = None
            best_x = None
            x = current_image.clone().detach().requires_grad_(True)

            for _ in range(iterations):
                t = get_random_transformation()
                x = x.clone().detach().requires_grad_(True)
                prediction = self.model(t(x).view(1, 3, 224, 224))

                loss = criterion(prediction, label)
                loss.backward()

                grads = x.grad.detach().clone()
                x.grad.zero_()

                if best_loss is None:
                    best_loss = loss.clone().detach()
                    best_x = x.clone().detach()
                else:
                    if best_loss < loss:
                        best_loss = loss
                        best_x = x.clone().detach()

                x = step.step(x, grads)
                x = step.project(x)

            adversarial_images[index] = best_x

        return adversarial_images.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='dataset/imagenet-airplanes-images.pt')
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--step_size', type=float, default=1)
    parser.add_argument('--num_iterations', type=int, default=500)
    parser.add_argument('--save_file_name', type=str, default='results/eot_results_8.pt')
    args = parser.parse_args()

    model = get_model(args.model).cuda()

    dataset = torch.load(args.dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=2)
    results = []

    for (batch_index, images_batch) in enumerate(data_loader):
        attacker = Attacker(images_batch.cuda(), model)
        original_predictions = model(images_batch.cuda())

        adversarial_examples = attacker.get_adversarial_examples(original_predictions,
                                                                 args.eps/255.0,
                                                                 args.step_size/255.0,
                                                                 args.num_iterations,
                                                                 False)
        adversarial_predictions = model(adversarial_examples.cuda())

        results.append(adversarial_examples.cpu())

        for index in range(len(original_predictions)):
            print('Original prediction: ' + str(torch.argmax(original_predictions[index].cpu())))
            print('Adversarial prediction: ' + str(torch.argmax(adversarial_predictions[index].cpu())))

    torch.save(results, args.save_file_name)


if __name__ == '__main__':
    main()
