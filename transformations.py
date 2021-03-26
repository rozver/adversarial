import torch
from torch.nn import functional as F
import random
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import sys
import argparse


def plot(image):
    if image.size(0) == 3:
        plt.imshow(image.permute(1, 2, 0))
        plt.show()
    else:
        plt.imshow(image)
        plt.show()


def get_transformation_bounds_dict():
    bounds_dict = {
        'light': [-0.1, 0.1],
        'noise': [0.0, 0.05],
        'translation': [-10.0, 10.0],
        'rotation': [-10, 10],
    }

    return bounds_dict


def get_transformation(transformation_type):
    if transformation_type == 'light':
        return LightAdjustment()
    elif transformation_type == 'noise':
        return Noise()
    elif transformation_type == 'translation':
        return Translation()
    elif transformation_type == 'rotation':
        return Rotation()
    else:
        raise ValueError


def get_random_transformation():
    transformation_types_list = ['rotation', 'noise', 'light', 'translation']
    transformation_type = random.choice(transformation_types_list)

    t = get_transformation(transformation_type)
    return t


class Transformation:
    def __init__(self, transformation_type):
        if transformation_type not in get_transformation_bounds_dict():
            sys.exit('Invalid transformation type!')

        self.transformation_type = transformation_type
        self.lower_bound, self.upper_bound = get_transformation_bounds_dict().get(self.transformation_type)
        self.parameter = None
        self.algorithm = 1

    def __call__(self, x):
        if len(x.size()) != 4:
            x = x.unsqueeze(0)

        if self.parameter is None:
            self.parameter = [self.get_random_parameter() for i in range(x.size(0))]

        x_chunks = torch.chunk(x, x.size(0), dim=0)
        x_chunks_transformed = []

        for index in range(len(x)):
            x_chunks_transformed.append(self.transform(x_chunks[index], index))
        x = torch.cat(x_chunks_transformed, 0)
        
        self.parameter = None
        return x

    def get_random_parameter(self):
        return random.uniform(self.lower_bound, self.upper_bound)

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def transform(self, x, index):
        raise NotImplementedError


class LightAdjustment(Transformation):
    def __init__(self):
        self.transformation_type = 'light'
        super(LightAdjustment, self).__init__(transformation_type=self.transformation_type)

    def transform(self, x, index):
        light = torch.cuda.FloatTensor().new_full(x.size(), self.parameter[index])
        x = torch.add(x, light)
        return x


class Noise(Transformation):
    def __init__(self):
        self.transformation_type = 'noise'
        super(Noise, self).__init__(transformation_type=self.transformation_type)

    def transform(self, x, index):
        noise = torch.normal(mean=0.0, std=self.parameter[index], size=x.size(), device=torch.device('cuda'))
        x = torch.add(x, noise).float()
        return x


class Translation(Transformation):
    def __init__(self):
        self.transformation_type = 'translation'
        super(Translation, self).__init__(transformation_type=self.transformation_type)

    def get_random_parameter(self):
        return (random.randint(self.lower_bound, self.upper_bound),
                random.randint(self.lower_bound, self.upper_bound))

    def transform(self, x, index):
        """
        translation = [parameter for parameter in self.parameter]

        x = torch.roll(x, shifts=translation[index], dims=(1, 2))
        for j in range(3):
            if translation[0][0] < 0:
                x[index, j, translation[index][0]:x.size(2)] = 0
            else:
                x[index, j, 0: translation[index][0]] = 0

            if translation[0][1] < 0:
                for i in range(-1, translation[index][1]-1, -1):
                    x[index, j, :, i] = 0
            else:
                for i in range(translation[index][1]):
                    x[index, j, :, i] = 0
        """

        return x


class Rotation(Transformation):
    def __init__(self):
        self.transformation_type = 'rotation'
        super(Rotation, self).__init__(transformation_type=self.transformation_type)

    def transform(self, x, index):
        x_t = x.clone()

        sin = torch.sin(torch.deg2rad(torch.cuda.FloatTensor([self.parameter[index]])))
        cos = torch.cos(torch.deg2rad(torch.cuda.FloatTensor([self.parameter[index] / 2])))
        rotation_matrix = torch.tensor([[cos, sin, 0],
                                       [sin, cos, 0]]).cuda().repeat(x_t.shape[0], 1, 1)

        grid = F.affine_grid(rotation_matrix, x_t.size(), align_corners=False)

        x_t = F.grid_sample(x_t, grid, align_corners=False)
        x = x_t
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--transformation_type', type=str, default='translation')
    args = parser.parse_args()

    image = Image.open(args.image)
    transforms = torchvision.transforms.ToTensor()
    image = transforms(image)

    transformation = get_transformation(args.transformation_type)
    image = transformation(image.cuda())

    plot(image[0].cpu())


if __name__ == '__main__':
    main()
