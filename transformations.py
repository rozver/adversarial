import torch
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
        'rotation': [-35, 35],
        'light': [-0.1, 0.1],
        'noise': [0.0, 0.1]
    }

    return bounds_dict


class LightAdjustment:
    def __init__(self, parameter, algorithm):
        self.parameter = parameter
        self.algorithm = algorithm

    def __call__(self, x):
        light = torch.FloatTensor().new_full(x.size(), self.parameter).cuda()
        x = torch.add(x, light)
        return x


class Noise:
    def __init__(self, parameter, algorithm):
        self.parameter = parameter
        self.algorithm = algorithm

    def __call__(self, x):
        noise = torch.normal(mean=0.0, std=self.parameter, size=x.size()).cuda()
        x = torch.add(x, noise).float()
        return x


class Rotation:
    def __init__(self, parameter, algorithm):
        self.parameter = torch.FloatTensor([parameter])
        self.rotation_type = algorithm

    def __call__(self, x):
        sin = torch.sin(torch.deg2rad(self.parameter))
        tan = torch.tan(torch.deg2rad(self.parameter/2))

        x = x.permute(1, 2, 0)

        if self.rotation_type == 1:
            new_dim_rows = x.size(0)
            new_dim_columns = x.size(1)
        elif self.rotation_type == 2:
            new_dim_rows = x.size(0) + 100
            new_dim_columns = x.size(1) + 100
        else:
            sys.exit('Unknown rotation type!')

        tensor = torch.zeros((new_dim_rows, new_dim_columns, 3)).cuda()

        outer_matrix = torch.FloatTensor([[1, -tan],
                                          [0, 1]]).cuda()
        inner_matrix = torch.FloatTensor([[1, 0],
                                          [sin, 1]]).cuda()

        first_translation = torch.FloatTensor([int(-x.size(0) / 2), int(-x.size(1) / 2)]).cuda()
        second_translation = torch.FloatTensor([int(new_dim_rows / 2), int(new_dim_columns / 2)]).cuda()

        for row in range(x.size(0)):
            for column in range(x.size(1)):
                point_vector = torch.FloatTensor([row, column]).cuda()

                point_vector = torch.add(point_vector, first_translation)

                point_vector = torch.round(torch.matmul(point_vector, outer_matrix))
                point_vector = torch.matmul(torch.round(point_vector), inner_matrix)
                point_vector = torch.matmul(torch.round(point_vector), outer_matrix)

                point_vector = torch.add(point_vector, second_translation)

                new_row = torch.round(point_vector[0]).int().item()
                new_column = torch.round(point_vector[1]).int().item()

                if self.rotation_type == 1:
                    if x.size(0) > new_row >= 0 and 0 <= new_column < x.size(1):
                        tensor[new_row][new_column] = x[row][column]
                else:
                    tensor[new_row][new_column] = x[row][column]

        return tensor.permute(2, 0, 1)


class Transformation:
    def __init__(self, transformation_type):
        if transformation_type not in get_transformation_bounds_dict():
            sys.exit('Invalid transformation type!')

        self.transformation_type = transformation_type
        self.lower_bound, self.upper_bound = get_transformation_bounds_dict().get(self.transformation_type)
        self.parameter = None
        self.algorithm = 1

    def __call__(self, x):
        if self.parameter is None:
            sys.exit('Transformation parameter is None!')
        return self.transform(x)

    def set_random_parameter(self):
        self.parameter = random.uniform(self.lower_bound, self.upper_bound)

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def transform(self, x):
        if self.transformation_type == 'noise':
            return Noise(parameter=self.parameter, algorithm=self.algorithm)(x)
        if self.transformation_type == 'light':
            return LightAdjustment(parameter=self.parameter, algorithm=self.algorithm)(x)
        if self.transformation_type == 'rotation':
            return Rotation(parameter=self.parameter, algorithm=self.algorithm)(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--transformation_type', type=str, default='rotation')
    args = parser.parse_args()

    image = Image.open(args.image)
    transforms = torchvision.transforms.ToTensor()
    image = transforms(image)

    transformation = Transformation(args.transformation_type)
    transformation.set_random_parameter()
    image = transformation(image)

    plot(image)


if __name__ == '__main__':
    main()
