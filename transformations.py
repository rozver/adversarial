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
        'light': [-0.1, 0.1],
        'noise': [0.0, 0.05],
        'translation': [-10.0, 10.0],
        'rotation': [-35, 35],
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
        raise NotImplementedError


class LightAdjustment(Transformation):
    def __init__(self):
        self.transformation_type = 'light'
        super(LightAdjustment, self).__init__(transformation_type=self.transformation_type)

    def transform(self, x):
        light = torch.cuda.FloatTensor().new_full(x.size(), self.parameter)
        x = torch.add(x, light)
        return x


class Noise(Transformation):
    def __init__(self):
        self.transformation_type = 'noise'
        super(Noise, self).__init__(transformation_type=self.transformation_type)

    def transform(self, x):
        noise = torch.normal(mean=0.0, std=self.parameter, size=x.size(), device=torch.device('cuda'))
        x = torch.add(x, noise).float()
        return x


class Translation(Transformation):
    def __init__(self):
        self.transformation_type = 'translation'
        super(Translation, self).__init__(transformation_type=self.transformation_type)

    def set_random_parameter(self):
        self.parameter = (random.uniform(self.lower_bound, self.upper_bound),
                          random.uniform(self.lower_bound, self.upper_bound))

    def transform(self, x):
        x = x.permute(1, 2, 0)

        output_tensor = torch.ones(x.size())
        translation = torch.round(torch.cuda.FloatTensor([parameter for parameter in self.parameter]))

        for row in range(x.size(0)):
            for column in range(x.size(1)):
                point_vector = torch.cuda.FloatTensor([row, column])

                new_point_vector = torch.add(point_vector, translation)

                new_row = new_point_vector[0].int().item()
                new_column = new_point_vector[1].int().item()

                if x.size(0) > new_row >= 0 and 0 <= new_column < x.size(1):
                    output_tensor[new_row, new_column] = x[row, column]

        return output_tensor.permute(2, 0, 1)


class Rotation(Transformation):
    def __init__(self):
        self.transformation_type = 'rotation'
        super(Rotation, self).__init__(transformation_type=self.transformation_type)

    def transform(self, x):
        x = x.permute(1, 2, 0)

        sin = torch.sin(torch.deg2rad(torch.cuda.FloatTensor([self.parameter])))
        tan = torch.tan(torch.deg2rad(torch.cuda.FloatTensor([self.parameter/2])))

        if self.algorithm == 1:
            new_dim_rows = x.size(0)
            new_dim_columns = x.size(1)
        elif self.algorithm == 2:
            new_dim_rows = x.size(0) + 100
            new_dim_columns = x.size(1) + 100
        else:
            sys.exit('Unknown rotation type!')

        output_tensor = torch.zeros((new_dim_rows, new_dim_columns, 3)).cuda()

        outer_matrix = torch.cuda.FloatTensor([[1, -tan],
                                          [0, 1]])
        inner_matrix = torch.cuda.FloatTensor([[1, 0],
                                          [sin, 1]])

        first_translation = torch.cuda.FloatTensor([int(-x.size(0) / 2), int(-x.size(1) / 2)])
        second_translation = torch.cuda.FloatTensor([int(new_dim_rows / 2), int(new_dim_columns / 2)])

        for row in range(x.size(0)):
            for column in range(x.size(1)):
                point_vector = torch.cuda.FloatTensor([row, column])

                point_vector = torch.add(point_vector, first_translation)

                point_vector = torch.round(torch.matmul(point_vector, outer_matrix))
                point_vector = torch.matmul(torch.round(point_vector), inner_matrix)
                point_vector = torch.matmul(torch.round(point_vector), outer_matrix)

                point_vector = torch.add(point_vector, second_translation)

                new_row = torch.round(point_vector[0]).int().item()
                new_column = torch.round(point_vector[1]).int().item()

                if self.algorithm == 1:
                    if x.size(0) > new_row >= 0 and 0 <= new_column < x.size(1):
                        output_tensor[new_row, new_column] = x[row, column]
                else:
                    output_tensor[new_row, new_column] = x[row, column]

        return output_tensor.permute(2, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--transformation_type', type=str, default='translation')
    args = parser.parse_args()

    image = Image.open(args.image)
    transforms = torchvision.transforms.ToTensor()
    image = transforms(image)

    transformation = get_transformation(args.transformation_type)
    transformation.set_random_parameter()
    image = transformation(image.cuda())

    plot(image.cpu())


if __name__ == '__main__':
    main()
