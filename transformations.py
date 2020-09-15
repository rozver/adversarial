import torch
import random
import torchvision
from PIL import Image
from matplotlib import pyplot as plt


def get_transformation_bounds_dict():
    bounds_dict = {
        'scale': [0.9, 1.4],
        'rotation': [0, 22.5],
        'light': [-0.05, 0.05],
        'noise': [0.0, 0.1]
    }
    return bounds_dict


class LightAdjustment:
    def __init__(self, parameter):
        self.parameter = parameter

    def __call__(self, x):
        return x + torch.ones(x.size())*self.parameter


class GaussianNoise:
    def __init__(self, parameter=0.03):
        self.parameter = parameter

    def __call__(self, x):
        return x + torch.zeros(x.size()).data.normal_()*self.parameter


class Transformation:
    def __init__(self, transformation_type):
        self.transformation_type = transformation_type
        self.lower_bound, self.upper_bound = get_transformation_bounds_dict().get(self.transformation_type)
        self.parameter = None

    def __call__(self, x):
        return self.transform(x)

    def set_random_parameter(self):
        self.parameter = random.uniform(self.lower_bound, self.upper_bound)

    def transform(self, x):
        if self.transformation_type == 'noise':
            return GaussianNoise(parameter=self.parameter)(x)
        if self.transformation_type == 'light':
            return LightAdjustment(parameter=self.parameter)(x)


def main():
    image = Image.open('adv.jpg')
    transforms = torchvision.transforms.ToTensor()
    image = transforms(image)

    transformation = Transformation('light')
    transformation.set_random_parameter()
    image = transformation(image)

    plt.imshow(image.permute(1, 2, 0))
    plt.show()

    transformation = Transformation('noise')
    transformation.set_random_parameter()
    image = transformation(image)

    plt.imshow(image.permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    main()
