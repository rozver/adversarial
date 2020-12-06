import torch
import torchvision
from robustness.datasets import  ImageNet
from robustness.model_utils import make_and_restore_model
import os


def get_models_dict(pretrained=False):
    models = {
        'resnet18': torchvision.models.resnet18(pretrained=pretrained).eval(),
        'resnet50': torchvision.models.resnet50(pretrained=pretrained).eval(),
        'resnet152': torchvision.models.resnet152(pretrained=pretrained).eval(),
        'alexnet': torchvision.models.alexnet(pretrained=pretrained).eval(),
        'vgg16': torchvision.models.vgg16(pretrained=pretrained).eval(),
        'vgg19': torchvision.models.vgg19(pretrained=pretrained).eval(),
        'inception_v3': torchvision.models.inception_v3(pretrained=pretrained, init_weights=True).eval(),
    }

    return models


def get_model(arch, pretrained=False):
    models_dict = get_models_dict(pretrained)
    model = models_dict.get(arch, models_dict.get('resnet50'))
    model.name = arch if arch in models_dict.keys() else 'resnet50'
    return model


def get_state_dict(location, return_model_name=False):
    obj = torch.load(location)

    if type(obj) == dict:
        if 'state_dict' in obj.keys():
            if return_model_name:
                if 'state_dict' in obj.keys():
                    return obj['state_dict'], obj['pgd_training_args']['model']
                else:
                    print('Serialized dictionary does not have property pgd_training_args - returning None for name')
                    return obj['state_dict'], None
            return obj['state_dict']
        else:
            raise ValueError('Serialized dictionary does not have a property state_dict!')
    else:
        print('Serialized object is not a dict, returning the object itself (and None if return_model_name)!')
        if return_model_name:
            return obj, None
        return obj


def load_model(location, arch=None, from_robustness=False):
    if os.path.exists(location):
        if from_robustness:
            if arch is None:
                raise ValueError('Please, specify model architecture name when loading with robustness')
            dataset = ImageNet('dataset/imagenet-airplanes')
            model, _ = make_and_restore_model(arch=arch, dataset=dataset, resume_path=location)
            return model
        if arch is not None:
            state_dict = get_state_dict(location=location, return_model_name=False)
        else:
            state_dict = get_state_dict(location=location, return_model_name=True)
        model = get_model(arch=arch, pretrained=False)
        model.load_state_dict(state_dict)
        return model
    else:
        raise ValueError('Invalid checkpoint location')
