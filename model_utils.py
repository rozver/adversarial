import torch
import torchvision
from robustness.datasets import  ImageNet
from robustness.model_utils import make_and_restore_model
import os


MODELS_LIST = [
    'resnet18',
    'resnet50',
    'resnet152',
    'alexnet',
    'vgg16',
    'vgg19',
    'inception_v3',
]


def get_model(arch, pretrained=False):
    if arch in MODELS_LIST:
        model = getattr(torchvision.models, arch)()
        model.arch = arch
        if pretrained:
            state_dict = torch.load('models/' + arch + '.pt')
            model.load_state_dict(state_dict)
        return model
    else:
        raise ValueError('Specified model is not in the list of available ones!')


def get_state_dict(location, return_model_arch=False):
    obj = torch.load(location)

    if type(obj) == dict:
        if 'state_dict' in obj.keys():
            if return_model_arch:
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
        if return_model_arch:
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
            state_dict = get_state_dict(location=location, return_model_arch=False)
        else:
            state_dict = get_state_dict(location=location, return_model_arch=True)
        model = get_model(arch=arch, pretrained=False)
        model.load_state_dict(state_dict)
        return model
    else:
        raise ValueError('Invalid checkpoint location')
