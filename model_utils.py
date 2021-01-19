import torch
import torchvision
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
import os
import pretrainedmodels

STANDARD_PARAMETERS = {
    'torchvision': [True],
    'pretrainedmodels': [1000, 'imagenet'],
}

LOADERS = {
    'pretrainedmodels': pretrainedmodels.models,
    'torchvision': torchvision.models,
}

MODELS_LIST = [
    'resnet34',
    'fbresnet152',
    'bninception',
    'resnext101_32x4d',
    'resnext101_64x4d',
    'inceptionv4',
    'inceptionresnetv2',
    'alexnet',
    'densenet121',
    'densenet169',
    'densenet201',
    'densenet161',
    'resnet18',
    'resnet50',
    'resnet101',
    'resnet152',
    'inceptionv3',
    'squeezenet1_0',
    'squeezenet1_1',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19_bn',
    'vgg19',
    'nasnetamobile',
    'nasnetalarge',
    'dpn68',
    'dpn68b',
    'dpn92',
    'dpn98',
    'dpn131',
    'dpn107',
    'xception',
    'senet154',
    'se_resnet50',
    'se_resnet101',
    'se_resnet152',
    'se_resnext50_32x4d',
    'se_resnext101_32x4d',
    'cafferesnet101',
    'pnasnet5large',
    'polynet',
    'mobilenet_v2',
    'inception_v3'
]


def download_models():
    for arch in MODELS_LIST:
        if not os.path.exists('models/transfer_archs/' + arch + '.pt'):
            for loader_type in LOADERS.keys():
                loader = LOADERS[loader_type]
                if arch in loader.__dict__.keys():
                    try:
                        model = get_model(arch, 'standard', loader_type)
                        break
                    except EOFError:
                        break


def convert_to_robustness(model, state_dict):
    dataset = ImageNet('dataset/imagenet-airplanes')
    model, _ = make_and_restore_model(arch=model, dataset=dataset)
    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    return model, state_dict


def get_model(arch, parameters=None, loader_type='torchvision'):
    if loader_type in LOADERS:
        loader = LOADERS[loader_type]

        if parameters is None:
            parameters = []
        elif parameters == 'standard':
            parameters = STANDARD_PARAMETERS[loader_type]

        if type(parameters) == list:
            if arch in MODELS_LIST:
                model = loader.__dict__[arch](*parameters)
                model.arch = arch
                return model
            else:
                raise ValueError('Specified model is not in the list of available ones!')
        else:
            raise ValueError('Incorrect model parameters format - has to be a list!')
    else:
        raise ValueError('Invalid package for model loading specified!')


def get_state_dict(location):
    obj = torch.load(location)

    if type(obj) == dict:
        if 'state_dict' or 'model' in obj.keys():
            state_dict_key = 'state_dict' if 'state_dict' in obj.keys() else 'model'
            return obj[state_dict_key]
        else:
            raise ValueError('Serialized dictionary does not have a property state_dict!')
    else:
        print('Serialized object is not a dict, returning the object itself!')
        return obj


def load_model(location, arch=None, from_robustness=False, loader_type='torchvision'):
    if os.path.exists(location):
        if from_robustness:
            if arch is None:
                raise ValueError('Please, specify model architecture name when loading with robustness')
            model, state_dict = convert_to_robustness(get_model('resnet50',
                                                                parameters=None,
                                                                loader_type=loader_type),
                                                      get_state_dict(location))
            model.load_state_dict(state_dict)
            return model

        state_dict = get_state_dict(location=location)
        model = get_model(arch=arch, parameters=None)
        model.load_state_dict(state_dict)
        return model
    else:
        raise ValueError('Invalid checkpoint location')


def predict(model, image):
    prediction = model(image.unsqueeze(0))
    if type(prediction) == tuple:
        return prediction[0]
    return prediction
