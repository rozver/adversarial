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
    'torchvision': torchvision.models,
    'pretrainedmodels': pretrainedmodels.models,
}

TORCHVISION_ARCHS = [
    'mobilenet_v2',
    'shufflenet_v2_x0_5',
    'squeezenet1_1',
    'mnasnet1_0',
    'googlenet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152'
]

PRETRAINEDMODELS_ARCHS = [
    'fbresnet152',
    'bninception',
    'resnext101_32x4d',
    'resnext101_64x4d',
    'alexnet',
    'densenet121',
    'densenet169',
    'densenet201',
    'densenet161',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19_bn',
    'vgg19',
    'nasnetamobile',
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
    'cafferesnet101'
]

OTHER_MODELS = [
    'inception_v4',
    'inceptionresnetv2',
    'inceptionv3',
    'squeezenet1_0',
    'nasnetalarge',
    'pnasnet5large',
    'polynet'
]

ARCHS_LIST = TORCHVISION_ARCHS + PRETRAINEDMODELS_ARCHS


def predict(model, x):
    if len(x.size()) != 4:
        x = x.unsqueeze(0)

    prediction = model(x)

    if type(prediction) == tuple:
        return prediction[0]
    return prediction


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def get_archs_dict():
    torchvision_models_dict = dict.fromkeys(TORCHVISION_ARCHS, 'torchvision')
    pretrainedmodels_dict = dict.fromkeys(PRETRAINEDMODELS_ARCHS, 'pretrainedmodels')
    archs_dict = {**torchvision_models_dict, **pretrainedmodels_dict}
    return archs_dict


def download_models():
    for arch in ARCHS_LIST:
        if not os.path.exists('models/transfer_archs/' + arch + '.pt'):
            if arch in ARCHS_LIST:
                try:
                    model = get_model(arch, 'standard')
                    print('Model ' + arch + ' successfully downloaded!')
                except EOFError:
                    continue


def convert_to_robustness(model, state_dict):
    dataset = ImageNet('dataset/imagenet-airplanes')
    model, _ = make_and_restore_model(arch=model, dataset=dataset)
    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    return model, state_dict


def get_model(arch, parameters=None, freeze=False):
    if arch in ARCHS_LIST:
        archs_dict = get_archs_dict()
        loader = LOADERS[archs_dict[arch]]

        if parameters is None:
            parameters = []
        elif parameters == 'standard':
            parameters = STANDARD_PARAMETERS[archs_dict[arch]]
        if type(parameters) == list:
            if len(parameters) == 2 and parameters[1] in pretrainedmodels.pretrained_settings[arch]:
                model = loader.__dict__[arch](*parameters)
            else:
                model = loader.__dict__[arch](*parameters[:1])
            model.arch = arch

            if archs_dict[arch] == 'pretrainedmodels' and len(parameters) == 0:
                model.apply(weight_reset)

            if freeze:
                model = freeze_parameters(model)
            return model

        else:
            raise ValueError('Incorrect model parameters format - has to be a list!')
    else:
        raise ValueError('Specified model is not in the list of available ones!')


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


def load_model(location, arch=None, from_robustness=False):
    if os.path.exists(location):
        if from_robustness:
            if arch is None:
                raise ValueError('Please, specify model architecture name when loading with robustness')
            model, state_dict = convert_to_robustness(get_model('resnet50',
                                                                parameters=None),
                                                      get_state_dict(location))
            model.load_state_dict(state_dict)
            return model.model

        state_dict = get_state_dict(location=location)
        model = get_model(arch=arch, parameters=None)
        model.load_state_dict(state_dict)
        return model
    else:
        raise ValueError('Invalid checkpoint location')
