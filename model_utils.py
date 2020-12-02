import torch
import torchvision


def get_models_dict(pretrained=False):
    models = {
        'resnet18': torchvision.models.resnet18(pretrained=pretrained).eval(),
        'resnet50': torchvision.models.resnet50(pretrained=pretrained).eval(),
        'resnet152': torchvision.models.resnet152(pretrained=pretrained).eval(),
        'alexnet': torchvision.models.alexnet(pretrained=pretrained).eval(),
        'vgg16': torchvision.models.vgg16(pretrained=pretrained).eval(),
        'vgg19': torchvision.models.vgg19(pretrained=pretrained).eval(),
        'inception_v3': torchvision.models.inception_v3(pretrained=pretrained).eval(),
    }

    return models


def get_model(model_name, pretrained=False):
    models_dict = get_models_dict(pretrained)
    model = models_dict.get(model_name, models_dict.get('resnet50'))
    model.name = model_name if model_name in models_dict.keys() else 'resnet50'
    return model


def get_state_dict(location, return_model_name=False):
    obj = torch.load(location)

    if type(obj) == dict:
        if 'state_dict' in obj.keys():
            if return_model_name:
                return obj['state_dict'], obj['pgd_training_args']['model']
            return obj['state_dict']
        else:
            raise ValueError('Serialized dictionary does not have a property state_dict!')
    else:
        print('Serialized object is not a dict, returning the object itself (and None if return_model_name)!')
        if return_model_name:
            return obj, None
        return obj


def load_model_from_state_dict(location):
    state_dict, model_name = get_state_dict(location=location, return_model_name=True)
    model = get_model(model_name=model_name, pretrained=False)
    model.load_state_dict(state_dict)
    return model
