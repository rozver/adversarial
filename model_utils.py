import torch
import timm
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
import os
import urllib

TIMM_ARCHS = timm.list_models(pretrained=True)

UNSUPPORTED_ARCHS = [
    'ecaresnet50d',
    'ecaresnet50d_pruned',
    'ecaresnetlight',
    'ecaresnet101d_pruned',
    'ecaresnet101d',
    'efficientnet_b1_pruned',
    'efficientnet_b2_pruned',
    'efficientnet_b3_pruned',
    'vit_small_patch32_384',
    'cait_xxs36_384',
    'tresnet_m',
    'swin_large_patch4_window12_384_in22k',
    'vit_base_patch16_384',
    'vit_base_r50_s16_384',
    'vit_small_patch16_384',
    'swin_base_patch4_window12_384_in22k',
    'vit_small_r26_s32_384',
    'vit_large_patch32_384',
    'cait_s24_384',
    'vit_large_patch16_384',
    'vit_tiny_r_s16_p8_384',
    'tresnet_m_miil_in21k',
    'tresnet_l_448',
    'tresnet_xl',
    'swin_base_patch4_window12_384',
    'deit_base_patch16_384',
    'tresnet_xl_448',
    'swin_large_patch4_window12_384',
    'cait_xxs24_384',
    'vit_base_patch32_384',
    'cait_m36_384',
    'cait_xs24_384',
    'vit_large_r50_s32_384',
    'deit_base_distilled_patch16_384',
    'vit_tiny_patch16_384',
    'tresnet_m_448',
    'cait_m48_448',
    'tresnet_l',
    'cait_s36_384',
    'vit_small_patch32_384'
]

ARCHS_LIST = ['resnet50', 'resnet34', 'vgg16', 'vgg13', 'vgg19', 'inception_v3']


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


def to_device(x, device):
    if x.device != device:
        x = x.to(device)
        if type(x) != torch.Tensor:
            x.device = device
    return x


def download_models():
    unsupported_archs = []
    for arch in ARCHS_LIST:
        try:
            model = get_model(arch, True)
        except urllib.error.URLError:
            unsupported_archs.append(arch)
            continue
        except EOFError:
            print('Error while downloading model ' + arch + '!')
            continue

    print(unsupported_archs)


def convert_to_robustness(model, state_dict):
    dataset = ImageNet('dataset/imagenet-airplanes')
    model, _ = make_and_restore_model(arch=model, dataset=dataset)
    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    return model, state_dict


def get_model(arch, pretrained=True, freeze=False, device='cpu'):
    if arch in ARCHS_LIST:
        model = timm.create_model(arch, pretrained=pretrained)
        model.arch = arch
        model.device = 'cpu'

        if freeze:
            model = freeze_parameters(model)

        model = to_device(model, device)
        return model
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
                                                                pretrained=False),
                                                      get_state_dict(location))
            model.load_state_dict(state_dict)
            return model.model

        state_dict = get_state_dict(location=location)
        model = get_model(arch=arch, pretrained=False)
        model.load_state_dict(state_dict)

        return model
    else:
        raise ValueError('Invalid checkpoint location')
