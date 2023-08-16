import torch

import torch.nn as nn

from configs import digsfunc

from timm.models.registry import register_model

from models import convnext, efficientnet, resnet, vision_transformer, t2t_vit, cait


class mohs_modals_factory(nn.Module):
    def __init__(self, sub_model, **kwargs):
        super().__init__()
        self.fc = nn.Linear(1001, kwargs['num_classes'])
        kwargs['num_classes'] = 1000
        self.forward_features = sub_model(**kwargs)

    def forward(self, x):
        x, rigs = x
        x = self.forward_features(x)
        x = torch.cat([x, rigs], dim=1)
        x = self.fc(x)
        return x


@register_model
def mohs_efficientnet_b0(**kwargs):
    model = mohs_modals_factory(efficientnet.efficientnet_b0, **kwargs)
    return model


@register_model
def mohs_efficientnet_b1(**kwargs):
    model = mohs_modals_factory(efficientnet.efficientnet_b1, **kwargs)
    return model


@register_model
def mohs_efficientnet_b2(**kwargs):
    model = mohs_modals_factory(efficientnet.efficientnet_b2, **kwargs)
    return model


@register_model
def mohs_efficientnet_b3(**kwargs):
    model = mohs_modals_factory(efficientnet.efficientnet_b3, **kwargs)
    return model


@register_model
def mohs_efficientnet_b4(**kwargs):
    model = mohs_modals_factory(efficientnet.efficientnet_b4, **kwargs)
    return model


@register_model
def mohs_efficientnet_b5(**kwargs):
    model = mohs_modals_factory(efficientnet.efficientnet_b5, **kwargs)
    return model


@register_model
def mohs_efficientnet_b6(**kwargs):
    model = mohs_modals_factory(efficientnet.efficientnet_b6, **kwargs)
    return model


@register_model
def mohs_efficientnet_b7(**kwargs):
    model = mohs_modals_factory(efficientnet.efficientnet_b7, **kwargs)
    return model


@register_model
def mohs_efficientnet_v2_s(**kwargs):
    model = mohs_modals_factory(efficientnet.efficientnet_v2_s, **kwargs)
    return model


@register_model
def mohs_efficientnet_v2_m(**kwargs):
    model = mohs_modals_factory(efficientnet.efficientnet_v2_m, **kwargs)
    return model


@register_model
def mohs_efficientnet_v2_l(**kwargs):
    model = mohs_modals_factory(efficientnet.efficientnet_v2_l, **kwargs)
    return model


@register_model
def mohs_convnext_small(**kwargs):
    model = mohs_modals_factory(convnext.convnext_small, **kwargs)
    return model


@register_model
def mohs_convnext_base(**kwargs):
    model = mohs_modals_factory(convnext.convnext_base, **kwargs)
    return model


@register_model
def mohs_convnext_large(**kwargs):
    model = mohs_modals_factory(convnext.convnext_large, **kwargs)
    return model


@register_model
def mohs_convnext_xlarge(**kwargs):
    model = mohs_modals_factory(convnext.convnext_xlarge, **kwargs)
    return model


@register_model
def mohs_resnet18(**kwargs):
    model = mohs_modals_factory(resnet.resnet18, **kwargs)
    return model


@register_model
def mohs_resnet34(**kwargs):
    model = mohs_modals_factory(resnet.resnet34, **kwargs)
    return model


@register_model
def mohs_resnet50(**kwargs):
    model = mohs_modals_factory(resnet.resnet50, **kwargs)
    return model


@register_model
def mohs_resnet101(**kwargs):
    model = mohs_modals_factory(resnet.resnet50, **kwargs)
    return model


@register_model
def mohs_resnet152(**kwargs):
    model = mohs_modals_factory(resnet.resnet152, **kwargs)
    return model


@register_model
def mohs_resnext50_32x4d(**kwargs):
    model = mohs_modals_factory(resnet.resnext50_32x4d, **kwargs)
    return model


@register_model
def mohs_resnext101_32x8d(**kwargs):
    model = mohs_modals_factory(resnet.resnext101_32x8d, **kwargs)
    return model


@register_model
def mohs_resnext101_64x4d(**kwargs):
    model = mohs_modals_factory(resnet.resnext101_64x4d, **kwargs)
    return model


@register_model
def mohs_wide_resnet50_2(**kwargs):
    model = mohs_modals_factory(resnet.wide_resnet50_2, **kwargs)
    return model


@register_model
def mohs_wide_resnet101_2(**kwargs):
    model = mohs_modals_factory(resnet.wide_resnet101_2, **kwargs)
    return model


@register_model
def mohs_vit_b_16(**kwargs):
    model = mohs_modals_factory(vision_transformer.vit_b_16, **kwargs)
    return model


@register_model
def mohs_vit_b_32(**kwargs):
    model = mohs_modals_factory(vision_transformer.vit_b_32, **kwargs)
    return model


@register_model
def mohs_vit_l_16(**kwargs):
    model = mohs_modals_factory(vision_transformer.vit_l_16, **kwargs)
    return model


@register_model
def mohs_vit_l_32(**kwargs):
    model = mohs_modals_factory(vision_transformer.vit_l_32, **kwargs)
    return model


@register_model
def mohs_vit_h_14(**kwargs):
    model = mohs_modals_factory(vision_transformer.vit_h_14, **kwargs)
    return model


@register_model
def mohs_t2t_vit_7(**kwargs):
    model = mohs_modals_factory(t2t_vit.t2t_vit_7, **kwargs)
    return model


@register_model
def mohs_t2t_vit_10(**kwargs):
    model = mohs_modals_factory(t2t_vit.t2t_vit_10, **kwargs)
    return model


@register_model
def mohs_t2t_vit_12(**kwargs):
    model = mohs_modals_factory(t2t_vit.t2t_vit_12, **kwargs)
    return model


@register_model
def mohs_t2t_vit_14(**kwargs):
    model = mohs_modals_factory(t2t_vit.t2t_vit_14, **kwargs)
    return model


@register_model
def mohs_t2t_vit_19(**kwargs):
    model = mohs_modals_factory(t2t_vit.t2t_vit_19, **kwargs)
    return model


@register_model
def mohs_cait_XXS24_224(**kwargs):
    model = mohs_modals_factory(cait.cait_XXS24_224, **kwargs)
    return model


@register_model
def mohs_cait_XXS36_224(**kwargs):
    model = mohs_modals_factory(cait.cait_XXS36_224, **kwargs)
    return model


@register_model
def mohs_cait_S24_224(**kwargs):
    model = mohs_modals_factory(cait.cait_S24_224, **kwargs)
    return model


if __name__ == '__main__':
    data = torch.rand(1, 3, 224, 224)
    model = mohs_t2t_vit_7(pretrained=False, num_classes=100)
    rigs = digsfunc(['dali', 'ganlan', 'huagan', 'liyan', 'niyan', 'xuanwu'], torch.tensor([1]))
    out = model([data, rigs])
    print(out.shape)
