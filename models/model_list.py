

model_list = [
    # ConvNeXt
    'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge',
    # AdacaNet test model
    'adacanet_small', 'adacanet_base', 'adacanet_large', 'adacanet_xlarge',
    # ResNet
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    # ResNeXt & wide resnet
    'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'wide_resnet50_2', 'wide_resnet101_2',
    # EfficientNet V1
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',
    'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
    # EfficientNet V2
    'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
    # VGG
    'myvgg16', 'myvgg16_bn', 'myvgg19', 'myvgg19_bn',
    # GoogleNet
    'mygooglenet',

    # vision transformer
    'myvit_b_16', 'myvit_b_32', 'myvit_l_16', 'myvit_l_32', 'myvit_h_14',
    # token to token vision transformer
    'myt2t_vit_7', 'myt2t_vit_10', 'myt2t_vit_12', 'myt2t_vit_14', 'myt2t_vit_19', 'myt2t_vit_24',
    'myt2t_vit_t_14', 'myt2t_vit_t_19', 'myt2t_vit_t_24', 'myt2t_vit_14_resnext', 'myt2t_vit_14_wide',
    'myt2t_vit_dense',
    # regionvit
    'regionvit_tiny_224', 'regionvit_small_224', 'regionvit_small_w14_peg_224',
    'regionvit_medium_224', 'regionvit_base_224', 'regionvit_base_w14_224',
    'regionvit_base_w14_peg_224',
    # MaxVit
    'max_vit_tiny_224', 'max_vit_small_224', 'max_vit_base_224', 'max_vit_large_224',
    # cct
    'cct_7_7x2_224', 'cct_7_7x2_224_sine', 'cct_14_7x2_224',
    # cait
    'cait_XXS24_224', 'cait_XXS36_224', 'cait_S24_224',
    # swin transformer
    'swin_t', 'swin_s', 'swin_b',

    # Multi modals-data: Moh's hardness
    'mohs_efficientnet_b0', 'mohs_efficientnet_b1',
    'mohs_efficientnet_b2', 'mohs_efficientnet_b3',
    'mohs_efficientnet_b4', 'mohs_efficientnet_b5',
    'mohs_efficientnet_b6', 'mohs_efficientnet_b7',
    'mohs_efficientnet_v2_s',
    'mohs_convnext_small', 'mohs_convnext_base',
    'mohs_convnext_large', 'mohs_convnext_xlarge',
    'mohs_resnet18', 'mohs_resnet34', 'mohs_resnet50',
    'mohs_resnet101', 'mohs_resnet152',
    'mohs_resnext50_32x4d', 'mohs_resnext101_32x8d',
    'mohs_resnext101_64x4d',
    'mohs_wide_resnet50_2', 'mohs_wide_resnet101_2',
    'mohs_vit_b_16', 'mohs_vit_b_32', 'mohs_vit_l_16', 'mohs_vit_l_32',
    'mohs_myt2t_vit_7', 'mohs_myt2t_vit_10', 'mohs_myt2t_vit_12', 'mohs_myt2t_vit_14', 'mohs_myt2t_vit_19',

    ]