import torch
import torch.nn as nn
from thop import profile
from timm.models.registry import register_model
from torch.hub import load_state_dict_from_url

from models.ResConViTModules import Tokenizer
from models.helpers import pe_check
from models.ResConViTModules import TransformerClassifier



model_urls = {
}


class ResConViT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=256,
                 n_input_channels=3,
                 kernel_size=16,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 n_conv_layers=1,
                 *args, **kwargs):
        super(ResConViT, self).__init__()
        assert img_size % kernel_size == 0, f"Image size ({img_size}) has to be" \
                                            f"divisible by patch size ({kernel_size})"
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=kernel_size,
                                   padding=0,
                                   max_pool=False,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=True,
                                   img_size=img_size)
        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def resconvit(arch, pretrained, progress,
            num_layers, num_heads, mlp_ratio, embedding_dim,
            kernel_size=4, positional_embedding='sine',
            *args, **kwargs):
    model = ResConViT(num_layers=num_layers,
                   num_heads=num_heads,
                   mlp_ratio=mlp_ratio,
                   embedding_dim=embedding_dim,
                   kernel_size=kernel_size,
                   *args, **kwargs)

    if pretrained and arch in model_urls:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        if positional_embedding == 'learnable':
            state_dict = pe_check(model, state_dict)
        elif positional_embedding == 'sine':
            state_dict['classifier.positional_emb'] = model.state_dict()['classifier.positional_emb']
        model.load_state_dict(state_dict)
    return model


def resconvit_1_2(*args, **kwargs):
    return resconvit(num_layers=1, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=3,
                   *args, **kwargs)


# mlp_ratio 不利于收敛
def resconvit_2(*args, **kwargs):
    return resconvit(num_layers=2, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=2,
                   *args, **kwargs)


def resconvit_4(*args, **kwargs):
    return resconvit(num_layers=4, num_heads=4, mlp_ratio=4, embedding_dim=512, n_conv_layers=2,
                   *args, **kwargs)


def resconvit_6(*args, **kwargs):
    return resconvit(num_layers=6, num_heads=8, mlp_ratio=6, embedding_dim=1024, n_conv_layers=3,
                   *args, **kwargs)


def resconvit_7(*args, **kwargs):
    return resconvit(num_layers=7, num_heads=8, mlp_ratio=6, embedding_dim=2048, n_conv_layers=3,
                   *args, **kwargs)


def resconvit_8(*args, **kwargs):
    return resconvit(num_layers=8, num_heads=16, mlp_ratio=6, embedding_dim=2048, n_conv_layers=3,
                   *args, **kwargs)


def resconvit_12(*args, **kwargs):
    return resconvit(num_layers=1, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=2,
                   *args, **kwargs)


@register_model
def resconvit_1_2_224(pretrained=False, progress=False,
                   img_size=224, positional_embedding='learnable', num_classes=10,
                   *args, **kwargs):
    return resconvit_1_2('resconvit_1_3', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


@register_model
def resconvit_2_4_224(pretrained=False, progress=False,
                   img_size=224, positional_embedding='learnable', num_classes=10,
                   *args, **kwargs):
    return resconvit_2('resconvit_2_4_32', pretrained, progress,
                    kernel_size=4,
                    img_size=img_size, positional_embedding=positional_embedding,
                    num_classes=num_classes,
                    *args, **kwargs)


@register_model
def resconvit_2_4_224_sine(pretrained=False, progress=False,
                        img_size=224, positional_embedding='sine', num_classes=10,
                        *args, **kwargs):
    return resconvit_2('resconvit_2_4_32_sine', pretrained, progress,
                    kernel_size=4,
                    img_size=img_size, positional_embedding=positional_embedding,
                    num_classes=num_classes,
                    *args, **kwargs)


@register_model
def resconvit_4_4_224(pretrained=False, progress=False,
                   img_size=224, positional_embedding='learnable', num_classes=10,
                   *args, **kwargs):
    return resconvit_4('resconvit_4_4_32', pretrained, progress,
                    kernel_size=4,
                    img_size=img_size, positional_embedding=positional_embedding,
                    num_classes=num_classes,
                    *args, **kwargs)


@register_model
def resconvit_4_4_224_sine(pretrained=False, progress=False,
                        img_size=224, positional_embedding='sine', num_classes=10,
                        *args, **kwargs):
    return resconvit_4('resconvit_4_4_32_sine', pretrained, progress,
                    kernel_size=4,
                    img_size=img_size, positional_embedding=positional_embedding,
                    num_classes=num_classes,
                    *args, **kwargs)


@register_model
def resconvit_6_4_224(pretrained=False, progress=False,
                   img_size=224, positional_embedding='learnable', num_classes=10,
                   *args, **kwargs):
    return resconvit_6('resconvit_6_4_32', pretrained, progress,
                    kernel_size=4,
                    img_size=img_size, positional_embedding=positional_embedding,
                    num_classes=num_classes,
                    *args, **kwargs)


@register_model
def resconvit_6_4_224_sine(pretrained=False, progress=False,
                        img_size=224, positional_embedding='sine', num_classes=10,
                        *args, **kwargs):
    return resconvit_6('resconvit_6_4_32_sine', pretrained, progress,
                    kernel_size=4,
                    img_size=img_size, positional_embedding=positional_embedding,
                    num_classes=num_classes,
                    *args, **kwargs)


@register_model
def resconvit_7_4_224(pretrained=False, progress=False,
                   img_size=224, positional_embedding='learnable', num_classes=10,
                   *args, **kwargs):
    return resconvit_7('resconvit_7_4_32', pretrained, progress,
                    kernel_size=4,
                    img_size=img_size, positional_embedding=positional_embedding,
                    num_classes=num_classes,
                    *args, **kwargs)


@register_model
def resconvit_7_4_224_sine(pretrained=False, progress=False,
                        img_size=224, positional_embedding='sine', num_classes=10,
                        *args, **kwargs):
    return resconvit_7('resconvit_7_4_32_sine', pretrained, progress,
                    kernel_size=4,
                    img_size=img_size, positional_embedding=positional_embedding,
                    num_classes=num_classes,
                    *args, **kwargs)


def resconvit_11(*args, **kwargs):
    return resconvit(num_layers=1, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=1,
                   *args, **kwargs)


@register_model
def resconvit_11_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_11('resconvit_11_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


@register_model
def resconvit_12_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_12('resconvit_12_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_13(*args, **kwargs):
    return resconvit(num_layers=1, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=3,
                   *args, **kwargs)


@register_model
def resconvit_13_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_13('resconvit_13_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_21(*args, **kwargs):
    return resconvit(num_layers=2, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=1,
                   *args, **kwargs)


@register_model
def resconvit_21_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_21('resconvit_21_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_22(*args, **kwargs):
    return resconvit(num_layers=2, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=2,
                   *args, **kwargs)


@register_model
def resconvit_22_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_22('resconvit_22_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_23(*args, **kwargs):
    return resconvit(num_layers=2, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=3,
                   *args, **kwargs)


@register_model
def resconvit_23_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_23('resconvit_23_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_31(*args, **kwargs):
    return resconvit(num_layers=3, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=1,
                   *args, **kwargs)


@register_model
def resconvit_31_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_31('resconvit_31_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_32(*args, **kwargs):
    return resconvit(num_layers=3, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=2,
                   *args, **kwargs)


@register_model
def resconvit_32_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_32('resconvit_32_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_33(*args, **kwargs):
    return resconvit(num_layers=3, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=3,
                   *args, **kwargs)


@register_model
def resconvit_33_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_33('resconvit_33_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_41(*args, **kwargs):
    return resconvit(num_layers=4, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=1,
                   *args, **kwargs)


@register_model
def resconvit_41_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_41('resconvit_41_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_42(*args, **kwargs):
    return resconvit(num_layers=4, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=2,
                   *args, **kwargs)


@register_model
def resconvit_42_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_42('resconvit_42_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_43(*args, **kwargs):
    return resconvit(num_layers=4, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=3,
                   *args, **kwargs)


@register_model
def resconvit_43_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_43('resconvit_43_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_51(*args, **kwargs):
    return resconvit(num_layers=5, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=1,
                   *args, **kwargs)


@register_model
def resconvit_51_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_51('resconvit_51_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_52(*args, **kwargs):
    return resconvit(num_layers=5, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=2,
                   *args, **kwargs)


@register_model
def resconvit_52_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_52('resconvit_52_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_53(*args, **kwargs):
    return resconvit(num_layers=5, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=3,
                   *args, **kwargs)


@register_model
def resconvit_53_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_53('resconvit_53_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_61(*args, **kwargs):
    return resconvit(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=1,
                   *args, **kwargs)


@register_model
def resconvit_61_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_61('resconvit_61_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_62(*args, **kwargs):
    return resconvit(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=2,
                   *args, **kwargs)


@register_model
def resconvit_62_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_62('resconvit_62_224', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


def resconvit_63(*args, **kwargs):
    return resconvit(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256, n_conv_layers=3,
                   *args, **kwargs)


@register_model
def resconvit_63_244(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return resconvit_63('resconvit_63_244', pretrained, progress,
                      kernel_size=4,
                      img_size=img_size, positional_embedding=positional_embedding,
                      num_classes=num_classes,
                      *args, **kwargs)


if __name__ == '__main__':
    data = torch.rand(1, 3, 224, 224)
    model = resconvit_63_244(pretrained=False, num_classes=100)
    out = model(data)
    print(out.shape)

    flops, params = profile(model, inputs=(data,), verbose=False)
    print('flops:', flops / 1E6, 'M')
    print('params:', params)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
