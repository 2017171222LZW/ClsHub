import torch
import torch.nn as nn


class ModelFactory(nn.Module):
    def __init__(self, model: nn.Module, num_classes=1000, **kargs):
        super(ModelFactory, self).__init__()
        self.model = model
        if num_classes != 1000:
            self.linear = nn.Linear(1000, num_classes)
        else:
            self.linear = nn.Identity()
        # self.linear = nn.Linear(1000, num_classes)


    def forward(self, x):
        assert x.shape[1] == 3, "Only three-channel data is supported."
        out = self.model(x)
        out = self.linear(out)
        return out


class GoogleFactory(nn.Module):
    def __init__(self, model: nn.Module, num_classes=1000, **kargs):
        super(GoogleFactory, self).__init__()
        self.model = model
        if num_classes != 1000:
            self.linear = nn.Linear(1000, num_classes)
        else:
            self.linear = nn.Identity()

        self.loss_split = torch.nn.Parameter(torch.tensor([0.625, 0.1875, 0.1875]))

    def forward(self, x):
        assert x.shape[1] == 3, "Only three-channel data is supported."
        out = self.model(x)
        # googleNet branch loss
        if not isinstance(out, torch.Tensor):
            loss_split = self.loss_split / torch.sum(self.loss_split)
            out = loss_split[0] * out[0] + loss_split[1] * out[1] + loss_split[2] * out[2]
        out = self.linear(out)
        return out
