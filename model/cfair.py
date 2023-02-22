import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from model.network import Network


class GradReverse(Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)

class CFairNet(nn.Module):
    """
    CNN with adversarial training for conditional fairness.
    """
    def __init__(self, model_cfg):
        super(CFairNet, self).__init__()
        self.model_cfg = model_cfg
        self.num_classes = model_cfg.num_classes

        try:
            featurizer = eval("torchvision.models.{}(pretrained=True)".format(self.model_cfg.encoder))
            self.encoder = torch.nn.Sequential(*(list(featurizer.children())[:-1]))
        except:
            try:
                self.encoder = Network(self.model_cfg.encoder)
            except:
                self.encoder = torch.nn.Identity()

        self.encoder_fc = Network(model_cfg.encoder_fc)

        self.classifier = Network(model_cfg.classifier)

        self.discriminator = nn.ModuleList(
            [Network(model_cfg.discriminator) for _ in range(self.num_classes)]
        )

        self.kernel = False
        if model_cfg.trainer_key == 'fair_kernel_dp':
            sigma_phi = torch.FloatTensor(np.ones(1) * np.sqrt(0.005))
            self.register_parameter(name = 'sigma_phi', param = torch.nn.Parameter(sigma_phi))
            self.kernel_fc = Network(model_cfg.kernel_fc)
            self.kernel = True

    def forward(self, inputs, labels=None):
        h_relu = inputs
        h_relu = self.encoder(h_relu)
        h_relu = self.encoder_fc(torch.flatten(h_relu, start_dim=1))

        if self.kernel:
            features = self.kernel_fc(h_relu.detach())
            return features

        # Classification probabilities
        if self.model_cfg.s_cls:
            h_relu = h_relu.detach()
        pre_softmax = self.classifier(h_relu)
        logprobs = F.log_softmax(pre_softmax, dim=1)

        # Adversary classification component
        c_losses = []
        h_relu = grad_reverse(h_relu)
        for j in range(self.num_classes):
            idx = labels == j
            c_h_relu = h_relu[idx]
            c_cls = F.log_softmax(self.discriminator[j](c_h_relu), dim=1)
            c_losses.append(c_cls)

        return logprobs, c_losses

    def inference(self, inputs):
        h_relu = inputs
        h_relu = self.encoder(h_relu)
        h_relu = self.encoder_fc(torch.flatten(h_relu, start_dim=1))

        # Classification probabilities
        pre_softmax = self.classifier(h_relu)
        logprobs = F.log_softmax(pre_softmax, dim=1)

        return logprobs, h_relu


class CFairNetBuilder:
    """CfairNet Model Builder Class
    """

    def __init__(self):
        """CfairNet Model Builder Class Constructor
        """
        self._instance = None

    def __call__(self, model_cfg, **_ignored):
        """Callback function
        Args:
            model_cfg (ModelConfig): Model Config object
            **_ignored: ignore extra arguments
        Returns:
            CfairNet: Instantiated CfairNet network object
        """
        self._instance = CFairNet(model_cfg=model_cfg)
        return self._instance
