import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model.network import Network


class LaftrNet(nn.Module):
    def __init__(self, model_cfg):
        super(LaftrNet, self).__init__()
        self.model_cfg = model_cfg

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
        self.discriminator = Network(model_cfg.discriminator)

        self.kernel = False
        if model_cfg.trainer_key == 'fair_kernel_dp':
            sigma_phi = torch.FloatTensor(np.ones(1) * np.sqrt(0.005))
            self.register_parameter(name = 'sigma_phi', param = torch.nn.Parameter(sigma_phi))
            self.kernel_fc = Network(model_cfg.network['kernel_fc'])
            self.kernel = True

    def forward(self, inputs, **_ignored):
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
        apreds = self.discriminator(h_relu)
        return logprobs, apreds

    def inference(self, inputs, **_ignored):
        h_relu = inputs
        h_relu = self.encoder(h_relu)
        h_relu = self.encoder_fc(torch.flatten(h_relu, start_dim=1))

        # Classification probabilities
        pre_softmax = self.classifier(h_relu)
        logprobs = F.log_softmax(pre_softmax, dim=1)

        return logprobs, h_relu


class LaftrNetBuilder:
    """LaftrNet Model Builder Class
    """

    def __init__(self):
        """LaftrNet Model Builder Class Constructor
        """
        self._instance = None

    def __call__(self, model_cfg, **_ignored):
        """Callback function
        Args:
            model_cfg (ModelConfig): Model Config object
            **_ignored: ignore extra arguments
        Returns:
            LaftrNet: Instantiated LaftrNet network object
        """
        self._instance = LaftrNet(model_cfg=model_cfg)
        return self._instance
