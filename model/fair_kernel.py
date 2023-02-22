import torch
import torchvision
import numpy as np

from model.base import BaseModel
from model.network import Network


class FairKernel(BaseModel):
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(FairKernel, self).__init__()

    def _build_modules(self):
        self._build_featurizer()
        self._init_parameters()

    def _build_featurizer(self):
        try:
            featurizer = eval("torchvision.models.{}(pretrained=True)".format(self.cfg.network['featurizer']))
            self.featurizer = torch.nn.Sequential(*(list(featurizer.children())[:-1]))
        except:
            try:
                self.featurizer = Network(self.cfg.network['featurizer'])
            except:
                self.featurizer = torch.nn.Identity()
        self.fc = Network(self.cfg.network['fc'])
        if self.cfg.finetune:
            self.kernel_fc = Network(self.cfg.network['kernel_fc'])

    def _init_parameters(self):
        sigma_phi = torch.FloatTensor(np.ones(1) * np.sqrt(0.005))
        self.register_parameter(name = 'sigma_phi', param = torch.nn.Parameter(sigma_phi))

    def forward(self, X):
        feature = self.featurizer(X)
        feature = torch.flatten(feature, start_dim=1)
        feature = self.fc(feature)

        if self.cfg.finetune:
            feature = self.kernel_fc(feature.detach())
        return feature


class FairKernelBuilder(object):
    """FairKernel Model Builder Class
    """

    def __init__(self):
        """FairKernel Model Builder Class Constructor
        """
        self._instance = None

    def __call__(self, model_cfg, **_ignored):
        """Callback function
        Args:
            model_cfg (ModelConfig): Model Config object
            **_ignored: ignore extra arguments
        Returns:
            FairKernel: Instantiated FairKernel network object
        """
        self._instance = FairKernel(model_cfg=model_cfg)
        return self._instance
