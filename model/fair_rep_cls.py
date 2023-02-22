import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from model.base import BaseModel
from model.network import Network

from utils.utils import copy_state_dict


class FairRepCls(BaseModel):
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(FairRepCls, self).__init__()

    def _build_modules(self):
        self._build_featurizer()
        self._init_parameters()

    def _build_featurizer(self):
        try:
            featurizer = eval("torchvision.models.{}(pretrained=True)".format(self.cfg.network['featurizer']))
            if self.cfg.pretrained_base is not None:
                pretrained_base = torch.load(self.cfg.pretrained_base)
                copy_state_dict(featurizer.state_dict(), pretrained_base)
            self.featurizer = torch.nn.Sequential(*(list(featurizer.children())[:-1]))
        except:
            try:
                self.featurizer = Network(self.cfg.network['featurizer'])
            except:
                self.featurizer = torch.nn.Identity()
        self.fc = Network(self.cfg.network['fc'])
        self.classifier = Network(self.cfg.network['classifier'])

    def _init_parameters(self):
        sigma_phi = torch.FloatTensor(np.ones(1) * np.sqrt(0.005))
        self.register_parameter(name = 'sigma_phi', param = torch.nn.Parameter(sigma_phi))

    def forward(self, x):
        features = self.featurizer(x)
        features = torch.flatten(features, start_dim=1)
        # if self.cfg.pretrained_base is not None:
        #     features = features.detach()
        features = self.fc(features)

        pre_softmax = self.classifier(features)
        logprobs = F.log_softmax(pre_softmax, dim=1)

        return features, logprobs


class FairRepClsBuilder(object):
    """FairRepCls Model Builder Class
    """

    def __init__(self):
        """FairRepCls Model Builder Class Constructor
        """
        self._instance = None

    def __call__(self, model_cfg, **_ignored):
        """Callback function
        Args:
            model_cfg (ModelConfig): Model Config object
            **_ignored: ignore extra arguments
        Returns:
            FairRepCls: Instantiated FairRepCls network object
        """
        self._instance = FairRepCls(model_cfg=model_cfg)
        return self._instance
