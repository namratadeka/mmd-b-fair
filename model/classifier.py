import torch
import torchvision
import torch.nn.functional as F

from model.base import BaseModel
from model.network import Network

from utils.utils import copy_state_dict


class Classifier(BaseModel):
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(Classifier, self).__init__()

    def _build_modules(self):
        self._build_subnets()

    def _build_subnets(self):
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

    def forward(self, X, finetune, return_ft=False):
        feature = self.featurizer(X)
        feature = torch.flatten(feature, start_dim=1)
        feature = self.fc(feature)

        if finetune:
            pre_softmax = self.classifier(feature.detach())
        else:
            pre_softmax = self.classifier(feature)
        logprobs = F.log_softmax(pre_softmax, dim=1)

        if return_ft:
            return logprobs, feature
        return logprobs


class ClassifierBuilder(object):
    """Classifier Model Builder Class
    """

    def __init__(self):
        """Classifier Model Builder Class Constructor
        """
        self._instance = None

    def __call__(self, model_cfg, **_ignored):
        """Callback function
        Args:
            model_cfg (ModelConfig): Model Config object
            **_ignored: ignore extra arguments
        Returns:
            Classifier: Instantiated Classifier network object
        """
        self._instance = Classifier(model_cfg=model_cfg)
        return self._instance
