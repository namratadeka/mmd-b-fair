import torch
import torchvision

from model.base import BaseModel
from model.network import Network

from utils.utils import copy_state_dict


class sIPMModel(BaseModel):
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(sIPMModel, self).__init__()

    def _build_modules(self):
        self._build_encoder()
        self._build_decoder()
        self._build_discriminator()

    def _build_encoder(self):
        try:
            encoder = eval("torchvision.models.{}(pretrained=True)".format(self.cfg.network['encoder']))
            if self.cfg.pretrained_base is not None:
                pretrained_base = torch.load(self.cfg.pretrained_base)
                copy_state_dict(encoder.state_dict(), pretrained_base)
            self.encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]))
        except:
            try:
                self.encoder = Network(self.cfg.network['encoder'])
            except:
                self.encoder = torch.nn.Identity()
        self.encoder_fc = Network(self.cfg.network['encoder_fc'])
        self.head = Network(self.cfg.network['head'])

    def _build_decoder(self):
        self.decoder = Network(self.cfg.network['decoder'])

    def _build_discriminator(self):
        self.aud_model = Network(self.cfg.network['aud'])

    def forward(self, x):
        z = self.encoder(x)
        z = torch.flatten(z, start_dim=1)
        z = self.encoder_fc(z)
        if self.cfg.finetune:
            preds = self.head(z.detach())
        else:
            preds = self.head(z)
        
        recon = None
        if self.cfg.lmdaR != 0:
            recon = self.decoder(z)

        return z, preds, recon


class sIPMModelBuilder:
    """sIPMModel Builder Class
    """

    def __init__(self):
        """sIPMModel Model Builder Class Constructor
        """
        self._instance = None

    def __call__(self, model_cfg, **_ignored):
        """Callback function
        Args:
            model_cfg (ModelConfig): Model Config object
            **_ignored: ignore extra arguments
        Returns:
            sIPMModel: Instantiated sIPMModel network object
        """
        self._instance = sIPMModel(model_cfg=model_cfg)
        return self._instance   
