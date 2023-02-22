from tqdm import tqdm
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn.functional as F

from trainer.base import BaseTrainer
from utils.utils import reset_weights


class Classifier(BaseTrainer):
    def __init__(self, data_cfg, model_cfg, exp_cfg) -> None:
        super(Classifier, self).__init__(
            data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg
        )
        if self.model_cfg.finetune:
            reset_weights(self.model.classifier)

    def _setup_optimizers(self):
        params = list(self.model.parameters())
        optim_cfg = self.model_cfg.optimizers['cls']

        self.opt =  eval(
            "optim.{}(params, **{})".format([*optim_cfg.keys()][0], [*optim_cfg.values()][0])
        )

        if self.exp_cfg.resume:
            saved_opt = torch.load(self.exp_cfg.load, map_location=self.device)['optimizer']
            self.opt.load_state_dict(saved_opt)

    def forward(self, mode:str, return_ft:bool=False):
        self.model.eval()
        tqdm_iter = tqdm(range(len(self.loaders[mode])))
        data_iter = iter(self.loaders[mode])

        data = defaultdict(list)
        for _ in tqdm_iter:
            xs, ts, ss = next(data_iter)
            xs = xs.to(self.device)
            with torch.no_grad():
                if return_ft:
                    logprobs, feature = self.model(xs, finetune=False, return_ft=return_ft)
                else:
                    logprobs = self.model(xs, finetune=False, return_ft=return_ft)
            pred = torch.max(logprobs, dim=1)[1].cpu().numpy()
            data['pred'].append(pred)
            data['ss'].append(ss.numpy())
            if self.model_cfg.scls:
                data['actual'].append(ss.numpy())
            else:
                data['actual'].append(ts.numpy())
            if return_ft:
                data['ft'].append(feature.detach().cpu().numpy())
        
        return data

    def _epoch(self, mode:str, epochID:int):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        tqdm_iter = tqdm(range(len(self.loaders[mode])))
        data_iter = iter(self.loaders[mode])
        metrics = defaultdict(list)

        for _ in tqdm_iter:
            xs, ts, ss = next(data_iter)
            xs, ts, ss = xs.to(self.device), ts.to(self.device), ss.to(self.device)

            if mode == 'train':
                logprobs = self.model(xs, finetune=self.model_cfg.finetune)
            else:
                with torch.no_grad():
                    logprobs = self.model(xs, finetune=self.model_cfg.finetune)

            if self.model_cfg.scls:
                loss = F.nll_loss(logprobs, ss)
            else: 
                loss = F.nll_loss(logprobs, ts)
            if mode == 'train':
                self._backprop(loss, self.opt)
            
            metrics['loss'].append(loss.item())

            tqdm_iter.set_description('V: {} | {} | Epoch {} | Loss {:.4f}'.format(self.exp_cfg.version, mode, epochID, loss.item()), refresh=True)
        
        agg_stats = self._aggregate(metrics)
        self._log(epochID, mode, agg_stats)

        return agg_stats['loss']


class ClassifierBuilder(object):
    """Classifier  Builder Class
    """

    def __init__(self):
        """Classifier  Builder Class Constructor
        """
        self._instance = None

    def __call__(self, data_cfg, model_cfg, exp_cfg, **_ignored):
        """Callback function
        Args:
            data_cfg (Config): Data Config object
            model_cfg (Config): Model Config object
            exp_cfg (Config): Experiment Config object
        Returns:
            Classifier: Instantiated Classifier  object
        """
        self._instance = Classifier(data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg)
        return self._instance
