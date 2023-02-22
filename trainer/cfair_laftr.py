import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn.functional as F

from trainer.base import BaseTrainer
from utils.utils import reset_weights


class CfairLaftrTrainer(BaseTrainer):
    def __init__(self, data_cfg, model_cfg, exp_cfg):
        super(CfairLaftrTrainer, self).__init__(
            data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg
        )
        if model_cfg.model_key == 'cfair' or model_cfg.model_key == 'cfair-conv':
            self.reweight_tensors()

        if self.model_cfg.s_cls and 'test' not in self.model_cfg.modes:
            reset_weights(self.model.classifier)
            assert self.model_cfg.mu == 0

    def _setup_optimizers(self):
        params = list(self.model.parameters())
        optim_cfg = self.model_cfg.optimizers["opt"]

        self.opt =  eval(
            "optim.{}(params, **{})".format([*optim_cfg.keys()][0], [*optim_cfg.values()][0])
        )

    def reweight_tensors(self):
        self.reweight_target_tensor = {}
        self.reweight_attr_tensors = {}

        for mode in self.model_cfg.modes:
            # Pre-compute the statistics in the set.
            dataset = self.loaders[mode].dataset
            if self.data_cfg.data_key == 'shapes-cls':
                idx = dataset.att == 0 
                base_0, base_1 = np.mean(dataset.lbl[idx]), np.mean(dataset.lbl[~idx])
                y_1 = np.mean(dataset.lbl)
            else:
                idx = dataset.A == 0 
                base_0, base_1 = np.mean(dataset.Y[idx]), np.mean(dataset.Y[~idx])
                y_1 = np.mean(dataset.Y)
            # For reweighing purpose.
            self.reweight_target_tensor[mode] = torch.FloatTensor([1.0 / (1.0 - y_1), 1.0 / y_1]).to(self.device)
            reweight_attr_0_tensor = torch.FloatTensor([1.0 / (1.0 - base_0), 1.0 / base_0]).to(self.device)
            reweight_attr_1_tensor = torch.FloatTensor([1.0 / (1.0 - base_1), 1.0 / base_1]).to(self.device)
            self.reweight_attr_tensors[mode] = [reweight_attr_0_tensor, reweight_attr_1_tensor]

    def forward(self, mode, **_ignored):
        assert mode == 'test'
        self.model.eval()
        tqdm_iter = tqdm(range(len(self.loaders[mode])))
        data_iter = iter(self.loaders[mode])

        data = defaultdict(list)

        for _ in tqdm_iter:
            xs, ts, ss = next(data_iter)
            xs = xs.to(self.device)

            with torch.no_grad():
                logprobs, feature = self.model.inference(xs)
            
            pred = torch.max(logprobs, 1)[1].cpu().numpy()
            data['pred'].append(pred)
            if self.model_cfg.s_cls:
                data['actual'].append(ss.numpy())
            else:
                data['actual'].append(ts.numpy())
            data['ss'].append(ss.numpy())
            data['ft'].append(feature.detach().cpu().numpy())

        return data

    def _epoch(self, mode, epochID):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        tqdm_iter = tqdm(range(len(self.loaders[mode])))
        data_iter = iter(self.loaders[mode])
        metrics = defaultdict(list)

        for i in tqdm_iter:
            xs, ts, ss = next(data_iter)
            xs, ts, ss = xs.to(self.device), ts.to(self.device), ss.to(self.device)
            if self.model_cfg.s_cls:
                ts = ss
            ypreds, apreds = self.model(inputs=xs, labels=ts)

            if 'laftr' in self.model_cfg.model_key:
                apreds = apreds.flatten()
                # Compute both the prediction loss and the adversarial loss.
                loss = F.nll_loss(ypreds, ts)
                # LaftrNet uses conditional L1 distances for 4 possible cases instead.
                if self.model_cfg.mu != 0:
                    adv_loss = []
                    for k in range(self.model_cfg.num_groups):
                        for j in range(self.model_cfg.num_classes):
                            cond_indices = (ts == j) & (ss == k)
                            adv_loss.append(F.l1_loss(apreds[cond_indices], ss[cond_indices]))
                    adv_loss = torch.mean(torch.stack(adv_loss))
                else:
                    adv_loss = torch.tensor(0.0)
            
            elif 'cfair' in self.model_cfg.model_key:
                if not self.model_cfg.s_cls:
                    loss = F.nll_loss(ypreds, ts, weight=self.reweight_target_tensor[mode])
                else:
                    loss = F.nll_loss(ypreds, ts)
                adv_loss = torch.mean(torch.stack([F.nll_loss(apreds[j], ss[ts == j], weight=self.reweight_attr_tensors[mode][j])
                                              for j in range(self.model_cfg.num_classes)]))

            metrics['adv_loss'].append(adv_loss.item())
            metrics['loss'].append(loss.item())

            if mode == 'train':
                loss += self.model_cfg.mu * adv_loss
                self._backprop(loss, self.opt)

            tqdm_iter.set_description('V: {} | {} | Epoch {}| loss: {:.4f}'.format(self.exp_cfg.version, mode, epochID, loss.item()), refresh=True)

        agg_stats = self._aggregate(metrics)
        self._log(epochID, mode, agg_stats)

        return agg_stats['loss']


class CfairLaftrTrainerBuilder(object):
    """CfairLaftrTrainer Builder Class
    """

    def __init__(self):
        """CfairLaftrTrainer Builder Class Constructor
        """
        self._instance = None

    def __call__(self, data_cfg, model_cfg, exp_cfg, **_ignored):
        """Callback function
        Args:
            data_cfg (Config): Data Config object
            model_cfg (Config): Model Config object
            exp_cfg (Config): Experiment Config object
        Returns:
            CfairLaftrTrainer: Instantiated CfairLaftrTrainer  object
        """
        self._instance = CfairLaftrTrainer(data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg)
        return self._instance
