from tqdm import tqdm
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.mmd import MMDu
from trainer.base import BaseTrainer
from utils.utils import reset_weights


class MMDClassifier(BaseTrainer):
    def __init__(self, data_cfg, model_cfg, exp_cfg):
        super(MMDClassifier, self).__init__(
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

    def mmd_regularizer(self, xs, ss, feature):
        s0_idx = torch.where(ss == 0)[0]
        s1_idx = torch.where(ss == 1)[0]
        n = min(s0_idx.shape[0], s1_idx.shape[0])
        if n < 2:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        Fx, Fy = feature[s0_idx][:n], feature[s1_idx][:n]
        samples_x = torch.flatten(xs[s0_idx][:n], start_dim=1)
        samples_y = torch.flatten(xs[s1_idx][:n], start_dim=1)
        samples = torch.cat([samples_x, samples_y])
        mmd, mmd_var, _ = MMDu(
            features = torch.cat([Fx, Fy]),
            n_samples = n,
            n_population = n,
            images = samples,
            sigma_phi = torch.FloatTensor([0.1]).unsqueeze(-1).unsqueeze(-1).to(self.device),
            is_smooth=False,
            unbiased_variance=False
        )
        mmd_std = torch.sqrt(torch.max(mmd_var, torch.tensor(1e-6)))
        J = torch.div(mmd[0], mmd_std[0])
        return J, mmd[0], mmd_std[0]

    def _epoch(self, mode:str, epochID:int):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        tqdm_iter = tqdm(range(len(self.loaders[mode])))
        data_iter = iter(self.loaders[mode])
        metrics = defaultdict(list)

        for itr in tqdm_iter:
            xs, ts, ss = next(data_iter)
            xs, ts = xs.to(self.device), ts.to(self.device)
            if mode == 'train':
                logprobs, feature = self.model(xs, return_ft=True, finetune=self.model_cfg.finetune)
            else:
                with torch.no_grad():
                    logprobs, feature = self.model(xs, return_ft=True, finetune=self.model_cfg.finetune)
            
            cls_loss = F.nll_loss(logprobs, ts)
            if self.model_cfg.regularizer == 'dp':
                mmd_loss, mmdu, mmd_std = self.mmd_regularizer(xs, ss, feature)
            elif self.model_cfg.regularizer == 'eq':
                mmd_loss_0, mmdu_0, mmd_std_0 = self.mmd_regularizer(xs[ts==0], ss[ts==0], feature[ts==0])
                mmd_loss_1, mmdu_1, mmd_std_1 = self.mmd_regularizer(xs[ts==1], ss[ts==1], feature[ts==1])
                mmd_loss = mmd_loss_0 + mmd_loss_1
            loss = cls_loss + self.model_cfg.lamda_s*mmd_loss

            if mode == 'train':# and epochID > 0:
                self._backprop(loss, self.opt)

            metrics['loss'].append(loss.item())
            metrics['cls_loss'].append(cls_loss.item())
            metrics['mmd_loss'].append(mmd_loss.item())
            # metrics['MMDu'].append(mmdu.item())
            # metrics['mmd_std'].append(mmd_std.item())

            tqdm_iter.set_description('V: {} | {} | Epoch {} | Loss {:.4f}'.format(self.exp_cfg.version, mode, epochID, loss.item()), refresh=True)

        agg_stats = self._aggregate(metrics)
        self._log(epochID, mode, agg_stats)

        return agg_stats['loss']


class MMDClassifierBuilder(object):
    """MMDClassifier  Builder Class
    """

    def __init__(self):
        """MMDClassifier  Builder Class Constructor
        """
        self._instance = None

    def __call__(self, data_cfg, model_cfg, exp_cfg, **_ignored):
        """Callback function
        Args:
            data_cfg (Config): Data Config object
            model_cfg (Config): Model Config object
            exp_cfg (Config): Experiment Config object
        Returns:
            MMDClassifier: Instantiated MMDClassifier  object
        """
        self._instance = MMDClassifier(data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg)
        return self._instance
