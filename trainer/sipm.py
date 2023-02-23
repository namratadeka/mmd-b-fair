from tqdm import tqdm
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn.functional as F

from trainer.base import BaseTrainer
from utils.utils import reset_weights


class sIPMTrainer(BaseTrainer):
    def __init__(self, data_cfg, model_cfg, exp_cfg):
        super(sIPMTrainer, self).__init__(
            data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg
        )
        if self.model_cfg.finetune:
            reset_weights(self.model.head)
    
    def _setup_optimizers(self):
        params = list(self.model.encoder.parameters()) \
               + list(self.model.encoder_fc.parameters()) \
               + list(self.model.head.parameters()) \
               + list(self.model.decoder.parameters())
        optim_cfg = self.model_cfg.optimizers["opt"]

        self.opt =  eval(
            "optim.{}(params, **{})".format([*optim_cfg.keys()][0], [*optim_cfg.values()][0])
        )

        params = list(self.model.aud_model.parameters())
        optim_cfg = self.model_cfg.optimizers["fair_opt"]
        self.fair_opt = eval(
            "optim.{}(params, **{})".format([*optim_cfg.keys()][0], [*optim_cfg.values()][0])
        )
    
    def fair_criterion(self, proj0, proj1):        
        proj0, proj1 = proj0.flatten(), proj1.flatten()
        mean0, mean1 = proj0.mean(), proj1.mean()
        loss = (mean0 - mean1).abs()
            
        return loss

    def forward(self, mode:str, **_ignored):
        self.model.eval()
        tqdm_iter = tqdm(range(len(self.loaders[mode])))
        data_iter = iter(self.loaders[mode])

        data = defaultdict(list)
        for itr in tqdm_iter:
            xs, ts, ss = next(data_iter)
            xs = xs.to(self.device)

            with torch.no_grad():
                feature, preds, _ = self.model.forward(xs)
            
            pred = (preds.flatten().detach().cpu().numpy() > 0.5).astype(int)
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

        for itr in tqdm_iter:
            xs, ts, ss = next(data_iter)
            xs, ts, ss = xs.to(self.device), ts.to(self.device), ss.to(self.device)

            if mode == 'train':
                z, preds, recon = self.model(xs)
            else:
                with torch.no_grad():
                    z, preds, recon = self.model(xs)

            # task loss
            if self.model_cfg.s_cls:
                task_loss = F.binary_cross_entropy(preds, ss.reshape(-1,1).float())
            else:
                task_loss = F.binary_cross_entropy(preds, ts.reshape(-1,1).float())

            # reconstruction loss
            recon_loss = torch.tensor(0.0)
            if self.model_cfg.lmdaR > 0.0:
                recon_loss = ((xs - recon)**2).sum(dim=1).mean()

            # fair loss
            fair_loss = torch.tensor(0.0)
            if self.model_cfg.lmdaF > 0.0:
                z0, z1 = z[ss == 0], z[ss == 1]
                aud_z0, aud_z1 = self.model.aud_model(z0), self.model.aud_model(z1)
                fair_loss = self.fair_criterion(aud_z0, aud_z1)

            # total loss
            loss = self.model_cfg.lmda*task_loss
            if self.model_cfg.lmdaF > 0.0:
                loss += self.model_cfg.lmdaF*fair_loss
            if self.model_cfg.lmdaR > 0.0:
                loss += self.model_cfg.lmdaR*recon_loss

            metrics['loss'].append(task_loss.item())
            metrics['total_loss'].append(loss.item())
            metrics['recon_loss'].append(recon_loss.item())

            if mode == 'train':
                # self.freeze_model(module='discriminator')
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()

                # train adversarial network
                if self.model_cfg.lmdaF > 0:
                    # self.freeze_model(module='encoder')
                    for k in range(self.model_cfg.aud_steps):
                        z, preds, _ = self.model(xs)
                        z0, z1 = z[ss == 0].clone().detach(), z[ss == 1].clone().detach()
                        aud_z0, aud_z1 = self.model.aud_model(z0), self.model.aud_model(z1)
                        fair_loss = -self.fair_criterion(aud_z0, aud_z1)
                        self.fair_opt.zero_grad()
                        fair_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.aud_model.parameters(), 5.0)
                        self.fair_opt.step()
            
            metrics['adv_loss'].append(fair_loss.item())

            tqdm_iter.set_description('V: {} | {} | Epoch {}| loss: {:.4f}'.format(self.exp_cfg.version, mode, epochID, loss.item()), refresh=True)

        agg_stats = self._aggregate(metrics)
        self._log(epochID, mode, agg_stats)

        return agg_stats['total_loss']


class sIPMTrainerBuilder(object):
    """sIPMTrainer  Builder Class
    """

    def __init__(self):
        """sIPMTrainer  Builder Class Constructor
        """
        self._instance = None

    def __call__(self, data_cfg, model_cfg, exp_cfg, **_ignored):
        """Callback function
        Args:
            data_cfg (Config): Data Config object
            model_cfg (Config): Model Config object
            exp_cfg (Config): Experiment Config object
        Returns:
            sIPMTrainer: Instantiated sIPMTrainer  object
        """
        self._instance = sIPMTrainer(data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg)
        return self._instance
