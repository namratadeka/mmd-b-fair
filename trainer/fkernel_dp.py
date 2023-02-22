import numpy as np
from tqdm import tqdm
from collections import defaultdict, OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import data
from trainer.base import BaseTrainer
from visualization import wandb_utils
from utils.mmd import MMDu, block_approx_test_power, strong_test


class FairKernelDPTrainer(BaseTrainer):
    def __init__(self, data_cfg, model_cfg, exp_cfg):
        super(FairKernelDPTrainer, self).__init__(
            data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg
        )
        self._setup_schedulers()
        self.normal_dist = torch.distributions.normal.Normal(loc=0., scale=1.)
        self.threshold = self.normal_dist.icdf(torch.tensor(1 - self.model_cfg.alpha)).to(self.device)

    def _setup_dataloaders(self):
        print("Loading data ..")
        self.datasets = dict()
        data_args = self.data_cfg.args
        for mode in self.model_cfg.modes:
            self.datasets[mode] = defaultdict(list)
            data_args.update({'loc': data_args[mode]})
            data_args.update({'mode': mode})
            for k, factor in enumerate(data_args['factors']):
                data_args.update({'factor': factor})
                for i in range(len(data_args['factor_values'][k])):
                    data_args.update({'factor_value': data_args['factor_values'][k][i]})
                    data_args.update({'value_fraction': data_args['value_fractions'][k][i]})
                    self.datasets[mode][factor].append((data.factory.create(self.data_cfg.data_key, **data_args), data_args['batch_size'], data_args['factor_values'][k][i]))

        self.dataloaders = self.create_dataloaders()

    def create_dataloaders(self):
        print("Creating data loaders ..")
        dataloaders = dict()

        for mode in self.datasets.keys():
            dataloaders[mode] = defaultdict(OrderedDict)

            shuffle = True
            drop_last = True

            for factor in self.datasets[mode]:
                for dataset in self.datasets[mode][factor]:
                    dataloader = DataLoader(
                        dataset=dataset[0],
                        batch_size=dataset[1],
                        shuffle=shuffle,
                        drop_last=drop_last,
                        pin_memory=False,
                        num_workers=self.data_cfg.num_workers,
                    )

                    dataloaders[mode][factor][dataset[2]] = dataloader

        return dataloaders

    def _get_iterators(self, mode):
        tqdm_iters = defaultdict()
        tqdm_iter = []
        data_iters = {}
        for factor in self.dataloaders[mode]:
            data_iters[factor] = dict()
            loader_keys = [*self.dataloaders[mode][factor].keys()]
            for key in loader_keys:
                data_iters[factor][key] = iter(self.dataloaders[mode][factor][key])

            max_loader_len = np.max([len(self.dataloaders[mode][factor][i]) for i in loader_keys])
            tqdm_iters[factor] = tqdm(range(max_loader_len), dynamic_ncols=True)
            if len(tqdm_iters[factor]) > len(tqdm_iter):
                tqdm_iter = tqdm_iters[factor]

        return data_iters, tqdm_iter

    def _setup_optimizers(self):
        params = list(self.model.parameters())
        optim_cfg = self.model_cfg.optimizers["ft"]

        self.opt =  eval(
            "optim.{}(params, **{})".format([*optim_cfg.keys()][0], [*optim_cfg.values()][0])
        )

    def _setup_schedulers(self):
        scheduler = self.model_cfg.scheduler
        if scheduler is not None:
            self.scheduler = eval("optim.lr_scheduler.{}(self.opt, **{})".format([*scheduler.keys()][0],
                                                                             [*scheduler.values()][0]))
    
    def forward(self, mode, return_ft):
        pass

    def _epoch(self, epochID, mode, inference=False):
        if mode.__eq__('train'):
            self.model.train()
        else:
            self.model.eval()

        data_iters, tqdm_iter = self._get_iterators(mode)

        stats = {}
        samples_per_set = {}
        n_blocks = {}
        block_size = {}
        for factor in self.data_cfg.args['factors']:
            stats[factor] = defaultdict(list)
            samples_per_set[factor] = int(len(self.datasets[mode][factor][0][0]) / self.data_cfg.args['batch_size']) * self.data_cfg.args['batch_size']
            n_blocks[factor] = int(np.sqrt(samples_per_set[factor]))
            block_size[factor] = samples_per_set[factor] // n_blocks[factor]
        pred_hypothesis = defaultdict(list)
        loss = []

        for i in tqdm_iter:
            objective = {}
            for factor in self.data_cfg.args['factors']:
                batch_data = list()
                for key in data_iters[factor]:
                    try:
                        batch = next(data_iters[factor][key])
                    except StopIteration:
                        data_iters[factor][key] = iter(self.dataloaders[mode][factor][key])
                        batch = next(data_iters[factor][key])

                    batch_data.append(batch['data'].to(self.device))
                n_batch = batch_data[0].shape[0]

                X = torch.cat(batch_data)
                if mode.__eq__('train'):
                    model_out = self.model(X)
                else:
                    with torch.no_grad():
                        model_out = self.model(X)

                sigma_phi = self.model.sigma_phi ** 2
                samples = X.view(model_out.shape[0], -1)
                
                mmd_value, mmd_var, _ = MMDu(features=model_out,
                                            n_samples=n_batch, n_population=block_size[factor], 
                                            images=samples, 
                                            sigma_phi=sigma_phi.unsqueeze(-1).unsqueeze(-1),
                                            kernel=self.model_cfg.kernel,
                                            is_smooth=False,
                                            unbiased_variance=self.model_cfg.unbiased_variance)
                mmd_value = mmd_value[0]
                mmd_var = mmd_var[0]
                mmd_std = torch.sqrt(torch.max(mmd_var, torch.tensor(self.model_cfg.var_offset)))

                h = strong_test(features=model_out, n=n_batch, m=samples_per_set[factor], images=samples,
                                sigma_phi=sigma_phi.unsqueeze(-1).unsqueeze(-1),
                                unbiased_variance=self.model_cfg.unbiased_variance,
                                n_per=self.model_cfg.n_permutations,
                                alpha=self.model_cfg.alpha)
                pred_hypothesis[factor].append(h)

                power, cdf_arg = block_approx_test_power(mmd_value, mmd_std, n_blocks[factor], self.threshold, self.normal_dist)
                objective[factor] = -self.model_cfg.fX[factor] * power

                stats[factor][f'{factor}/Power'].append(power.item())
                stats[factor][f'{factor}/MMDu'].append(mmd_value.item())
                stats[factor][f'{factor}/mmd_variance'].append(mmd_var.item())
                stats[factor][f'{factor}/mmd_std'].append(mmd_std.item())
                stats[factor][f'{factor}/sigma_phi'].append(self.model.sigma_phi.item())
                stats[factor][f'{factor}/cdf_arg'].append(cdf_arg.item())

            sensitive_objs, target_objs = [], []
            for factor in objective:
                if self.model_cfg.fX[factor] == -1:
                    sensitive_objs.append(objective[factor])
                else:
                    target_objs.append(objective[factor])
            sensitive_objs = [torch.tensor(0.)] if len(sensitive_objs) == 0 else sensitive_objs
            target_objs = [torch.tensor(0.)] if len(target_objs) == 0 else target_objs
            total_loss = self.model_cfg.lamda_s * torch.stack(sensitive_objs).mean() \
                       + self.model_cfg.lamda_t * torch.stack(target_objs).mean()
            loss.append(total_loss.item())
            if mode.__eq__('train') and epochID > 0:
                self._backprop(total_loss, self.opt)

            tqdm_iter.set_description('V: {} | {} | Epoch {}'.format(self.exp_cfg.version, mode, epochID), refresh=True)

        if self.exp_cfg.wandb:
            wandb_utils.log_epoch_stats(stats, epochID, mode)

        emp_powers = {}
        for factor in self.data_cfg.args['factors']:
            emp_power = np.mean(pred_hypothesis[factor])
            emp_powers[factor] = emp_power
            print(f'\nEmpirical Power ({factor}): {emp_power}')
            self._log(epochID, mode, {f'Empirical Power ({factor})': emp_power})

            agg_stats = self._aggregate(stats[factor])
            self._log(epochID, mode, agg_stats)

        loss = np.mean(loss)
        self._log(epochID, mode, {'loss': loss})
        if inference:
            return emp_powers
        else:
            return loss

    def _inference(self, mode):
        emp_powers = self._epoch(mode, inference=True)
        metrics = {}
        for factor in emp_powers:
            if self.model_cfg.fX[factor] == -1:
                metrics['sensitive-power'] = emp_powers[factor]
            elif self.model_cfg.fX[factor] == 1:
                metrics['target-power'] = emp_powers[factor]
        return metrics

class FairKernelDPTrainerBuilder(object):
    """FairKernelDP Trainer Builder Class
    """

    def __init__(self):
        """FairKernelDP Trainer Builder Class Constructor
        """
        self._instance = None

    def __call__(self, data_cfg, model_cfg, exp_cfg, **_ignored):
        """Callback function
        Args:
            data_cfg (Config): Data Config object
            model_cfg (Config): Model Config object
            exp_cfg (Config): Experiment Config object
        Returns:
            FairKernelDPTrainer: Instantiated FairKernelDP trainer object
        """
        self._instance = FairKernelDPTrainer(data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg)
        return self._instance
