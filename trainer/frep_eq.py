import numpy as np
from tqdm import tqdm
from collections import defaultdict, OrderedDict

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import data
from trainer.base import BaseTrainer
from visualization import wandb_utils
from utils.mmd import MMDu, block_approx_test_power, strong_test


class FRepEqTrainer(BaseTrainer):
    def __init__(self, data_cfg, model_cfg, exp_cfg):
        super(FRepEqTrainer, self).__init__(
            data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg
        )
        self._setup_schedulers()
        self.normal_dist = torch.distributions.normal.Normal(loc=0., scale=1.)
        self.threshold = self.normal_dist.icdf(torch.tensor(1 - self.model_cfg.alpha)).to(self.device)

        self.factors = self.data_cfg.args['factors']

        self.sigmas = torch.FloatTensor(np.linspace(model_cfg.sigma_range[0], model_cfg.sigma_range[1], 5)).to(self.device)
        sigma_list = []
        for _ in range(len(self.factors)):
            sigma_list.append(self.sigmas.unsqueeze(1))
        self.sigmas = torch.cat(sigma_list, axis=1).to(self.device)

    def _setup_dataloaders(self):
        print("Loading data ..")
        self.datasets = dict()
        data_args = self.data_cfg.args
        for mode in self.model_cfg.modes:
            self.datasets[mode] = dict()
            data_args.update({'loc': data_args[mode]})
            data_args.update({'mode': mode})
            for k, factor in enumerate(data_args['factors']):
                self.datasets[mode][factor] = defaultdict(list)
                data_args.update({'factor': factor})
                for i in range(len(data_args['factor_values'][k])):
                    factor_value = data_args['factor_values'][k][i]
                    data_args.update({'factor_value': factor_value})
                    data_args.update({'value_fraction': data_args['value_fractions'][k][i]})
                    for j in range(len(data_args['c_factor_values'][k][i])):
                        c_factor_value = data_args['c_factor_values'][k][i][j]
                        data_args.update({'c_factor_value': c_factor_value})
                        dataset = data.factory.create(self.data_cfg.data_key, **data_args)
                        self.datasets[mode][factor][factor_value].append((dataset, data_args['batch_size'], c_factor_value))

        self.dataloaders = self.create_dataloaders()

    def create_dataloaders(self):
        print("Creating data loaders ..")
        dataloaders = dict()

        for mode in self.datasets.keys():
            dataloaders[mode] = dict()

            shuffle = True
            drop_last = True

            for factor in self.datasets[mode]:
                dataloaders[mode][factor] = defaultdict(OrderedDict)
                for factor_value in self.datasets[mode][factor]:
                    for dataset in self.datasets[mode][factor][factor_value]:
                        dataloader = DataLoader(
                            dataset=dataset[0],
                            batch_size=dataset[1],
                            shuffle=shuffle,
                            drop_last=drop_last,
                            pin_memory=False,
                            num_workers=self.data_cfg.num_workers,
                        )

                        dataloaders[mode][factor][factor_value][dataset[2]] = dataloader

        return dataloaders

    def _get_iterators(self, mode):
        tqdm_iter = []
        data_iters = dict()
        data_sizes = dict()
        data_loaders = dict()
        for factor in self.dataloaders[mode]:
            data_iters[factor] = dict()
            data_sizes[factor] = dict()
            data_loaders[factor] = dict()
            pairs = []
            sizes = []
            loader_pairs = []
            factor_values = [*self.dataloaders[mode][factor]]
            num_c_factors = len(self.dataloaders[mode][factor][factor_values[0]])
            for c in range(num_c_factors):
                pair = []
                size = []
                loader_pair = []
                for f in factor_values:
                    c_factors = [*self.dataloaders[mode][factor][f].keys()]
                    loader = self.dataloaders[mode][factor][f][c_factors[c]]
                    iterator = iter(loader)
                    pair.append(iterator)
                    size.append(len(loader))
                    loader_pair.append(loader)

                    if len(loader) > len(tqdm_iter):
                        tqdm_iter = tqdm(range(len(loader)), dynamic_ncols=True)
                pairs.append(pair)
                sizes.append(tuple(size))
                loader_pairs.append(tuple(loader_pair))
            data_iters[factor] = pairs
            data_sizes[factor] = sizes
            data_loaders[factor] = loader_pairs

        return data_iters, tqdm_iter, data_sizes, data_loaders

    def _setup_optimizers(self):
        ft_params = list(self.model.featurizer.parameters()) \
                    + list(self.model.fc.parameters()) \
                    + list(self.model.classifier.parameters())
        optim_cfg = self.model_cfg.optimizers["ft"]
        self.opt =  eval(
            "optim.{}(ft_params, **{})".format([*optim_cfg.keys()][0], [*optim_cfg.values()][0])
        )
        if self.exp_cfg.load is not None:
            saved_model = torch.load(self.exp_cfg.load, map_location=self.device)
            self.opt.load_state_dict(saved_model['optimizer'])

    def forward(self, mode:str, return_ft:bool=False):
        pass

    def _epoch(self, mode, epochID):
        if mode.__eq__('train'):
            self.model.train()
        else:
            self.model.eval()

        data_iters, tqdm_iter, data_sizes, data_loaders = self._get_iterators(mode)
        n_batch = self.data_cfg.args['batch_size']
        stats = {}
        pred_hypothesis = {}
        for factor in self.data_cfg.args['factors']:
            for j in range(len(data_iters[factor])):
                stats[f'{factor}_{j}'] = defaultdict(list)
                pred_hypothesis[f'{factor}_{j}'] = []
        loss = []

        for i in tqdm_iter:
            kernel_stats = {}
            objectives = defaultdict(list)
            target_objs, sensitive_objs = [], []
            features, images = defaultdict(dict), defaultdict(dict)

            for k, factor in enumerate(self.data_cfg.args['factors']):
                for j, pair in enumerate(data_iters[factor]):
                    kernel_stats[f'{factor}_{j}'] = {}
                    samples_per_set = max(data_sizes[factor][j]) * n_batch
                    n_blocks = int(np.sqrt(samples_per_set))
                    block_size = samples_per_set // n_blocks

                    batch_data = list()
                    target_label = list()
                    s_label = list()
                    for p, data_iter in enumerate(pair):
                        try:
                            batch = next(data_iter)
                        except StopIteration:
                            pair[p] = iter(data_loaders[factor][j][p])
                            batch = next(pair[p])
                        batch_data.append(batch['data'])
                        target_label.append(batch['target'])
                        s_label.append(batch['sensitive'])
                    
                    target_label = torch.cat(target_label)
                    s_label = torch.cat(s_label)

                    X = torch.cat(batch_data).to(self.device)
                    if mode.__eq__('train'):
                        ft, logprobs = self.model(X)
                    else:
                        with torch.no_grad():
                            ft, logprobs = self.model(X)
                    cls_loss = F.nll_loss(logprobs, target_label.to(self.device))

                    sigmas = self.sigmas[:, k] ** 2
                    sigmas = sigmas.unsqueeze(-1).unsqueeze(-1)
                    samples = X.view(ft.shape[0], -1)
                    features[factor][j] = ft
                    images[factor][j] = samples
                    mmd_value, mmd_var, _ = MMDu(features=ft, 
                                                n_samples=n_batch, 
                                                n_population=block_size, 
                                                images=samples, sigma_phi=sigmas, 
                                                kernel=self.model_cfg.kernel,
                                                is_smooth=False,
                                                unbiased_variance=self.model_cfg.unbiased_variance)
                    mmd_std = torch.sqrt(torch.max(mmd_var, torch.tensor(self.model_cfg.var_offset)))
                    power, cdf_arg = block_approx_test_power(mmd_value, mmd_std, n_blocks, 
                                                            self.threshold, self.normal_dist)
                    obj = -self.model_cfg.fX[factor] * power
                    kernel_stats[f'{factor}_{j}']['power'] = power
                    kernel_stats[f'{factor}_{j}']['cdf_arg'] = cdf_arg
                    kernel_stats[f'{factor}_{j}']['mmd_value'] = mmd_value
                    kernel_stats[f'{factor}_{j}']['mmd_std'] = mmd_std
                    kernel_stats[f'{factor}_{j}']['mmd_variance'] = mmd_var
                    kernel_stats[f'{factor}_{j}']['cls_loss'] = cls_loss

                    objectives[factor].append(obj)
            
                objectives[factor] = torch.stack(objectives[factor]).sum(dim=0)
                if self.model_cfg.fX[factor] == -1:
                    sensitive_objs.append(objectives[factor])
                else:
                    target_objs.append(objectives[factor])

            sensitive_objs = [torch.FloatTensor(np.array([0])).to(self.device)] if len(sensitive_objs) == 0 else sensitive_objs
            target_objs = [torch.FloatTensor(np.array([0])).to(self.device)] if len(target_objs) == 0 else target_objs
            rho_target = torch.stack(target_objs, axis=1).mean(axis=1)
            rho_sensitive = torch.stack(sensitive_objs, axis=1).mean(axis=1)
            sigma_idx_target = rho_target.argmin().item()
            sigma_idx_sensitive = rho_sensitive.argmax().item()
            total_loss = self.model_cfg.lamda_t * rho_target.min() \
                       + self.model_cfg.lamda_s * rho_sensitive.max()

            for k, factor in enumerate(self.data_cfg.args['factors']):
                for j in range(len(data_iters[factor])):
                    total_loss += self.model_cfg.cls_coeff * kernel_stats[f'{factor}_{j}']['cls_loss']
                    if self.model_cfg.fX[factor] == -1:
                        sigma_phi = self.sigmas[sigma_idx_sensitive, k].reshape(-1,)
                        sigma_idx = sigma_idx_sensitive
                    else:
                        sigma_phi = self.sigmas[sigma_idx_target, k].reshape(-1,)
                        sigma_idx = sigma_idx_target
                    stats[f'{factor}_{j}'][f'{factor}_{j}/Power'].append(kernel_stats[f'{factor}_{j}']['power'][sigma_idx].item())
                    stats[f'{factor}_{j}'][f'{factor}_{j}/MMDu'].append(kernel_stats[f'{factor}_{j}']['mmd_value'][sigma_idx].item())
                    stats[f'{factor}_{j}'][f'{factor}_{j}/mmd_std'].append(kernel_stats[f'{factor}_{j}']['mmd_std'][sigma_idx].item())
                    stats[f'{factor}_{j}'][f'{factor}_{j}/mmd_variance'].append(kernel_stats[f'{factor}_{j}']['mmd_variance'][sigma_idx].item())
                    stats[f'{factor}_{j}'][f'{factor}_{j}/cdf_arg'].append(kernel_stats[f'{factor}_{j}']['cdf_arg'][sigma_idx].item())
                    stats[f'{factor}_{j}'][f'{factor}_{j}/sigma_phi'].append(sigma_phi.item())
                    stats[f'{factor}_{j}'][f'{factor}_{j}/cls_loss'].append(kernel_stats[f'{factor}_{j}']['cls_loss'].item())
                    h = strong_test(features=features[factor][j],
                                    n=n_batch, 
                                    m=max(data_sizes[factor][j]) * n_batch, 
                                    images=images[factor][j],
                                    sigma_phi=sigma_phi.unsqueeze(-1).unsqueeze(-1),
                                    unbiased_variance=self.model_cfg.unbiased_variance,
                                    n_per=self.model_cfg.n_permutations,
                                    alpha=self.model_cfg.alpha)
                    pred_hypothesis[f'{factor}_{j}'].append(h)

            loss.append(total_loss.item())
            if mode.__eq__('train'):
                self._backprop(total_loss, self.opt)

            tqdm_iter.set_description('V: {} | {} | Epoch {}'.format(self.exp_cfg.version, mode, epochID), refresh=True)

        if self.exp_cfg.wandb:
            wandb_utils.log_epoch_stats(stats, epochID, mode)

        for factor in stats:
            emp_power = np.mean(pred_hypothesis[factor])
            print(f'\nEmpirical Power ({factor}): {emp_power}')
            self._log(epochID, mode, {f'Empirical Power ({factor})': emp_power})

            agg_stats = self._aggregate(stats[factor])
            self._log(epochID, mode, agg_stats)

        loss = np.mean(loss)
        self._log(epochID, mode, {'loss': loss})
        return loss

class FRepEqTrainerBuilder(object):
    """FairRepEq Trainer Builder Class
    """

    def __init__(self):
        """FairRepEq Trainer Builder Class Constructor
        """
        self._instance = None

    def __call__(self, data_cfg, model_cfg, exp_cfg, **_ignored):
        """Callback function
        Args:
            data_cfg (Config): Data Config object
            model_cfg (Config): Model Config object
            exp_cfg (Config): Experiment Config object
        Returns:
            FRepEqTrainer: Instantiated FairRepEq trainer object
        """
        self._instance = FRepEqTrainer(data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg)
        return self._instance
