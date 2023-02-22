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


class FRepDPTrainer(BaseTrainer):
    def __init__(self, data_cfg, model_cfg, exp_cfg):
        super(FRepDPTrainer, self).__init__(
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
        ft_params = list(self.model.featurizer.parameters()) \
                    + list(self.model.fc.parameters()) \
                    + list(self.model.classifier.parameters())
        ft_optim_cfg = self.model_cfg.optimizers["ft"]
        self.opt =  eval(
            "optim.{}(ft_params, **{})".format([*ft_optim_cfg.keys()][0], [*ft_optim_cfg.values()][0])
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
        data_iters, tqdm_iter = self._get_iterators(mode)

        stats = {}
        sample_sizes = {}
        for factor in self.data_cfg.args['factors']:
            stats[factor] = defaultdict(list)
            samples_per_set = int(len(self.datasets[mode][factor][0][0]) / self.data_cfg.args['batch_size']) * self.data_cfg.args['batch_size']
            n_blocks = int(np.sqrt(samples_per_set))
            block_size = samples_per_set // n_blocks

            sample_sizes[factor] = {
                'n_blocks': n_blocks,
                'block_size': block_size,
                'samples_per_set': samples_per_set
            }

        pred_hypothesis = defaultdict(list)
        loss = []
        for i in tqdm_iter:
            kernel_stats = {}
            sensitive_objs, target_objs = [], []
            features, images = {}, {}

            for k, factor in enumerate(self.data_cfg.args['factors']):
                batch_data = list()
                target_label = list()
                s_label = list()
                kernel_stats[factor] = {}
                for key in data_iters[factor]:
                    try:
                        batch = next(data_iters[factor][key])
                    except StopIteration:
                        data_iters[factor][key] = iter(self.dataloaders[mode][factor][key])
                        batch = next(data_iters[factor][key])

                    batch_data.append(batch['data'])
                    target_label.append(batch['target'])
                    s_label.append(batch['sensitive'])
                n_batch = batch_data[0].shape[0]
                target_label = torch.cat(target_label)
                s_label = torch.cat(s_label)
    
                X = torch.cat(batch_data).to(self.device)
                is_target = self.model_cfg.fX[factor] == 1
                if mode.__eq__('train'):
                    ft, logprobs = self.model(X)
                else:
                    with torch.no_grad():
                        ft, logprobs = self.model(X)
                if is_target:
                    cls_loss = F.nll_loss(logprobs, target_label.to(self.device))
                else:
                    # In DP-setting we assume target labels are available only for S_t0 and S_t1.
                    cls_loss = torch.tensor(0.0)

                sigmas = self.sigmas[:, k] ** 2
                sigmas = sigmas.unsqueeze(-1).unsqueeze(-1)
                samples = X.view(ft.shape[0], -1)
                features[factor] = ft
                images[factor] = samples
                mmd_value, mmd_var, _ = MMDu(features=ft, 
                                            n_samples=n_batch, 
                                            n_population=sample_sizes[factor]['block_size'], 
                                            images=samples, sigma_phi=sigmas, 
                                            kernel=self.model_cfg.kernel,
                                            is_smooth=False,
                                            unbiased_variance=self.model_cfg.unbiased_variance)
                
                mmd_std = torch.sqrt(torch.max(mmd_var,torch.tensor(self.model_cfg.var_offset)))
                power, cdf_arg = block_approx_test_power(mmd_value, mmd_std, sample_sizes[factor]['n_blocks'], 
                                                         self.threshold, self.normal_dist)
                obj = -self.model_cfg.fX[factor] * power
                if self.model_cfg.fX[factor] == -1:
                    sensitive_objs.append(obj)
                else:
                    target_objs.append(obj)

                kernel_stats[factor]['power'] = power
                kernel_stats[factor]['cdf_arg'] = cdf_arg
                kernel_stats[factor]['mmd_value'] = mmd_value
                kernel_stats[factor]['mmd_std'] = mmd_std
                kernel_stats[factor]['mmd_variance'] = mmd_var
                kernel_stats[factor]['cls_loss'] = cls_loss

            rho_target = torch.stack(target_objs, axis=1).mean(axis=1)
            rho_sensitive = torch.stack(sensitive_objs, axis=1).mean(axis=1)
            sigma_idx_target = rho_target.argmin().item()
            sigma_idx_sensitive = rho_sensitive.argmax().item()
            total_loss = self.model_cfg.lamda_t * rho_target.min() \
                       + self.model_cfg.lamda_s * rho_sensitive.max()

            for k, factor in enumerate(self.data_cfg.args['factors']):
                total_loss += self.model_cfg.cls_coeff * kernel_stats[factor]['cls_loss']
                if self.model_cfg.fX[factor] == -1:
                    sigma_phi = self.sigmas[sigma_idx_sensitive, k].reshape(-1,)
                    sigma_idx = sigma_idx_sensitive
                else:
                    sigma_phi = self.sigmas[sigma_idx_target, k].reshape(-1,)
                    sigma_idx = sigma_idx_target
                stats[factor][f'{factor}/Power'].append(kernel_stats[factor]['power'][sigma_idx].item())
                stats[factor][f'{factor}/MMDu'].append(kernel_stats[factor]['mmd_value'][sigma_idx].item())
                stats[factor][f'{factor}/mmd_std'].append(kernel_stats[factor]['mmd_std'][sigma_idx].item())
                stats[factor][f'{factor}/mmd_variance'].append(kernel_stats[factor]['mmd_variance'][sigma_idx].item())
                stats[factor][f'{factor}/cdf_arg'].append(kernel_stats[factor]['cdf_arg'][sigma_idx].item())
                stats[factor][f'{factor}/sigma_phi'].append(sigma_phi.item())
                stats[factor][f'{factor}/cls_loss'].append(kernel_stats[factor]['cls_loss'].item())
                h = strong_test(features=features[factor],
                                n=n_batch, 
                                m=sample_sizes[factor]['samples_per_set'], 
                                images=images[factor],
                                sigma_phi=sigma_phi.unsqueeze(-1).unsqueeze(-1),
                                unbiased_variance=self.model_cfg.unbiased_variance,
                                n_per=self.model_cfg.n_permutations,
                                alpha=self.model_cfg.alpha)
                pred_hypothesis[factor].append(h)
            
            loss.append(total_loss.item())
            if mode.__eq__('train') and epochID > 0:
                self._backprop(total_loss, self.opt)

            tqdm_iter.set_description('V: {} | {} | Epoch {}'.format(self.exp_cfg.version, mode, epochID), refresh=True)

        if self.exp_cfg.wandb:
            wandb_utils.log_epoch_stats(stats, epochID, mode)

        for factor in self.data_cfg.args['factors']:
            emp_power = np.mean(pred_hypothesis[factor])
            print(f'\nEmpirical Power ({factor}): {emp_power}')
            self._log(epochID, mode, {f'Empirical Power ({factor})': emp_power})

            agg_stats = self._aggregate(stats[factor])
            self._log(epochID, mode, agg_stats)

        loss = np.mean(loss)
        self._log(epochID, mode, {'loss': loss})
        return loss


class FRepDPTrainerBuilder(object):
    """FairRepDP Trainer Builder Class
    """

    def __init__(self):
        """FairRepDP Trainer Builder Class Constructor
        """
        self._instance = None

    def __call__(self, data_cfg, model_cfg, exp_cfg, **_ignored):
        """Callback function
        Args:
            data_cfg (Config): Data Config object
            model_cfg (Config): Model Config object
            exp_cfg (Config): Experiment Config object
        Returns:
            FRepDPTrainer: Instantiated FairRepDP trainer object
        """
        self._instance = FRepDPTrainer(data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg)
        return self._instance
