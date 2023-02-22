import wandb
import torch
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from abc import abstractmethod, ABC
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from data import data
from model import model
from visualization import wandb_utils
from fairness import metrics as f_metrics
from utils.utils import save_model, copy_state_dict


class BaseTrainer(ABC):
    '''
    Abstract base trainer with methods that are mostly shared between different
    training algorithms. All trainer must inherit this class.
    '''
    def __init__(self, data_cfg, model_cfg, exp_cfg):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.exp_cfg = exp_cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._setup_dataloaders()
        self._setup_model()
        self._setup_optimizers()

        self.val_loss = np.inf
        self.last_best = -1

    def _setup_dataloaders(self):
        print("Loading data ..")
        self.loaders = dict()
        data_args = self.data_cfg.args
        for mode in self.model_cfg.modes:
            data_args.update({'path': data_args[mode]})
            data_args.update({'mode': mode})
            dataset = data.factory.create(self.data_cfg.data_key, **data_args)
            self.loaders[mode] = DataLoader(
                dataset,
                batch_size=data_args['batch_size'],
                shuffle=True,
                drop_last=True,
                pin_memory=False,
                num_workers=self.data_cfg.num_workers
            )

    def _setup_model(self):
        self.model = model.factory.create(self.model_cfg.model_key, **{"model_cfg": self.model_cfg}).to(self.device)
        if self.exp_cfg.load is not None:
            saved_model = torch.load(self.exp_cfg.load, map_location=self.device)
            copy_state_dict(self.model.state_dict(), saved_model['state_dict'])
        if self.exp_cfg.wandb:
            wandb.watch(self.model)

    @abstractmethod
    def _setup_optimizers(self):
        pass

    def _setup_schedulers(self):
        scheduler = self.model_cfg.scheduler
        if scheduler is not None:
            self.scheduler = eval("optim.lr_scheduler.{}(self.opt, **{})".format([*scheduler.keys()][0],
                                                                                 [*scheduler.values()][0]))

    def _backprop(self, loss, opt):
        opt.zero_grad()
        loss.backward()
        opt.step()

    def _aggregate(self, losses):
        for key in losses:
            losses[key] = np.mean(losses[key])
        return losses

    def _log(self, epochID, mode, metrics):
        if self.exp_cfg.wandb:
            wandb_utils.log_epoch_summary(epochID, mode, metrics)
    
    def _metrics(self, pred, actual, ss, protected, unprotected):
        metrics = dict()
        metrics['dp'] = f_metrics.demographic_parity(pred, ss, protected, unprotected)
        metrics['eq_odds_1'] = f_metrics.equality_odds_1(pred, actual, ss, protected, unprotected)
        metrics['eq_odds_0'] = f_metrics.equality_odds_0(pred, actual, ss, protected, unprotected)
        metrics['eq_odds'] = f_metrics.equality_odds(pred, actual, ss, protected, unprotected)
        metrics['up_acc'] = f_metrics.unprotected_accuracy(pred, actual, ss, unprotected)
        metrics['p_acc'] = f_metrics.protected_accuracy(pred, actual, ss, protected)
        metrics['acc-score'] = accuracy_score(actual, pred)
        metrics['del_dp'] = f_metrics.del_DP(pred, ss, protected, unprotected)
        metrics['del_eo'] = f_metrics.del_EO(pred, actual, ss, protected, unprotected)

        return metrics
    
    @abstractmethod
    def _epoch(self, mode, epochID):
        pass

    def train(self):
        exit = False
        for epochID in range(self.model_cfg.epochs):
            for mode in self.model_cfg.modes:
                if mode in ['train', 'val']:
                    loss = self._epoch(mode=mode, epochID=epochID)
                else:
                    # self._tsne(mode)
                    self._inference(mode)
                if mode == 'val':
                    exit = self.save(epochID=epochID, loss=loss)
            if exit:
                print("No improvement in the last 20 epochs. EARLY STOPPING.")
                break

    @abstractmethod
    def forward(self, mode:str, return_ft:bool=False):
        pass

    def _inference(self, mode):
        data = self.forward(mode)

        pred = np.concatenate(data['pred'])
        actual = np.concatenate(data['actual'])
        ss = np.concatenate(data['ss'])

        metrics = self._metrics(pred, actual, ss, 0, 1)
        for k in metrics:
            print('{}: {:.4f}'.format(k, metrics[k]))

        return metrics

    def _tsne(self, mode:str, title:str):
        sns.set(rc={'figure.figsize':(5,5)})
        sns.set_context("paper", font_scale=2.5)
        sns.set_style("white")
        data = self.forward(mode, return_ft=True)

        features = np.concatenate(data['ft'])
        ts = np.concatenate(data['ts'])
        ss = np.concatenate(data['ss'])

        x_embed = TSNE(n_components=2).fit_transform(features)
        plot = sns.scatterplot(
                x=x_embed[:, 0], y=x_embed[:, 1], 
                hue=ss, 
                palette=sns.color_palette("hls", 2),
                legend=False
            )
        plot.set(xticklabels=[])
        plot.set(yticklabels=[])
        plot.figure.savefig('sensitive_tsne_{}.png'.format(self.exp_cfg.version.replace('/','-')), bbox_inches='tight')
        plot.figure.clf()

        plot = sns.scatterplot(
                    x=x_embed[:, 0], y=x_embed[:, 1], 
                    hue=ts, 
                    palette=sns.color_palette("hls", 2),
                    legend=False
                )
        plot.set(title = title)
        plot.set(xticklabels=[])
        plot.set(yticklabels=[])
        plot.figure.savefig('target_tsne_{}.png'.format(self.exp_cfg.version.replace('/','-')), bbox_inches='tight')
        plot.figure.clf()

    def save(self, epochID, loss):
        save = False
        exit = False
        if loss < self.val_loss:
            self.val_loss = loss
            save = True
            self.last_best = epochID
            ckpt_type = 'best'
        elif epochID - self.last_best > 20:
            exit = True
        elif epochID > 0 and epochID % 5 == 0:
            save = True
            ckpt_type = 'last'
        if save:
            save_model(self.model, self.opt, epochID, loss, self.exp_cfg.output_location, ckpt_type)
        return exit

    def eval(self, model_pth):
        saved_model = torch.load(model_pth, map_location=self.device)
        copy_state_dict(self.model.state_dict(), saved_model['state_dict'])
        return self._inference('test')
