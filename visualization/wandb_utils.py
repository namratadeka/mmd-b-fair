import wandb
from visualization.plots import plot_features_2D, plot_stat_histograms

def init_wandb(cfg: dict, seed, mode='online') -> None:
    """Initialize project on Weights & Biases
    Args:
        cfg (dict): Configuration dictionary
    """
    for key in cfg:
        cfg[key] = cfg[key].__dict__

    wandb.init(project="deep-kernels", name=f'{cfg["exp_cfg"]["run_name"]}/{seed}', 
        notes=cfg["exp_cfg"]["description"], config=cfg, mode=mode)

def log_epoch_summary(epochID:int, mode:str, losses:dict):
    logs = {}
    for key in losses.keys():
        logs.update({"{}/{}".format(mode, key): losses[key]})

    wandb.log(logs, step=epochID)

def log_features_2D(epochID:int, mode:str, factor:str, features:list, sigma:float):
    fig = plot_features_2D(features[0], features[1], sigma)
    wandb.log({"{}_{}_features_2D".format(mode,factor): wandb.Image(fig)}, step=epochID)

def log_epoch_stats(stats:dict, epochID:int, mode:str):
    fig = plot_stat_histograms(stats)
    wandb.log({"{}_batch-histograms".format(mode): wandb.Image(fig)}, step=epochID)
