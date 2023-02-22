import os
import yaml
from os.path import join, basename, splitext


class Config(object):
    """
    Class for all attributes and functionalities related to a particular training run.
    """
    def __init__(self, cfg_file: str, params: dict):
        self.cfg_file = cfg_file
        self.__dict__.update(params)


def cfg_parser(cfg_file: str, sweep: bool = False, seed=0) -> dict:
    """
    This functions reads an input config file and instantiates objects of
    Config types.
    args:
        cfg_file (string): path to cfg file
    returns:
        data_cfg (Config)
        model_cfg (Config)
        exp_cfg (Config)
    """
    cfg = yaml.load(open(cfg_file, "r"), Loader=yaml.FullLoader)

    exp_cfg = Config(cfg_file, cfg["experiment"])
    dir_dict = dir_util(exp_cfg, seed, sweep)
    exp_cfg.__dict__.update(dir_dict)

    data_cfg = Config(cfg_file, cfg["data"])
    model_cfg = Config(cfg_file, cfg["model"])

    # set defaults for backward compatibility
    if not hasattr(model_cfg, 'pretrained_base'):
        model_cfg.pretrained_base = None

    return {"data_cfg": data_cfg, "model_cfg": model_cfg, "exp_cfg": exp_cfg}
    
def dir_util(cfg, seed, sweep):
    dirs = dict()
    dirs['output_location'] = join(cfg.output_location, splitext(basename(cfg.cfg_file))[0], str(seed))
    if sweep:
        dirs['output_location'] = join(dirs['output_location'], 'sweep')
    os.makedirs(dirs['output_location'], exist_ok=True)

    return dirs