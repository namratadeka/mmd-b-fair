import os
import wandb
import argparse
from os.path import join, splitext

from trainer import trainer
from config.config import cfg_parser
from utils.utils import seed_everything
from visualization.wandb_utils import init_wandb


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="mmd-b-fair", allow_abbrev=False
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="default.yml",
        help="name of the config file to use"
        )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID"
        )
    parser.add_argument(
        "-w", "--wandb", action="store_true", help="Log to wandb"
    )
    parser.add_argument(
        "-m", "--mode", type=str, default='online', help="wandb mode"
    )
    parser.add_argument(
        "-s", "--sweep", action="store_true", help="Use wandb sweep or not"
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=0,
        help="Seed to initialize random functions."
    )
    (args, unknown_args) = parser.parse_known_args()
    seed_everything(seed=args.seed, harsh=False)

    # Uncomment to provide GPU ID as input argument:
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = cfg_parser(join("config", args.version), args.sweep, args.seed)
    cfg["exp_cfg"].version = splitext(args.version)[0]
    cfg["exp_cfg"].run_name = cfg["exp_cfg"].version
    cfg["exp_cfg"].wandb = args.wandb

    for unknown_arg in unknown_args:
        t = unknown_arg.split("--")[1]
        x = t.split("=")
        exec("{} = {}".format(x[0], type(eval(x[0]))(x[1])))
    
    if args.wandb:
        init_wandb(cfg.copy(), args.seed, args.mode)

    pipeline = trainer.factory.create(cfg["model_cfg"].trainer_key, **cfg)
    pipeline.train()
    if args.wandb:
        wandb.finish()
