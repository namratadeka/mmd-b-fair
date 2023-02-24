import os
import yaml
import argparse
import pandas as pd
from os.path import join, splitext, exists, basename, dirname

from trainer import trainer
from config.config import cfg_parser
from utils.utils import seed_everything

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate MMD-B-Fair", allow_abbrev=False
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        required=True,
        help="name of the test config file to use"
    )
    parser.add_argument(
        "-p",
        "--paths",
        type=str,
        default=None,
        help="name of the file containing all the model paths"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID"
    )
    (args, unknown_args) = parser.parse_known_args()
    seed_everything(seed=0, harsh=False)

    # Uncomment to provide GPU ID as input argument:
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = cfg_parser(join("config", args.version), False, 0)
    cfg["exp_cfg"].version = splitext(args.version)[0]
    cfg["exp_cfg"].run_name = cfg["exp_cfg"].version
    cfg["exp_cfg"].wandb = False

    os.makedirs('./outputs', exist_ok=True)
    outpath = join('./outputs', basename(args.version).replace('.yml', '.csv'))

    pipeline = trainer.factory.create(cfg["model_cfg"].trainer_key, **cfg)
    if args.paths is not None:
        paths = yaml.load(open(args.paths, "r"), Loader=yaml.FullLoader)
        for lmda in paths['lambdas']:
            for seed in paths['lambdas'][lmda]:
                model_pth = paths['lambdas'][lmda][seed]
                print(model_pth)
                metrics = pipeline.eval(model_pth)
                metrics['lambda'] = lmda
                metrics['seed'] = seed
                df = pd.DataFrame([metrics])
                df.to_csv(outpath, mode='a', header=not exists(outpath))
    else:
        result_dir = dirname(dirname(cfg['exp_cfg'].output_location))
        lambdas = sorted([x.split('_')[-1] for x in os.listdir(result_dir)])
        for lmda in lambdas:
            if lmda != 'test':
                model_dir = join(result_dir, f'lambda_{lmda}')
                seeds = sorted(os.listdir(model_dir))
                for seed in seeds:
                    model_pth = join(model_dir, seed, 'best.pth')
                    print(model_pth)
                    if exists(model_pth):
                        metrics = pipeline.eval(model_pth)
                        metrics['lambda'] = float(lmda)
                        metrics['seed'] = seed
                        df = pd.DataFrame([metrics])
                        df.to_csv(outpath, mode='a', header=not exists(outpath))
