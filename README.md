# MMD-B-Fair: Learning Fair Representations with Statistical Testing

Official implementations of [MMD-B-Fair](https://arxiv.org/abs/2211.07907) published in [AISTATS 2023](http://aistats.org/aistats2023/index.html) by [Namrata Deka](https://namratadeka.github.io/) and [Danica J. Sutherland](https://djsutherland.ml/).

## Environment
- Python 3.8.10
- PyTorch 1.12.1
- Torchvision 0.13.1
- Wandb 0.13.3

## Datasets
- [UCI Adult](https://archive.ics.uci.edu/ml/datasets/adult)
- [COMPAS](https://foreverdata.org/1015/index.html)
- [Heritage Health](https://foreverdata.org/1015/index.html)

## Training
1. Create a .yml config file and store it in `config` (see available examples, VERY IMPORTANT). The file should specify all data, model and trainer settings/hyperparameters.

2. Then execute ```python main.py -v <relative-path-to-yml-from-config> --seed <seed> -w```
    
    Example: ```python main.py -v eq/adult/lambda_1.yml --seed 42 -w```

If you do not wish to sync to wandb while training add the option ```-m offline``` and sync anytime later with the `wandb sync` command.

If seed is not specified it will default to `0`.

Trained models are saved in the location specified in `experiment.output_location` in a subfolder named as per the seed. In wandb, experiments are logged under `<config file name>/<seed>` in the `mmd-b-fair` workspace.

## Adding more datasets/models/trainers
This repository heavily uses the factory design pattern for increased modularity. To add new datasets, models and/or trainers follow the steps below:

1. Create new data/model/trainer class under the appropriate directories. All trainer classes must inherit `BaseTrainer` and models must inherit `BaseModel`.
2. Create corresponding builders for new classes.
3. Register all builder objects to the respective factories in `data/data.py`, `model/model.py` and `trainer/trainer.py`.
4. `data.data_key`, `model.model_key` and `model.trainer_key` in the config files must match the registered factory keys.
5. Specify class-specific arguments in the config file. Example, dataset arguements must go in `data.args`.
