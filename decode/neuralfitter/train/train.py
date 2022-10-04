import sys
from pathlib import Path

import argparse
import hydra
import pytorch_lightning as pl

from . import setup_cfg
from decode.neuralfitter import model
from decode.neuralfitter.data import datamodel


def train(cfg):
    exp_train, exp_val = setup_cfg.setup_sampler(cfg)

    dm = datamodel.DataModel(
        experiment_train=exp_train,
        experiment_val=exp_val,
        num_workers=cfg.Hardware.cpu.worker,
    )
    proc = setup_cfg.setup_processor(cfg)
    backbone = setup_cfg.setup_model(cfg)
    loss = setup_cfg.setup_loss(cfg)
    evaluator = None
    optimizer = setup_cfg.setup_optimizer(backbone, cfg)
    m = model.Model(model=backbone, optimizer=optimizer, proc=proc, loss=loss, evaluator=evaluator)

    trainer = pl.Trainer(reload_dataloaders_every_n_epochs=1)
    trainer.fit(model=m, datamodule=dm)


def parse_args():
    parser = argparse.ArgumentParser(description="Training Arguments")

    parser.add_argument(
        "-c",
        "--config",
        help="Specify your parameter file (.yaml).",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--overwrites",
        default=None,
        help="Specify arbitrary overwrites to config file.",
        required=False,
    )

    parser.add_argument(
        "-n",
        "--no_log",
        default=False,
        action="store_true",
        help="Set no log if you do not want to log the current run.",
    )

    # ToDo: Implementation
    # parser.add_argument(
    #     "-d",
    #     "--debug",
    #     default=False,
    #     action="store_true",
    #     help="Debug the specified parameter file. Will reduce ds size for example.",
    # )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # hack hydra and fake sys args
    argv = sys.argv
    sys.argv = [argv[0], argv[-1].strip("--overwrites=")]
    #argv = [a for a in sys.argv if (c not in a for c in {"-c=", "--config="})]

    config_file = Path(args.config)
    config_dir = config_file.parent
    train_wrapped = hydra.main(config_path=config_dir, config_name=config_file.stem)(train)
    train_wrapped()
