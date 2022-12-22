import sys
from pathlib import Path

import argparse
import hydra
import omegaconf
import pytorch_lightning as pl
import torch

from decode.neuralfitter.train import setup_cfg
from decode.neuralfitter import model
from decode.neuralfitter.data import datamodel
from decode.utils import param_auto


def train(cfg: omegaconf.DictConfig):
    auto_cfg = param_auto.AutoConfig(
        fill=True, fill_test=True, auto_scale=True, return_type=omegaconf.DictConfig
    )
    cfg = omegaconf.OmegaConf.to_container(cfg)
    cfg = auto_cfg.parse(cfg)

    if (cs := cfg.Computing.multiprocessing_sharing_strategy) is not None:
        torch.multiprocessing.set_sharing_strategy(cs)

    exp_train, exp_val = setup_cfg.setup_sampler(cfg)

    dm = datamodel.DataModel(
        experiment_train=exp_train,
        experiment_val=exp_val,
        num_workers=cfg.Hardware.cpu.worker,
        batch_size=cfg.Trainer.batch_size,
    )
    proc = setup_cfg.setup_processor(cfg)
    backbone = setup_cfg.setup_model(cfg)
    loss = setup_cfg.setup_loss(cfg)
    evaluator = None
    optimizer = setup_cfg.setup_optimizer(backbone, cfg)
    m = model.Model(
        model=backbone,
        optimizer=optimizer,
        proc=proc,
        loss=loss,
        evaluator=evaluator,
        batch_size=cfg.Trainer.batch_size,
    )
    m_ckpt = pl.callbacks.ModelCheckpoint(dirpath=cfg.Paths.experiment)
    logger = setup_cfg.setup_logger(cfg)

    trainer = pl.Trainer(
        default_root_dir=cfg.Paths.experiment,
        accelerator="gpu" if "cuda" in cfg.Hardware.device.training else "cpu",
        precision=cfg.Computing.precision,
        reload_dataloaders_every_n_epochs=1,
        logger=logger,
        max_epochs=cfg.Trainer.max_epochs,
        callbacks=[m_ckpt],
    )
    trainer.fit(
        model=m,
        datamodule=dm,
        ckpt_path=cfg.Paths.checkpoint_init,  # resumes training if not None
    )

    print("Done")


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
    argv = [sys.argv[0]]
    if args.overwrites is not None:
        argv.extend(args.overwrites.split(" "))
    sys.argv = argv

    config_file = Path(args.config)
    config_dir = config_file.parent
    train_wrapped = hydra.main(config_path=config_dir, config_name=config_file.stem)(
        train
    )
    train_wrapped()
