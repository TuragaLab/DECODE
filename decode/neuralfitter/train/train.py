import argparse
import copy
import logging
import structlog
import socket
import sys
from datetime import datetime
from pathlib import Path

import hydra
import omegaconf
import pytorch_lightning as pl
import torch

import decode
from decode.neuralfitter import model
from decode.neuralfitter.data import datamodel
from decode.neuralfitter.train import setup_cfg
from decode.utils import param_auto, system, hardware


logger = structlog.get_logger()


def log(cfg_raw, cfg_filled):
    logger.info("DECODE", version=decode.__version__)
    logger.info("System", system=system.collect_system())
    logger.info("Hardware", hardware=hardware.collect_hardware())
    logger.info("Input cfg", cfg=cfg_raw)
    logger.info("Resulting cfg", cfg=cfg_filled)


def train(cfg: omegaconf.DictConfig):
    cfg_backup = copy.copy(cfg)
    auto_cfg = param_auto.AutoConfig(
        fill=True, fill_test=True, auto_scale=True, return_type=omegaconf.DictConfig
    )
    cfg = omegaconf.OmegaConf.to_container(cfg)
    cfg = auto_cfg.parse(cfg)

    # setup experiment paths and dump backups
    path_exp = Path(cfg.Paths.experiment)
    exp_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + socket.gethostname()
    path_exp = path_exp / exp_id
    path_exp.mkdir(exist_ok=False)

    path_cfg_in = path_exp / "param_run_in.yaml"
    path_cfg_run = path_exp / "param_run.yaml"

    omegaconf.OmegaConf.save(cfg_backup, path_cfg_in)
    omegaconf.OmegaConf.save(cfg, path_cfg_run)

    # setup logging
    path_log = path_exp / "training.log"
    fh = logging.FileHandler(filename=path_log)
    root_logger = logging.getLogger()
    root_logger.addHandler(fh)
    log(cfg_backup, cfg)

    if (st := cfg.Computing.multiprocessing.sharing_strategy) is not None:
        torch.multiprocessing.set_sharing_strategy(st)
    if (sm := cfg.Computing.multiprocessing.start_method) is not None:
        torch.multiprocessing.set_start_method(sm)

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
    evaluator = setup_cfg.setup_evaluator(cfg)
    optimizer = setup_cfg.setup_optimizer(backbone, cfg)
    lr_sched = setup_cfg.setup_scheduler(optimizer, cfg)
    m = model.Model(
        model=backbone,
        optimizer=optimizer,
        lr_sched=lr_sched,
        proc=proc,
        loss=loss,
        evaluator=evaluator,
        batch_size=cfg.Trainer.batch_size,
    )
    m_ckpt = pl.callbacks.ModelCheckpoint(dirpath=path_exp)
    logger_training = setup_cfg.setup_logger(cfg)

    trainer = pl.Trainer(
        default_root_dir=cfg.Paths.experiment,
        accelerator="gpu" if "cuda" in cfg.Hardware.device.lightning else "cpu",
        devices=[int(cfg.Hardware.device.lightning.lstrip("cuda:"))] if "cuda" in cfg.Hardware.device.lightning else None,
        precision=cfg.Computing.precision,
        reload_dataloaders_every_n_epochs=1,
        logger=logger_training,
        max_epochs=cfg.Trainer.max_epochs,
        # gradient_clip_val=0.03,
        # gradient_clip_algorithm="norm",
        callbacks=[m_ckpt, pl.callbacks.LearningRateMonitor("epoch")],
        num_sanity_val_steps=0,
    )
    trainer.fit(
        model=m,
        datamodule=dm,
        ckpt_path=cfg.Paths.checkpoint_init,  # resumes training if not None
    )


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
    structlog.configure(
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    args = parse_args()

    # hack hydra and fake sys args
    argv = [sys.argv[0]]
    if args.overwrites is not None:
        argv.extend(args.overwrites.split(" "))
    sys.argv = argv

    config_file = Path(args.config)
    config_dir = config_file.parent

    if not config_file.is_file():
        raise FileNotFoundError(
            f"Config file does not exist at {config_file.absolute()}"
        )

    train_wrapped = hydra.main(config_path=config_dir, config_name=config_file.stem)(
        train
    )
    train_wrapped()
