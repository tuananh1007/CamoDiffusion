#!/usr/bin/env python
#
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import argparse
import logging
import os.path as osp
from contextlib import ExitStack
from typing import MutableSequence
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import create_ddp_model, default_argument_parser, hooks, launch
from detectron2.evaluation import print_csv_format
from detectron2.utils import comm
from detectron2.utils.events import JSONWriter
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from iopath.common.s3 import S3PathHandler
from omegaconf import OmegaConf

from DiffCIS.checkpoint import DiffCISCheckpointer
from DiffCIS.config import auto_scale_workers, instantiate_diffcis
from DiffCIS.engine.defaults import default_setup, get_dataset_from_loader, get_model_from_module
from DiffCIS.engine.hooks import EvalHook
from DiffCIS.engine.train_loop import AMPTrainer, SimpleTrainer
from DiffCIS.evaluation import inference_on_dataset
from DiffCIS.utils.events import CommonMetricPrinter, WandbWriter, WriterStack
from DiffCIS.data.datasets.register_cis import register_all_cod10k_instance

PathManager.register_handler(S3PathHandler())

logger = logging.getLogger("DiffCIS")


def default_writers(cfg):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    if "log_dir" in cfg.train:
        log_dir = cfg.train.log_dir
    else:
        log_dir = cfg.train.output_dir
    PathManager.mkdirs(log_dir)
    ret = [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(
            cfg.train.max_iter, run_name=osp.join(cfg.train.run_name, cfg.train.run_tag)
        ),
        JSONWriter(osp.join(log_dir, "metrics.json")),
    ]
    if cfg.train.wandb.enable_writer:
        ret.append(
            WandbWriter(
                max_iter=cfg.train.max_iter,
                run_name=osp.join(cfg.train.run_name, cfg.train.run_tag),
                output_dir=log_dir,
                project=cfg.train.wandb.project,
                config=OmegaConf.to_container(cfg, resolve=False),
                resume=cfg.train.wandb.resume,
            )
        )

    return ret


class InferenceRunner:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

    def __call__(self, final_iter=False, next_iter=0):
        return do_test(self.cfg, self.model, final_iter=final_iter, next_iter=next_iter)


def do_test(cfg, model, *, final_iter=False, next_iter=0):
    all_ret = dict()
    # make a copy incase we modify it every time calling do_test
    cfg = OmegaConf.create(cfg)

    # BC for detectron
    if "evaluator" in cfg.dataloader and "test" in cfg.dataloader:
        task_final_iter_only = cfg.dataloader.get("final_iter_only", False)
        task_eval_period = cfg.dataloader.get("eval_period", 1)
        if not final_iter and (task_final_iter_only or next_iter % task_eval_period != 0):
            logger.info(
                f"Skip test set evaluation at iter {next_iter}, "
                f"since task_final_iter_only={task_final_iter_only}, "
                f"next_iter {next_iter} % task_eval_period {task_eval_period}"
                f"={next_iter % task_eval_period} != 0"
            )
        else:
            loader = instantiate(cfg.dataloader.test)

            if "wrapper" in cfg.dataloader:
                wrapper_cfg = cfg.dataloader.wrapper
                # look for the last wrapper
                while "model" in wrapper_cfg:
                    wrapper_cfg = wrapper_cfg.model
                wrapper_cfg.model = get_model_from_module(model)
                # poping _with_dataset_
                if wrapper_cfg.pop("_with_dataset_", False):
                    wrapper_cfg.dataset = get_dataset_from_loader(loader)
                inference_model = create_ddp_model(instantiate(cfg.dataloader.wrapper))
            else:
                inference_model = model

            # poping _with_dataset_
            if isinstance(cfg.dataloader.evaluator, MutableSequence):
                for i in range(len(cfg.dataloader.evaluator)):
                    if cfg.dataloader.evaluator[i].pop("_with_dataset_", False):
                        cfg.dataloader.evaluator[i].dataset = get_dataset_from_loader(loader)
            else:
                if cfg.dataloader.evaluator.pop("_with_dataset_", False):
                    cfg.dataloader.evaluator.dataset = get_dataset_from_loader(loader)

            ret = inference_on_dataset(
                inference_model,
                loader,
                instantiate(cfg.dataloader.evaluator),
                use_amp=cfg.train.amp.enabled,
            )
            print_csv_format(ret)
            all_ret.update(ret)
    if "extra_task" in cfg.dataloader:
        for task in cfg.dataloader.extra_task:
            task_final_iter_only = cfg.dataloader.extra_task[task].get("final_iter_only", False)
            task_eval_period = cfg.dataloader.extra_task[task].get("eval_period", 1)
            if not final_iter and (task_final_iter_only or next_iter % task_eval_period != 0):
                logger.info(
                    f"Skip {task} evaluation at iter {next_iter}, "
                    f"since task_final_iter_only={task_final_iter_only}, "
                    f"next_iter {next_iter} % task_eval_period {task_eval_period}"
                    f"={next_iter % task_eval_period} != 0"
                )
                continue

            logger.info(f"Evaluating extra task: {task}")
            loader = instantiate(cfg.dataloader.extra_task[task].loader)

            # poping _with_dataset_
            if isinstance(cfg.dataloader.extra_task[task].evaluator, MutableSequence):
                for i in range(len(cfg.dataloader.extra_task[task].evaluator)):
                    if cfg.dataloader.extra_task[task].evaluator[i].pop("_with_dataset_", False):
                        cfg.dataloader.extra_task[task].evaluator[
                            i
                        ].dataset = get_dataset_from_loader(loader)
            else:
                if cfg.dataloader.extra_task[task].evaluator.pop("_with_dataset_", False):
                    cfg.dataloader.extra_task[task].evaluator.dataset = get_dataset_from_loader(
                        loader
                    )

            if "wrapper" in cfg.dataloader.extra_task[task]:
                wrapper_cfg = cfg.dataloader.extra_task[task].wrapper
                # look for the last wrapper
                while "model" in wrapper_cfg:
                    wrapper_cfg = wrapper_cfg.model
                wrapper_cfg.model = get_model_from_module(model)
                # poping _with_dataset_
                if wrapper_cfg.pop("_with_dataset_", False):
                    wrapper_cfg.dataset = get_dataset_from_loader(loader)
                inference_model = create_ddp_model(
                    instantiate(cfg.dataloader.extra_task[task].wrapper)
                )
            else:
                inference_model = model

            ret = inference_on_dataset(
                inference_model,
                loader,
                instantiate(cfg.dataloader.extra_task[task].evaluator),
                use_amp=cfg.train.amp.enabled,
            )
            print_csv_format(ret)
            all_ret.update(ret)
    logger.info("Evaluation results for all tasks:")
    print_csv_format(all_ret)
    return all_ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                DiffCISCheckpointer (dict)
                ddp (dict)
    """
    logger = logging.getLogger("DiffCIS")
    # set wandb resume before create writer
    cfg.train.wandb.resume = args.resume and DiffCISCheckpointer.has_checkpoint_in_dir(
        cfg.train.output_dir
    )
    # create writers at the beginning for W&B logging
    if comm.is_main_process():
        writers = default_writers(cfg)
    comm.synchronize()

    # not sure why d2 use ExitStack(), maybe easier for multiple context
    with ExitStack() as stack:
        stack.enter_context(
            WriterStack(
                logger=logger,
                writers=writers if comm.is_main_process() else None,
            )
        )
        logger.info(f"Wandb resume: {cfg.train.wandb.resume}")
        # log config again for w&b
        logger.info(f"Config:\n{LazyConfig.to_py(cfg)}")

        model = instantiate_diffcis(cfg.model)
        model.to(cfg.train.device)

        cfg.optimizer.params.model = model
        optim = instantiate(cfg.optimizer)

        train_loader = instantiate(cfg.dataloader.train)

        if cfg.train.amp.enabled:
            model = create_ddp_model(model, **cfg.train.ddp)
            trainer = AMPTrainer(model, train_loader, optim, grad_clip=cfg.train.grad_clip)
        else:
            model = create_ddp_model(model, **cfg.train.ddp)
            trainer = SimpleTrainer(model, train_loader, optim, grad_clip=cfg.train.grad_clip)

        checkpointer = DiffCISCheckpointer(
            model,
            cfg.train.output_dir,
            trainer=trainer,
        )

        # set wandb resume before create writer
        cfg.train.wandb.resume = args.resume and checkpointer.has_checkpoint()
        logger.info(f"Wandb resume: {cfg.train.wandb.resume}")

        trainer.register_hooks(
            [
                hooks.IterationTimer(),
                hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
                hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
                if comm.is_main_process()
                else None,
                EvalHook(cfg.train.eval_period, InferenceRunner(cfg, model)),
                hooks.BestCheckpointer(checkpointer=checkpointer, **cfg.train.best_checkpointer)
                if comm.is_main_process() and "best_checkpointer" in cfg.train
                else None,
                hooks.PeriodicWriter(
                    writers=writers,
                    period=cfg.train.log_period,
                )
                if comm.is_main_process()
                else None,
            ]
        )
        comm.synchronize()

        checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
        if args.resume and checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            start_iter = trainer.iter + 1
        else:
            start_iter = 0
    comm.synchronize()
    # keep trainer.train() out of stack since it has try/catch handling
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg.train.run_name = (
        "${train.cfg_name}_bs${dataloader.train.total_batch_size}" + f"x{comm.get_world_size()}"
    )
    print(cfg.train.run_name)
    if hasattr(args, "reference_world_size") and args.reference_world_size:
        cfg.train.reference_world_size = args.reference_world_size
    cfg = auto_scale_workers(cfg, comm.get_world_size())
    cfg.train.cfg_name = osp.splitext(osp.basename(args.config_file))[0]
    if hasattr(args, "output") and args.output:
        cfg.train.output_dir = args.output
    else:
        cfg.train.output_dir = osp.join("output", cfg.train.run_name)
    if hasattr(args, "tag") and args.tag:
        cfg.train.run_tag = args.tag
        cfg.train.output_dir = osp.join(cfg.train.output_dir, cfg.train.run_tag)
    if hasattr(args, "wandb") and args.wandb:
        cfg.train.wandb.enable_writer = args.wandb
        cfg.train.wandb.enable_visualizer = args.wandb
    if hasattr(args, "amp") and args.amp:
        cfg.train.amp.enabled = args.amp
    if hasattr(args, "init_from") and args.init_from:
        cfg.train.init_checkpoint = args.init_from
    cfg.train.log_dir = cfg.train.output_dir
    if hasattr(args, "log_tag") and args.log_tag:
        cfg.train.log_dir = osp.join(cfg.train.log_dir, args.log_tag)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    logger = setup_logger(cfg.train.log_dir, distributed_rank=comm.get_rank(), name="DiffCIS")

    logger.info(f"Running with config:\n{LazyConfig.to_py(cfg)}")

    register_all_cod10k_instance()

    if args.eval_only:
        model = instantiate_diffcis(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DiffCISCheckpointer(model, cfg.train.output_dir).resume_or_load(
            cfg.train.init_checkpoint, resume=args.resume
        )
        with ExitStack() as stack:
            stack.enter_context(
                WriterStack(
                    logger=logger,
                    writers=default_writers(cfg) if comm.is_main_process() else None,
                )
            )
            logger.info(do_test(cfg, model, final_iter=True))
        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()
    else:
        do_train(args, cfg)


def parse_args():
    parser = argparse.ArgumentParser(
        "DiffCIS training and evaluation script",
        parents=[default_argument_parser()],
        add_help=False,
    )

    parser.add_argument(
        "--output",
        type=str,
        help="root of output folder, " "the full path is <output>/<model_name>/<tag>",
    )
    parser.add_argument("--init-from", type=str, help="init from the given checkpoint")
    parser.add_argument("--tag", default="default", type=str, help="tag of experiment")
    parser.add_argument("--log-tag", type=str, help="tag of experiment")
    parser.add_argument("--wandb", action="store_true", help="Use W&B to log experiments")
    parser.add_argument("--amp", action="store_true", help="Use AMP for mixed precision training")
    parser.add_argument("--reference-world-size", "--ref", type=int)
    parser.add_argument("--eval_only", action="store_true", help="Use for evaluate only")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
