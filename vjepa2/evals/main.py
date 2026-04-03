# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing as mp
import os
import pprint

import yaml

from evals.scaffold import main as eval_main

from fbvscodedebug import start_debugger
from src.utils.distributed import init_distributed


parser = argparse.ArgumentParser()
parser.add_argument(
    "--val_only", action="store_true", help="only run eval", default=False
)
parser.add_argument(
    "--save_only", action="store_true", help="save dataset locally", default=False
)
parser.add_argument(
    "--cache_encoder",
    action="store_true",
    help="cache encoder features to disk",
    default=False,
)
parser.add_argument(
    "--use_cached_features",
    action="store_true",
    help="use cached features for training",
    default=False,
)
parser.add_argument(
    "--fname", type=str, help="name of config file to load", default="configs.yaml"
)
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=[
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
        "cuda:4",
        "cuda:5",
        "cuda:6",
        "cuda:7",
    ],
    help="which devices to use on local machine",
)
parser.add_argument(
    "--debugmode",
    type=bool,
    default=False,
    help="Setting this to true will not spin up new processes. "
    "The main code runs the main process, which makes it easier to debug with checkpointing.",
)
parser.add_argument(
    "--folder",
    type=str,
    help="location to save logs",
    default="checkpoints/",
)
parser.add_argument("--override_config_folder", action="store_true")
parser.add_argument("--checkpoint", type=str, help="location of pretrained ckpt")
parser.add_argument("--model_name", type=str, help="Model name")
parser.add_argument("--batch_size", type=int)
parser.add_argument("--num_workers", type=int)
parser.add_argument("--use_fsdp", action="store_true")


def process_main(args, rank, fname, world_size, devices):
    import logging
    import os

    start_debugger()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # Load config
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        if args.val_only:
            params["val_only"] = True

        if args.save_only:
            params["save_only"] = True

        if args.cache_encoder:
            params["cache_encoder"] = True
            # 缓存模式只需要遍历一次数据集
            params["experiment"]["optimization"]["num_epochs"] = 1
            logger.info("[CACHE_ENCODER] Forcing num_epochs=1 for cache generation")

        if args.use_cached_features:
            params["use_cached_features"] = True

        if args.checkpoint:
            params["model_kwargs"]["checkpoint"] = args.checkpoint

        if args.model_name:
            params["model_kwargs"]["pretrain_kwargs"]["encoder"][
                "model_name"
            ] = args.model_name

        if args.batch_size:
            params["experiment"]["optimization"]["batch_size"] = args.batch_size

        if args.override_config_folder:
            params["folder"] = args.folder
        params["use_fsdp"] = args.use_fsdp
        logger.info("loaded params...")

    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running... (rank: {rank}/{world_size})")

    # Launch the eval with loaded config
    eval_main(params["eval_name"], args_eval=params)


if __name__ == "__main__":
    # CRITICAL: 在 fork 之前设置环境变量，防止 BLAS/OpenMP 多线程冲突
    # 必须在任何 import torch/numpy 之前设置
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # 确保使用 spawn（已有，保持不变）
    # mp.set_start_method("spawn") 在 line 162

    args = parser.parse_args()
    # start_debugger()
    if args.debugmode:
        # FSDP debugging (use torchrun)
        if args.use_fsdp:
            process_main(
                args=args,
                rank=int(os.environ["RANK"]),
                fname=args.fname,
                world_size=int(os.environ["WORLD_SIZE"]),
                devices=args.devices,
            )
        # Single-GPU debugging
        else:
            process_main(
                args=args, rank=0, fname=args.fname, world_size=1, devices=["cuda:0"]
            )
    else:
        num_gpus = len(args.devices)
        mp.set_start_method("spawn")
        for rank in range(num_gpus):
            mp.Process(
                target=process_main,
                args=(args, rank, args.fname, num_gpus, args.devices),
            ).start()
