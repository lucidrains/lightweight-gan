import os
import fire
import random
from retry.api import retry_call
from tqdm import tqdm
from datetime import datetime
from functools import wraps
from lightweight_gan import Trainer, NanException

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np

def cast_list(el):
    return el if isinstance(el, list) else [el]

def timestamped_filename(prefix = 'generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def run_training(rank, world_size, model_args, data, load_from, new, num_train_steps, name, seed):
    is_main = rank == 0
    is_ddp = world_size > 1

    if is_ddp:
        set_seed(seed)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        print(f"{rank + 1}/{world_size} process initialized.")

    model_args.update(
        is_ddp = is_ddp,
        rank = rank,
        world_size = world_size
    )

    model = Trainer(**model_args)

    if not new:
        model.load(load_from)
    else:
        model.clear()

    model.set_data_src(data)

    for _ in tqdm(range(num_train_steps - model.steps), initial = model.steps, total = num_train_steps, mininterval=10., desc=f'{name}<{data}>'):
        retry_call(model.train, tries=3, exceptions=NanException)
        if is_main and _ % 50 == 0:
            model.print_log()

    model.save(model.checkpoint_num)

    if is_ddp:
        dist.destroy_process_group()

def train_from_folder(
    data = './data',
    results_dir = './results',
    models_dir = './models',
    name = 'default',
    new = False,
    load_from = -1,
    image_size = 256,
    fmap_max = 512,
    transparent = False,
    batch_size = 10,
    gradient_accumulate_every = 4,
    num_train_steps = 150000,
    learning_rate = 2e-4,
    save_every = 1000,
    evaluate_every = 1000,
    generate = False,
    generate_interpolation = False,
    hamburger_res_layers = [32],
    disc_output_size = 5,
    interpolation_num_steps = 100,
    save_frames = False,
    num_image_tiles = 8,
    trunc_psi = 0.75,
    aug_prob = 0.,
    aug_types = ['cutout'],
    dataset_aug_prob = 0.,
    multi_gpus = False,
    calculate_fid_every = None,
    seed = 42,
    activation = 'leaky_relu'
):
    model_args = dict(
        name = name,
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        hamburger_res_layers = cast_list(hamburger_res_layers),
        disc_output_size = disc_output_size,
        image_size = image_size,
        fmap_max = fmap_max,
        transparent = transparent,
        lr = learning_rate,
        save_every = save_every,
        evaluate_every = evaluate_every,
        trunc_psi = trunc_psi,
        aug_prob = aug_prob,
        aug_types = cast_list(aug_types),
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        activation = activation
    )

    if generate:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        model.evaluate(samples_name, num_image_tiles)
        print(f'sample images generated at {results_dir}/{name}/{samples_name}')
        return

    if generate_interpolation:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        model.generate_interpolation(samples_name, num_image_tiles, num_steps = interpolation_num_steps, save_frames = save_frames)
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')
        return

    world_size = torch.cuda.device_count()

    if world_size == 1 or not multi_gpus:
        run_training(0, 1, model_args, data, load_from, new, num_train_steps, name, seed)
        return

    mp.spawn(run_training,
        args=(world_size, model_args, data, load_from, new, num_train_steps, name, seed),
        nprocs=world_size,
        join=True)

def main():
    fire.Fire(train_from_folder)
