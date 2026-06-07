# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import os
import os.path as osp

import imageio
import torch
import torchvision

__all__ = ['cache_video', 'cache_image', 'str2bool']


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file
    cache_dir = osp.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # save to cache
    error = None
    for _ in range(retry):
        try:
            # preprocess
            tensor = tensor.detach()
            frames = []
            for u in tensor.unbind(2):
                if u.is_floating_point():
                    u = u.clamp(min(value_range), max(value_range))
                    frame = torchvision.utils.make_grid(
                        u,
                        nrow=nrow,
                        normalize=normalize,
                        value_range=value_range)
                    frame = (frame * 255).round().clamp(0, 255).to(torch.uint8)
                else:
                    frame = torchvision.utils.make_grid(
                        u, nrow=nrow, normalize=False)
                    frame = frame.clamp(0, 255).to(torch.uint8)
                frames.append(frame)
            tensor = torch.stack(frames, dim=1).permute(1, 2, 3, 0).cpu()

            # write video
            writer = imageio.get_writer(
                cache_file, fps=fps, codec='libx264', quality=10)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            return cache_file
        except Exception as e:
            error = e
            continue
    else:
        print(f'cache_video failed, error: {error}', flush=True)
        return None


def cache_image(tensor,
                save_file,
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    save_dir = osp.dirname(save_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    # save to cache
    error = None
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            torchvision.utils.save_image(
                tensor,
                save_file,
                nrow=nrow,
                normalize=normalize,
                value_range=value_range)
            return save_file
        except Exception as e:
            error = e
            continue


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')
