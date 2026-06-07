import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
import numpy as np
import cv2

import tempfile
import subprocess

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.channel_coding import (aggregate_channel_scores,
                                   decode_channel_scores,
                                   expand_gray_symbols,
                                   prepare_channel_message,
                                   usable_channel_symbol_count)
import torchvision.io as io
from wan.utils.utils import cache_video
import datetime



def add_gaussian_noise(img, std=0.05):
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_img

def apply_image_level_sp(frame, pepper, salt):
    out = frame.astype(np.float32)

    out[pepper][:,0] = 0   
    out[salt][:,0]   = 255    

    return out.astype(np.uint8)

def adjust_brightness(img, delta=0.1):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + delta), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def apply_crf_compression(
    input_mp4: str,
    crf: int = 18,
    preset: str = "fast",
    output_suffix: str = "_crf",
):
    assert input_mp4.endswith(".mp4")

    base, ext = os.path.splitext(input_mp4)
    output_mp4 = f"{base}{output_suffix}{crf}{ext}"

    cmd = [
        "ffmpeg",
        "-y",                     
        "-i", input_mp4,          
        "-c:v", "libx264",        
        "-crf", str(crf),         
        "-preset", preset,       
        output_mp4,
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    return output_mp4


_TIERED_UEP_FRACTIONS = (0.2, 0.5, 0.3)
_TIERED_UEP_SPECS = {
    'score_uep_4_8_12': {
        'payload_redundancies': (4, 8, 12),
        'repair_weak_tier': False,
    },
    'score_uep_4_8_14': {
        'payload_redundancies': (4, 8, 14),
        'repair_weak_tier': False,
    },
    'score_uep_3_8_12': {
        'payload_redundancies': (3, 8, 12),
        'repair_weak_tier': False,
    },
    'score_uep_4_8_repair': {
        'payload_redundancies': (4, 8, None),
        'repair_weak_tier': True,
    },
    'score_uep_4_6_repair': {
        'payload_redundancies': (4, 6, None),
        'repair_weak_tier': True,
    },
    'score_uep_3_8_repair': {
        'payload_redundancies': (3, 8, None),
        'repair_weak_tier': True,
    },
}

_INTERLEAVE_SPECS = {
    'interleave_time': {
        'use_time': True,
        'use_space': False,
        'use_channel': False,
    },
    'interleave_time_space': {
        'use_time': True,
        'use_space': True,
        'use_channel': False,
    },
    'interleave_time_space_channel': {
        'use_time': True,
        'use_space': True,
        'use_channel': True,
    },
}


def _empty_group_plan_stats():
    return {
        'time_overlap': 0.0,
        'space_overlap': 0.0,
        'channel_overlap': 0.0,
        'signature_overlap': 0.0,
    }


def _flatten_to_latent_coords(flat_indices, latent_shape):
    channel_dim, time_dim, height_dim, width_dim = [int(value) for value in latent_shape]
    spatial_plane = height_dim * width_dim
    temporal_plane = time_dim * spatial_plane

    channels = torch.div(flat_indices, temporal_plane, rounding_mode='floor')
    temporal_offset = torch.remainder(flat_indices, temporal_plane)
    times = torch.div(temporal_offset, spatial_plane, rounding_mode='floor')
    spatial_offset = torch.remainder(temporal_offset, spatial_plane)
    heights = torch.div(spatial_offset, width_dim, rounding_mode='floor')
    widths = torch.remainder(spatial_offset, width_dim)
    return channels, times, heights, widths


def _sorted_bucket_pairs(first_bucket, second_bucket):
    low_bucket = torch.minimum(first_bucket, second_bucket)
    high_bucket = torch.maximum(first_bucket, second_bucket)
    return torch.stack([low_bucket, high_bucket], dim=1).to(torch.long)


def _build_pair_interleave_features(pair_i, pair_j, latent_shape):
    if pair_i is None or pair_j is None or latent_shape is None or pair_i.numel() == 0:
        return None

    pair_i_cpu = pair_i.detach().to(device='cpu', dtype=torch.long)
    pair_j_cpu = pair_j.detach().to(device='cpu', dtype=torch.long)

    channel_dim, time_dim, height_dim, width_dim = [int(value) for value in latent_shape]
    spatial_grid_h = max(1, min(4, height_dim))
    spatial_grid_w = max(1, min(4, width_dim))
    spatial_bucket_h = max(1, math.ceil(height_dim / spatial_grid_h))
    spatial_bucket_w = max(1, math.ceil(width_dim / spatial_grid_w))
    spatial_bucket_count = spatial_grid_h * spatial_grid_w
    channel_group_count = max(1, min(channel_dim, 4))
    channel_bucket_size = max(1, math.ceil(channel_dim / channel_group_count))

    channel_i, time_i, height_i, width_i = _flatten_to_latent_coords(pair_i_cpu, latent_shape)
    channel_j, time_j, height_j, width_j = _flatten_to_latent_coords(pair_j_cpu, latent_shape)

    spatial_i = torch.div(height_i, spatial_bucket_h, rounding_mode='floor').clamp_max(spatial_grid_h - 1)
    spatial_i = spatial_i * spatial_grid_w + torch.div(width_i, spatial_bucket_w, rounding_mode='floor').clamp_max(spatial_grid_w - 1)
    spatial_j = torch.div(height_j, spatial_bucket_h, rounding_mode='floor').clamp_max(spatial_grid_h - 1)
    spatial_j = spatial_j * spatial_grid_w + torch.div(width_j, spatial_bucket_w, rounding_mode='floor').clamp_max(spatial_grid_w - 1)

    channel_bucket_i = torch.div(channel_i, channel_bucket_size, rounding_mode='floor').clamp_max(channel_group_count - 1)
    channel_bucket_j = torch.div(channel_j, channel_bucket_size, rounding_mode='floor').clamp_max(channel_group_count - 1)

    time_buckets = _sorted_bucket_pairs(time_i, time_j)
    space_buckets = _sorted_bucket_pairs(spatial_i, spatial_j)
    channel_buckets = _sorted_bucket_pairs(channel_bucket_i, channel_bucket_j)

    time_pair_keys = time_buckets[:, 0] * time_dim + time_buckets[:, 1]
    space_pair_keys = space_buckets[:, 0] * spatial_bucket_count + space_buckets[:, 1]
    channel_pair_keys = channel_buckets[:, 0] * channel_group_count + channel_buckets[:, 1]

    return {
        'time_buckets': time_buckets,
        'space_buckets': space_buckets,
        'channel_buckets': channel_buckets,
        'time_pair_keys': time_pair_keys,
        'space_pair_keys': space_pair_keys,
        'channel_pair_keys': channel_pair_keys,
        'time_key_range': time_dim * time_dim,
        'space_key_range': spatial_bucket_count * spatial_bucket_count,
        'channel_key_range': channel_group_count * channel_group_count,
    }


def _build_round_robin_bucket_order(bucket_map, seed, device):
    random_generator = random.Random(seed)
    bucket_items = list(bucket_map.items())
    random_generator.shuffle(bucket_items)

    bucket_lists = []
    for _, positions in bucket_items:
        positions = list(positions)
        random_generator.shuffle(positions)
        bucket_lists.append(positions)
    bucket_lists.sort(key=len, reverse=True)

    ordered_positions = []
    active_lists = bucket_lists
    while active_lists:
        next_active_lists = []
        for positions in active_lists:
            ordered_positions.append(positions.pop())
            if positions:
                next_active_lists.append(positions)
        random_generator.shuffle(next_active_lists)
        next_active_lists.sort(key=len, reverse=True)
        active_lists = next_active_lists

    return torch.tensor(ordered_positions, device=device, dtype=torch.long)


def _build_uniform_interleave_positions(pair_i,
                                        pair_j,
                                        usable_pair_count,
                                        seed,
                                        device,
                                        latent_shape,
                                        interleave_spec):
    usable_pair_i = pair_i[:usable_pair_count]
    usable_pair_j = pair_j[:usable_pair_count]
    interleave_features = _build_pair_interleave_features(
        usable_pair_i,
        usable_pair_j,
        latent_shape)
    if interleave_features is None:
        return torch.empty(0, device=device, dtype=torch.long)

    combined_keys = torch.zeros(usable_pair_count, dtype=torch.long)
    current_range = 1
    if interleave_spec['use_time']:
        combined_keys += interleave_features['time_pair_keys']
        current_range = interleave_features['time_key_range']
    if interleave_spec['use_space']:
        combined_keys = combined_keys * interleave_features['space_key_range'] + interleave_features['space_pair_keys']
        current_range *= interleave_features['space_key_range']
    if interleave_spec['use_channel']:
        combined_keys = combined_keys * interleave_features['channel_key_range'] + interleave_features['channel_pair_keys']
        current_range *= interleave_features['channel_key_range']

    if current_range == 1:
        return torch.arange(usable_pair_count, device=device, dtype=torch.long)

    bucket_map = {}
    for pair_position, bucket_key in enumerate(combined_keys.tolist()):
        bucket_map.setdefault(int(bucket_key), []).append(pair_position)

    return _build_round_robin_bucket_order(bucket_map, seed + 23, device)


def _bucket_pairs_overlap(first_bucket_pair, second_bucket_pair):
    return bool(
        first_bucket_pair[0] == second_bucket_pair[0] or
        first_bucket_pair[0] == second_bucket_pair[1] or
        first_bucket_pair[1] == second_bucket_pair[0] or
        first_bucket_pair[1] == second_bucket_pair[1])


def _summarize_group_plan(grouped_pair_positions,
                          symbol_group_ids,
                          symbol_group_sizes,
                          pair_i,
                          pair_j,
                          latent_shape):
    if (grouped_pair_positions.numel() == 0 or symbol_group_sizes.numel() == 0 or
            pair_i is None or pair_j is None or latent_shape is None):
        return _empty_group_plan_stats()

    interleave_features = _build_pair_interleave_features(
        pair_i[grouped_pair_positions],
        pair_j[grouped_pair_positions],
        latent_shape)
    if interleave_features is None:
        return _empty_group_plan_stats()

    time_buckets = interleave_features['time_buckets'].tolist()
    space_buckets = interleave_features['space_buckets'].tolist()
    channel_buckets = interleave_features['channel_buckets'].tolist()

    comparison_count = 0
    time_overlap_count = 0
    space_overlap_count = 0
    channel_overlap_count = 0
    signature_overlap_sum = 0.0

    start = 0
    for group_size in symbol_group_sizes.detach().to(device='cpu', dtype=torch.long).tolist():
        end = start + int(group_size)
        if group_size >= 2:
            for first_idx in range(start, end):
                for second_idx in range(first_idx + 1, end):
                    time_overlap = _bucket_pairs_overlap(
                        time_buckets[first_idx], time_buckets[second_idx])
                    space_overlap = _bucket_pairs_overlap(
                        space_buckets[first_idx], space_buckets[second_idx])
                    channel_overlap = _bucket_pairs_overlap(
                        channel_buckets[first_idx], channel_buckets[second_idx])

                    comparison_count += 1
                    time_overlap_count += int(time_overlap)
                    space_overlap_count += int(space_overlap)
                    channel_overlap_count += int(channel_overlap)
                    signature_overlap_sum += (
                        4.0 * float(time_overlap) +
                        2.0 * float(space_overlap) +
                        1.0 * float(channel_overlap)) / 7.0
        start = end

    if comparison_count == 0:
        return _empty_group_plan_stats()

    return {
        'time_overlap': time_overlap_count / comparison_count,
        'space_overlap': space_overlap_count / comparison_count,
        'channel_overlap': channel_overlap_count / comparison_count,
        'signature_overlap': signature_overlap_sum / comparison_count,
    }


def _attach_group_plan_stats(grouped_pair_positions,
                             symbol_group_ids,
                             symbol_group_sizes,
                             pair_i,
                             pair_j,
                             latent_shape):
    return (
        grouped_pair_positions,
        symbol_group_ids,
        symbol_group_sizes,
        _summarize_group_plan(
            grouped_pair_positions,
            symbol_group_ids,
            symbol_group_sizes,
            pair_i,
            pair_j,
            latent_shape))


def _split_positions_by_fraction(sorted_positions, fractions):
    if sorted_positions.numel() == 0:
        return [sorted_positions[:0] for _ in fractions]

    pair_count = int(sorted_positions.numel())
    counts = [int(pair_count * fraction) for fraction in fractions[:-1]]
    counts.append(pair_count - sum(counts))

    tier_positions = []
    start = 0
    for count in counts:
        tier_positions.append(sorted_positions[start:start + count])
        start += count
    return tier_positions


def _finalize_explicit_group_plan(grouped_pair_positions,
                                  symbol_group_ids,
                                  group_count,
                                  device):
    if grouped_pair_positions.numel() == 0 or group_count <= 0:
        empty_positions = torch.empty(0, device=device, dtype=torch.long)
        empty_sizes = torch.empty(0, device=device, dtype=torch.long)
        return empty_positions, empty_positions, empty_sizes

    symbol_group_sizes = torch.bincount(
        symbol_group_ids,
        minlength=group_count).to(torch.long)
    return grouped_pair_positions, symbol_group_ids, symbol_group_sizes


def build_symbol_group_plan(pair_count,
                            symbol_redundancy,
                            seed,
                            device,
                            redundancy_schedule='uniform',
                            selection_scores=None,
                            pair_i=None,
                            pair_j=None,
                            latent_shape=None):
    redundancy = max(int(symbol_redundancy), 1)
    pair_positions = torch.arange(pair_count, device=device, dtype=torch.long)
    empty_positions = pair_positions[:0]
    empty_sizes = torch.empty(0, device=device, dtype=torch.long)

    def finalize(grouped_pair_positions, symbol_group_sizes):
        if grouped_pair_positions.numel() == 0 or symbol_group_sizes.numel() == 0:
            return empty_positions, empty_positions, empty_sizes, _empty_group_plan_stats()
        symbol_group_ids = torch.arange(
            symbol_group_sizes.numel(),
            device=device,
            dtype=torch.long).repeat_interleave(symbol_group_sizes)
        return _attach_group_plan_stats(
            grouped_pair_positions,
            symbol_group_ids,
            symbol_group_sizes,
            pair_i,
            pair_j,
            latent_shape)

    if pair_positions.numel() == 0:
        return empty_positions, empty_positions, empty_sizes, _empty_group_plan_stats()

    if redundancy_schedule == 'uniform':
        if redundancy == 1:
            return finalize(
                pair_positions,
                torch.ones(pair_count, device=device, dtype=torch.long))

        symbol_count = pair_count // redundancy
        if symbol_count == 0:
            return empty_positions, empty_positions, empty_sizes, _empty_group_plan_stats()

        group_generator = torch.Generator(device=str(device))
        group_generator.manual_seed(seed + 2)
        grouped_pair_positions = torch.randperm(
            pair_count,
            device=device,
            generator=group_generator)[:symbol_count * redundancy]
        symbol_group_sizes = torch.full(
            (symbol_count,),
            redundancy,
            device=device,
            dtype=torch.long)
        return finalize(grouped_pair_positions, symbol_group_sizes)

    interleave_spec = _INTERLEAVE_SPECS.get(redundancy_schedule)
    if interleave_spec is not None:
        if pair_i is None or pair_j is None or latent_shape is None:
            raise ValueError(
                f'{redundancy_schedule} requires pair_i, pair_j, and latent_shape.')
        symbol_count = pair_count // redundancy
        if symbol_count == 0:
            return empty_positions, empty_positions, empty_sizes, _empty_group_plan_stats()
        usable_pair_count = symbol_count * redundancy
        grouped_pair_positions = _build_uniform_interleave_positions(
            pair_i,
            pair_j,
            usable_pair_count,
            seed,
            device,
            latent_shape,
            interleave_spec)
        symbol_group_sizes = torch.full(
            (symbol_count,),
            redundancy,
            device=device,
            dtype=torch.long)
        return finalize(grouped_pair_positions, symbol_group_sizes)

    if redundancy_schedule == 'score_uep_6_10':
        if redundancy != 8:
            raise ValueError(
                'score_uep_6_10 currently expects symbol_redundancy=8.')
        if selection_scores is None or selection_scores.numel() != pair_count:
            raise ValueError(
                'score_uep_6_10 requires one selection score per pair.')

        low_redundancy = 6
        high_redundancy = 10
        base_group_count = pair_count // (low_redundancy + high_redundancy)
        low_group_count = base_group_count
        high_group_count = base_group_count
        remaining_pairs = pair_count - base_group_count * (
            low_redundancy + high_redundancy)
        if remaining_pairs >= low_redundancy:
            low_group_count += 1

        if low_group_count + high_group_count == 0:
            return empty_positions, empty_positions, empty_sizes, _empty_group_plan_stats()

        low_pair_count = low_group_count * low_redundancy
        high_pair_count = high_group_count * high_redundancy
        sorted_positions = torch.argsort(selection_scores, descending=True)
        low_positions = sorted_positions[:low_pair_count]
        high_positions = sorted_positions[-high_pair_count:] if high_pair_count > 0 else empty_positions

        low_generator = torch.Generator(device=str(device))
        low_generator.manual_seed(seed + 2)
        low_positions = low_positions[torch.randperm(
            low_positions.numel(),
            device=device,
            generator=low_generator)]

        if high_positions.numel() > 0:
            high_generator = torch.Generator(device=str(device))
            high_generator.manual_seed(seed + 3)
            high_positions = high_positions[torch.randperm(
                high_positions.numel(),
                device=device,
                generator=high_generator)]

        grouped_pair_positions = torch.cat(
            [low_positions, high_positions],
            dim=0)
        symbol_group_sizes = torch.cat(
            [
                torch.full(
                    (low_group_count,),
                    low_redundancy,
                    device=device,
                    dtype=torch.long),
                torch.full(
                    (high_group_count,),
                    high_redundancy,
                    device=device,
                    dtype=torch.long),
            ],
            dim=0)
        return finalize(grouped_pair_positions, symbol_group_sizes)

    tiered_spec = _TIERED_UEP_SPECS.get(redundancy_schedule)
    if tiered_spec is not None:
        if selection_scores is None or selection_scores.numel() != pair_count:
            raise ValueError(
                f'{redundancy_schedule} requires one selection score per pair.')

        sorted_positions = torch.argsort(selection_scores, descending=True)
        tier_positions = _split_positions_by_fraction(
            sorted_positions,
            _TIERED_UEP_FRACTIONS)

        grouped_pair_parts = []
        grouped_id_parts = []
        tier_group_ids = []
        tier_group_mean_scores = []
        total_group_count = 0

        for tier_idx, tier_pair_positions in enumerate(tier_positions):
            redundancy_value = tiered_spec['payload_redundancies'][tier_idx]
            is_repair_tier = tiered_spec['repair_weak_tier'] and redundancy_value is None

            if tier_pair_positions.numel() == 0 or is_repair_tier:
                tier_group_ids.append(empty_positions)
                tier_group_mean_scores.append(torch.empty(0, device=device, dtype=torch.float32))
                continue

            tier_group_count = tier_pair_positions.numel() // int(redundancy_value)
            if tier_group_count == 0:
                tier_group_ids.append(empty_positions)
                tier_group_mean_scores.append(torch.empty(0, device=device, dtype=torch.float32))
                continue

            usable_pair_count = tier_group_count * int(redundancy_value)
            tier_pair_positions = tier_pair_positions[:usable_pair_count]
            current_group_ids = torch.arange(
                total_group_count,
                total_group_count + tier_group_count,
                device=device,
                dtype=torch.long)

            grouped_pair_parts.append(tier_pair_positions)
            grouped_id_parts.append(current_group_ids.repeat_interleave(int(redundancy_value)))
            tier_group_ids.append(current_group_ids)
            tier_group_mean_scores.append(selection_scores[tier_pair_positions].float().view(
                tier_group_count,
                int(redundancy_value)).mean(dim=1))
            total_group_count += tier_group_count

        if tiered_spec['repair_weak_tier'] and tier_positions[2].numel() > 0 and total_group_count > 0:
            repair_positions = tier_positions[2]
            target_group_ids = tier_group_ids[1]
            target_group_scores = tier_group_mean_scores[1]

            if target_group_ids.numel() == 0:
                available_group_ids = [group_ids for group_ids in tier_group_ids if group_ids.numel() > 0]
                available_group_scores = [scores for scores in tier_group_mean_scores if scores.numel() > 0]
                target_group_ids = torch.cat(available_group_ids, dim=0) if available_group_ids else empty_positions
                target_group_scores = torch.cat(available_group_scores, dim=0) if available_group_scores else torch.empty(0, device=device, dtype=torch.float32)

            if target_group_ids.numel() > 0:
                target_order = torch.argsort(target_group_scores)
                ordered_target_group_ids = target_group_ids[target_order]
                repair_group_ids = ordered_target_group_ids[
                    torch.arange(repair_positions.numel(), device=device) % ordered_target_group_ids.numel()]
                grouped_pair_parts.append(repair_positions)
                grouped_id_parts.append(repair_group_ids)

        if not grouped_pair_parts:
            return empty_positions, empty_positions, empty_sizes, _empty_group_plan_stats()

        explicit_grouped_pair_positions, explicit_symbol_group_ids, explicit_symbol_group_sizes = _finalize_explicit_group_plan(
            torch.cat(grouped_pair_parts, dim=0),
            torch.cat(grouped_id_parts, dim=0),
            total_group_count,
            device)
        return _attach_group_plan_stats(
            explicit_grouped_pair_positions,
            explicit_symbol_group_ids,
            explicit_symbol_group_sizes,
            pair_i,
            pair_j,
            latent_shape)

    raise ValueError(f'Unsupported redundancy_schedule: {redundancy_schedule}')


def nearest_safe_angles(phi, lower, upper):
    full_turn = 2 * math.pi
    candidate_angles = []
    candidate_diffs = []
    for offset in (-full_turn, 0.0, full_turn):
        shifted_lower = phi.new_full(phi.shape, lower + offset)
        shifted_upper = phi.new_full(phi.shape, upper + offset)
        candidate = torch.maximum(
            torch.minimum(phi, shifted_upper), shifted_lower)
        candidate_angles.append(candidate)
        candidate_diffs.append(torch.abs(candidate - phi))

    candidate_angles = torch.stack(candidate_angles, dim=0)
    candidate_diffs = torch.stack(candidate_diffs, dim=0)
    best_idx = candidate_diffs.argmin(dim=0, keepdim=True)
    nearest = torch.gather(candidate_angles, 0, best_idx).squeeze(0)
    return torch.remainder(nearest, full_turn)


def build_sector_ranges(sector_margin_deg, k=4):
    if k == 2:
        return [(0.0, math.pi), (math.pi, 2 * math.pi)]
    if k == 4:
        sector_margin = math.radians(sector_margin_deg)
        return [
            (sector_margin, math.pi / 2 - sector_margin),
            (math.pi / 2 + sector_margin, math.pi - sector_margin),
            (math.pi + sector_margin, 3 * math.pi / 2 - sector_margin),
            (3 * math.pi / 2 + sector_margin, 2 * math.pi - sector_margin),
        ]
    raise ValueError(f"Unsupported k: {k}")


def build_pair_indices(mask, seed, device):
    flat_mask = mask.flatten()
    safe_indices = torch.nonzero(flat_mask, as_tuple=True)[0]
    empty = safe_indices[:0]
    if safe_indices.numel() < 2:
        return empty, empty

    pair_generator = torch.Generator(device=str(device))
    pair_generator.manual_seed(seed)
    shuffled_all = torch.randperm(
        flat_mask.numel(),
        device=flat_mask.device,
        generator=pair_generator)

    block_size = 4096
    pair_i_blocks = []
    pair_j_blocks = []
    for start in range(0, shuffled_all.numel(), block_size):
        block_indices = shuffled_all[start:start + block_size]
        safe_block = block_indices[flat_mask[block_indices]]
        pair_count = safe_block.numel() // 2
        if pair_count == 0:
            continue
        pair_i_blocks.append(safe_block[0:pair_count * 2:2])
        pair_j_blocks.append(safe_block[1:pair_count * 2:2])

    if not pair_i_blocks:
        return empty, empty

    pair_i = torch.cat(pair_i_blocks)
    pair_j = torch.cat(pair_j_blocks)
    return pair_i, pair_j


def build_anchor_pair_values(anchor_latents, pair_i, pair_j):
    if pair_i.numel() == 0:
        return torch.empty(0, 0, 2, device=pair_i.device, dtype=torch.float32)
    return torch.stack([
        torch.stack([
            anchor.flatten()[pair_i].float(),
            anchor.flatten()[pair_j].float()
        ], dim=-1)
        for anchor in anchor_latents
    ], dim=0)


def build_pair_values(latent, pair_i, pair_j):
    if pair_i.numel() == 0:
        return torch.empty(0, 2, device=pair_i.device, dtype=torch.float32)
    latent_flat = latent.flatten()
    return torch.stack([
        latent_flat[pair_i].float(),
        latent_flat[pair_j].float()
    ], dim=-1)


def compute_pair_separability(anchor_pair_values):
    if anchor_pair_values.numel() == 0:
        return torch.empty(0, device=anchor_pair_values.device, dtype=torch.float32)

    pairwise_distances = []
    num_states = anchor_pair_values.shape[0]
    for first_idx in range(num_states):
        for second_idx in range(first_idx + 1, num_states):
            pairwise_distances.append(torch.sum(
                torch.abs(anchor_pair_values[first_idx] - anchor_pair_values[second_idx]),
                dim=-1))
    return torch.stack(pairwise_distances, dim=0).min(dim=0).values


def compute_base_cosine_separability(anchor_pair_values, base_pair_values):
    if anchor_pair_values.numel() == 0:
        return torch.empty(0, device=anchor_pair_values.device, dtype=torch.float32)

    offsets = anchor_pair_values - base_pair_values.unsqueeze(0)
    offset_norms = torch.linalg.norm(offsets, dim=-1).clamp_min(1e-8)
    normalized_offsets = offsets / offset_norms.unsqueeze(-1)

    pairwise_scores = []
    num_states = anchor_pair_values.shape[0]
    for first_idx in range(num_states):
        for second_idx in range(first_idx + 1, num_states):
            cosine_similarity = torch.sum(
                normalized_offsets[first_idx] * normalized_offsets[second_idx],
                dim=-1).clamp(-1.0, 1.0)
            angular_separation = 1.0 - cosine_similarity
            radial_floor = torch.minimum(
                offset_norms[first_idx], offset_norms[second_idx])
            pairwise_scores.append(angular_separation * radial_floor)

    return torch.stack(pairwise_scores, dim=0).min(dim=0).values


def compute_base_cosine_channel_margin_scores(anchor_pair_values,
                                              clean_base_pair_values,
                                              replay_anchor_pair_values,
                                              replay_base_pair_values):
    if anchor_pair_values.numel() == 0:
        return torch.empty(0, device=anchor_pair_values.device, dtype=torch.float32)

    reference_offsets = anchor_pair_values - clean_base_pair_values.unsqueeze(0)
    observed_offsets = replay_anchor_pair_values - replay_base_pair_values.unsqueeze(0)

    reference_offset_norms = torch.linalg.norm(
        reference_offsets, dim=-1).clamp_min(1e-8)
    observed_offset_norms = torch.linalg.norm(
        observed_offsets, dim=-1).clamp_min(1e-8)

    normalized_reference_offsets = (
        reference_offsets / reference_offset_norms.unsqueeze(-1))
    normalized_observed_offsets = (
        observed_offsets / observed_offset_norms.unsqueeze(-1))

    cosine_scores = torch.sum(
        normalized_observed_offsets.unsqueeze(1) *
        normalized_reference_offsets.unsqueeze(0),
        dim=-1).clamp(-1.0, 1.0)

    num_states = anchor_pair_values.shape[0]
    state_indices = torch.arange(num_states, device=anchor_pair_values.device)
    true_scores = cosine_scores[state_indices, state_indices]
    identity_mask = torch.eye(
        num_states, device=anchor_pair_values.device, dtype=torch.bool).unsqueeze(-1)
    max_other_scores = cosine_scores.masked_fill(
        identity_mask, float('-inf')).max(dim=1).values

    margins = true_scores - max_other_scores
    radial_term = torch.minimum(reference_offset_norms, observed_offset_norms)
    state_scores = margins * radial_term
    return state_scores.min(dim=0).values


def compute_mask1_pair_scores(abs_diff1, pair_i, pair_j):
    if pair_i.numel() == 0:
        return torch.empty(0, device=pair_i.device, dtype=torch.float32)

    diff_flat = abs_diff1.flatten().float()
    worst_case_diff = torch.maximum(diff_flat[pair_i], diff_flat[pair_j])
    return -worst_case_diff


def select_pair_positions(pair_scores,
                          pair_selection_mode,
                          val_mask2,
                          max_pairs,
                          eligible_mask=None):
    if pair_scores.numel() == 0:
        return torch.empty(0, device=pair_scores.device, dtype=torch.long)

    if eligible_mask is None:
        eligible_positions = torch.arange(
            pair_scores.numel(),
            device=pair_scores.device,
            dtype=torch.long)
    else:
        eligible_positions = torch.nonzero(eligible_mask, as_tuple=True)[0]

    if eligible_positions.numel() == 0:
        return torch.empty(0, device=pair_scores.device, dtype=torch.long)

    eligible_scores = pair_scores[eligible_positions]

    if max_pairs is not None:
        keep_count = min(max(int(max_pairs), 0), eligible_scores.numel())
        if keep_count == 0:
            return torch.empty(0, device=pair_scores.device, dtype=torch.long)
        sorted_positions = torch.argsort(eligible_scores, descending=True)
        return eligible_positions[sorted_positions[:keep_count]]

    if pair_selection_mode in ('full', 'mask2_only'):
        selection_threshold = torch.quantile(
            eligible_scores,
            min(max(val_mask2, 0.0), 1.0))
        selected_mask = eligible_scores >= selection_threshold
        return eligible_positions[selected_mask]

    return eligible_positions





class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def _replay_latent_through_video(self,
                                     latent,
                                     save_file=None,
                                     mask1_mode='clean',
                                     mask1_attack_type='crf',
                                     mask1_attack_crf=28):
        decoded_video = self.vae.decode([latent])
        replay_file = cache_video(
            tensor=decoded_video[0][None],
            save_file=save_file,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        cleanup_paths = []
        replay_source = replay_file

        if mask1_mode == 'attack_aware':
            if mask1_attack_type != 'crf':
                raise NotImplementedError(
                    f"Unsupported mask1_attack_type: {mask1_attack_type}")
            replay_source = apply_crf_compression(
                replay_file,
                crf=mask1_attack_crf,
                output_suffix='_mask1_crf')
            cleanup_paths.append(replay_source)

        replay_video, _, _ = io.read_video(replay_source, pts_unit='pts')
        replay_video = replay_video.permute(3, 0, 1, 2).unsqueeze(0).to(
            self.device).float()
        replay_video = (replay_video / 127.5) - 1
        replay_latent = self.vae.encode(replay_video)
        if isinstance(replay_latent, list):
            replay_latent = replay_latent[0]

        if save_file is None and replay_file is not None:
            cleanup_paths.append(replay_file)

        for cleanup_path in cleanup_paths:
            try:
                os.remove(cleanup_path)
            except OSError:
                pass

        return replay_latent

    def _build_mask1(self,
                     latent_base,
                     val_mask1,
                     save_file=None,
                     mask1_mode='clean',
                     mask1_attack_type='crf',
                     mask1_attack_crf=28):
        test_rev = self._replay_latent_through_video(
            latent_base,
            save_file=save_file,
            mask1_mode=mask1_mode,
            mask1_attack_type=mask1_attack_type,
            mask1_attack_crf=mask1_attack_crf)
        abs_diff1 = torch.abs(latent_base - test_rev)
        flat_diff1 = abs_diff1.flatten().float()
        threshold1 = torch.quantile(flat_diff1, val_mask1)
        mask1 = abs_diff1 < threshold1
        return test_rev, abs_diff1, mask1

    def _build_channel_pair_values(self,
                                   latent_base,
                                   anchor_latents,
                                   pair_i,
                                   pair_j,
                                   replay_mode='clean',
                                   attack_type='crf',
                                   attack_crf=28,
                                   base_replay_latent=None):
        if pair_i.numel() == 0:
            empty_pair_values = torch.empty(
                0, 2, device=pair_i.device, dtype=torch.float32)
            empty_anchor_values = torch.empty(
                0, 0, 2, device=pair_i.device, dtype=torch.float32)
            return empty_pair_values, empty_anchor_values, base_replay_latent

        replay_base_latent = base_replay_latent
        if replay_base_latent is None:
            replay_base_latent = self._replay_latent_through_video(
                latent_base,
                mask1_mode=replay_mode,
                mask1_attack_type=attack_type,
                mask1_attack_crf=attack_crf)

        replay_anchor_latents = [
            self._replay_latent_through_video(
                anchor_latent,
                mask1_mode=replay_mode,
                mask1_attack_type=attack_type,
                mask1_attack_crf=attack_crf)
            for anchor_latent in anchor_latents
        ]

        replay_base_pair_values = build_pair_values(
            replay_base_latent, pair_i, pair_j)
        replay_anchor_pair_values = build_anchor_pair_values(
            replay_anchor_latents, pair_i, pair_j)
        return replay_base_pair_values, replay_anchor_pair_values, replay_base_latent

    def _compute_pair_selection_scores(self,
                                       latent_base,
                                       anchor_latents,
                                       pair_i,
                                       pair_j,
                                       reference_mode='clean',
                                       mask2_mode='clean',
                                       mask2_attack_type='crf',
                                       mask2_attack_crf=28,
                                       mask2_attack_weight=1.0,
                                       mask2_attack_gate_quantile=0.2,
                                       mask2_tiebreak_epsilon=1e-4,
                                       current_mask1_mode='clean',
                                       current_mask1_attack_type='crf',
                                       current_mask1_attack_crf=28,
                                       current_base_replay_latent=None,
                                       selection_keep_count=None):
        anchor_pair_values = build_anchor_pair_values(anchor_latents, pair_i, pair_j)
        if pair_i.numel() == 0:
            return (
                torch.empty(0, device=pair_i.device, dtype=torch.float32),
                anchor_pair_values,
                torch.empty(0, device=pair_i.device, dtype=torch.bool))

        clean_base_pair_values = build_pair_values(latent_base, pair_i, pair_j)
        clean_selection_scores = compute_base_cosine_separability(
            anchor_pair_values,
            clean_base_pair_values) if reference_mode == 'base_cosine' else None
        eligible_mask = torch.ones(
            pair_i.numel(),
            device=pair_i.device,
            dtype=torch.bool)

        if mask2_mode == 'clean':
            if reference_mode == 'full_replay':
                reference_anchor_latents = [
                    self._replay_latent_through_video(anchor_latent)
                    for anchor_latent in anchor_latents
                ]
                reference_anchor_pair_values = build_anchor_pair_values(
                    reference_anchor_latents, pair_i, pair_j)
                return (
                    compute_pair_separability(reference_anchor_pair_values),
                    reference_anchor_pair_values,
                    eligible_mask)
            if reference_mode == 'base_cosine':
                return clean_selection_scores, anchor_pair_values, eligible_mask
            return compute_pair_separability(anchor_pair_values), anchor_pair_values, eligible_mask

        if reference_mode != 'base_cosine':
            raise NotImplementedError(
                "attack-aware mask2 currently only supports reference_mode='base_cosine'")

        attack_base_replay_latent = None
        if (current_mask1_mode == 'attack_aware' and
                current_mask1_attack_type == mask2_attack_type and
                current_mask1_attack_crf == mask2_attack_crf):
            attack_base_replay_latent = current_base_replay_latent

        attack_replay_base_pair_values, attack_replay_anchor_pair_values, _ = self._build_channel_pair_values(
            latent_base,
            anchor_latents,
            pair_i,
            pair_j,
            replay_mode='attack_aware',
            attack_type=mask2_attack_type,
            attack_crf=mask2_attack_crf,
            base_replay_latent=attack_base_replay_latent)

        attack_channel_scores = compute_base_cosine_channel_margin_scores(
            anchor_pair_values,
            clean_base_pair_values,
            attack_replay_anchor_pair_values,
            attack_replay_base_pair_values)

        if mask2_mode == 'attack_only':
            return attack_channel_scores, anchor_pair_values, eligible_mask

        if mask2_mode == 'clean_gate':
            gate_quantile = min(max(mask2_attack_gate_quantile, 0.0), 1.0)
            attack_threshold = torch.quantile(attack_channel_scores, gate_quantile)
            eligible_mask = attack_channel_scores >= attack_threshold
            if selection_keep_count is not None and eligible_mask.sum().item() < int(selection_keep_count):
                top_gate_count = min(int(selection_keep_count), attack_channel_scores.numel())
                top_gate_indices = torch.argsort(
                    attack_channel_scores,
                    descending=True)[:top_gate_count]
                eligible_mask = torch.zeros_like(eligible_mask)
                eligible_mask[top_gate_indices] = True
            return clean_selection_scores, anchor_pair_values, eligible_mask

        if mask2_mode == 'clean_tiebreak':
            attack_rank = torch.argsort(torch.argsort(attack_channel_scores))
            if attack_channel_scores.numel() > 1:
                attack_rank = attack_rank.float() / float(attack_channel_scores.numel() - 1)
            else:
                attack_rank = attack_rank.float()
            clean_span = (clean_selection_scores.max() - clean_selection_scores.min()).clamp_min(1e-8)
            tiebreak_scores = clean_selection_scores + (
                mask2_tiebreak_epsilon * clean_span * attack_rank)
            return tiebreak_scores, anchor_pair_values, eligible_mask

        clean_base_replay_latent = current_base_replay_latent if current_mask1_mode == 'clean' else None
        clean_replay_base_pair_values, clean_replay_anchor_pair_values, _ = self._build_channel_pair_values(
            latent_base,
            anchor_latents,
            pair_i,
            pair_j,
            replay_mode='clean',
            base_replay_latent=clean_base_replay_latent)
        clean_channel_scores = compute_base_cosine_channel_margin_scores(
            anchor_pair_values,
            clean_base_pair_values,
            clean_replay_anchor_pair_values,
            clean_replay_base_pair_values)

        return (
            torch.minimum(
                clean_channel_scores,
                mask2_attack_weight * attack_channel_scores),
            anchor_pair_values,
            eligible_mask)

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 val_mask1=0.32,
                 val_mask2=0.98,
                 add_cfg=16,
                 sector_margin_deg=15.0,
                 reference_mode='clean',
                 symbol_redundancy=1,
                 mask1_mode='clean',
                 mask1_attack_type='crf',
                 mask1_attack_crf=28,
                 mask2_mode='clean',
                 mask2_attack_type='crf',
                 mask2_attack_crf=28,
                 mask2_attack_weight=1.0,
                 mask2_attack_gate_quantile=0.2,
                 mask2_tiebreak_epsilon=1e-4,
                 embedding_strength=1.0,
                 channel_coding='repetition',
                 redundancy_schedule='uniform',
                 pair_selection_mode='full',
                 max_pairs=None,
                 k_mode=4):
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        symbol_redundancy = max(int(symbol_redundancy), 1)
        max_pairs = None if max_pairs is None else max(int(max_pairs), 0)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        time_str = ""
        first_word = input_prompt.split()[0] if input_prompt else ""

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latents = noise
            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = torch.stack([t])
                self.model.to(self.device)

                if i == sampling_steps - 1:
                    step_index = sample_scheduler._step_index
                    lower_order_nums = sample_scheduler.lower_order_nums

                    def generate_latent(noise_pred):
                        sample_scheduler._step_index = step_index
                        sample_scheduler.lower_order_nums = lower_order_nums
                        temp_x = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                        return temp_x.squeeze(0)

                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]
                    delta_v = noise_pred_cond - noise_pred_uncond
                    noise_pred_base = noise_pred_uncond + guide_scale * delta_v
                    latent_base = generate_latent(noise_pred_base)

                    current_time = datetime.datetime.now()
                    time_str = current_time.strftime("%m_%d_%H_%M")
                    video_file_name = f'video_cover/{first_word}_{time_str}.mp4'
                    test_rev, abs_diff1, mask1 = self._build_mask1(
                        latent_base,
                        val_mask1,
                        save_file=video_file_name,
                        mask1_mode=mask1_mode,
                        mask1_attack_type=mask1_attack_type,
                        mask1_attack_crf=mask1_attack_crf)

                    candidate_mask = mask1
                    if pair_selection_mode == 'mask2_only':
                        candidate_mask = torch.ones_like(mask1, dtype=torch.bool)

                    pair_i, pair_j = build_pair_indices(
                        candidate_mask, seed, self.device)
                    symbol_message = torch.empty(0, device=self.device, dtype=torch.long)
                    bit_message = expand_gray_symbols(symbol_message)
                    final_latent = latent_base
                    selection_scores = torch.empty(
                        0, device=self.device, dtype=torch.float32)

                    if pair_i.numel() > 0:
                        delta_v_flat = delta_v.flatten()
                        delta_v_float = delta_v_flat.float()
                        vi = delta_v_float[pair_i]
                        vj = delta_v_float[pair_j]
                        pair_radii = torch.sqrt(vi.square() + vj.square())
                        scaled_pair_radii = embedding_strength * pair_radii
                        orig_angles = torch.remainder(torch.atan2(vj, vi), 2 * math.pi)
                        anchor_latents = []
                        if k_mode == 2:
                            target_angles_0 = orig_angles
                            target_angles_1 = torch.remainder(orig_angles + math.pi, 2 * math.pi)
                            for target_angles in [target_angles_0, target_angles_1]:
                                rotated_delta = delta_v_flat.clone()
                                rotated_delta[pair_i] = (
                                    scaled_pair_radii * torch.cos(target_angles)).to(delta_v_flat.dtype)
                                rotated_delta[pair_j] = (
                                    scaled_pair_radii * torch.sin(target_angles)).to(delta_v_flat.dtype)
                                rotated_delta = rotated_delta.view_as(delta_v)
                                anchor_noise = noise_pred_uncond + guide_scale * rotated_delta
                                anchor_latent = generate_latent(anchor_noise)
                                anchor_latents.append(anchor_latent)
                        else:
                            sector_ranges = build_sector_ranges(sector_margin_deg)
                            for lower, upper in sector_ranges:
                                target_angles = nearest_safe_angles(orig_angles, lower, upper)
                                rotated_delta = delta_v_flat.clone()
                                rotated_delta[pair_i] = (
                                    scaled_pair_radii * torch.cos(target_angles)).to(delta_v_flat.dtype)
                                rotated_delta[pair_j] = (
                                    scaled_pair_radii * torch.sin(target_angles)).to(delta_v_flat.dtype)
                                rotated_delta = rotated_delta.view_as(delta_v)
                                anchor_noise = noise_pred_uncond + guide_scale * rotated_delta
                                anchor_latent = generate_latent(anchor_noise)
                                anchor_latents.append(anchor_latent)

                        if pair_selection_mode == 'mask1_only':
                            selection_scores = compute_mask1_pair_scores(
                                abs_diff1, pair_i, pair_j)
                            selection_eligible_mask = None
                        else:
                            selection_scores, _, selection_eligible_mask = self._compute_pair_selection_scores(
                                latent_base,
                                anchor_latents,
                                pair_i,
                                pair_j,
                                reference_mode=reference_mode,
                                mask2_mode=mask2_mode,
                                mask2_attack_type=mask2_attack_type,
                                mask2_attack_crf=mask2_attack_crf,
                                mask2_attack_weight=mask2_attack_weight,
                                mask2_attack_gate_quantile=mask2_attack_gate_quantile,
                                mask2_tiebreak_epsilon=mask2_tiebreak_epsilon,
                                current_mask1_mode=mask1_mode,
                                current_mask1_attack_type=mask1_attack_type,
                                current_mask1_attack_crf=mask1_attack_crf,
                                current_base_replay_latent=test_rev,
                                selection_keep_count=max_pairs)

                        selected_pair_positions = select_pair_positions(
                            selection_scores,
                            pair_selection_mode,
                            val_mask2,
                            max_pairs,
                            eligible_mask=selection_eligible_mask)
                        pair_i = pair_i[selected_pair_positions]
                        pair_j = pair_j[selected_pair_positions]
                        selection_scores = selection_scores[selected_pair_positions]
                        final_latent = latent_base
                        bit_message = torch.empty(0, device=self.device, dtype=torch.long)
                        payload_symbol_count = 0
                        transmit_symbols = torch.empty(0, device=self.device, dtype=torch.long)
                        symbol_group_ids = torch.empty(0, device=self.device, dtype=torch.long)
                        symbol_group_sizes = torch.empty(0, device=self.device, dtype=torch.long)
                        group_plan_stats = _empty_group_plan_stats()

                        if pair_i.numel() > 0:
                            grouped_pair_positions, symbol_group_ids, symbol_group_sizes, group_plan_stats = build_symbol_group_plan(
                                pair_i.numel(),
                                symbol_redundancy,
                                seed,
                                self.device,
                                redundancy_schedule=redundancy_schedule,
                                selection_scores=selection_scores,
                                pair_i=pair_i,
                                pair_j=pair_j,
                                latent_shape=target_shape)
                            if grouped_pair_positions.numel() == 0:
                                pair_i = pair_i[:0]
                                pair_j = pair_j[:0]
                                selection_scores = selection_scores[:0]

                        if pair_i.numel() > 0:
                            usable_symbol_count = usable_channel_symbol_count(
                                symbol_group_sizes.numel(), channel_coding)
                            if usable_symbol_count == 0:
                                pair_i = pair_i[:0]
                                pair_j = pair_j[:0]
                                selection_scores = selection_scores[:0]
                            elif usable_symbol_count < symbol_group_sizes.numel():
                                keep_pair_count = int(
                                    symbol_group_sizes[:usable_symbol_count].sum().item())
                                grouped_pair_positions = grouped_pair_positions[:keep_pair_count]
                                symbol_group_ids = symbol_group_ids[:keep_pair_count]
                                symbol_group_sizes = symbol_group_sizes[:usable_symbol_count]

                        if pair_i.numel() > 0:
                            pair_i = pair_i[grouped_pair_positions]
                            pair_j = pair_j[grouped_pair_positions]
                            selection_scores = selection_scores[grouped_pair_positions]

                        if pair_i.numel() > 0:
                            transmit_symbols, bit_message, payload_symbol_count = prepare_channel_message(
                                symbol_group_sizes.numel(),
                                seed,
                                self.device,
                                channel_coding,
                                k=k_mode)
                            repeated_symbol_message = transmit_symbols[symbol_group_ids]

                            final_latent_flat = latent_base.flatten().clone()
                            for state, anchor_latent in enumerate(anchor_latents):
                                state_mask = repeated_symbol_message == state
                                if not state_mask.any():
                                    continue
                                state_pair_i = pair_i[state_mask]
                                state_pair_j = pair_j[state_mask]
                                anchor_flat = anchor_latent.flatten()
                                final_latent_flat[state_pair_i] = anchor_flat[state_pair_i]
                                final_latent_flat[state_pair_j] = anchor_flat[state_pair_j]
                            final_latent = final_latent_flat.view_as(latent_base)

                    logging.info(
                        "K=%d sender mask1 dims=%d usable_pairs=%d embedded_bits=%d payload_symbols=%d redundancy=%d redundancy_schedule=%s selection_mode=%s mask1_mode=%s mask2_mode=%s embedding_strength=%.4f channel_coding=%s min_score=%.6f mean_score=%.6f time_overlap=%.6f space_overlap=%.6f channel_overlap=%.6f signature_overlap=%.6f",
                        int(k_mode),
                        int(mask1.sum().item()),
                        int(pair_i.numel()),
                        int(bit_message.numel()),
                        int(payload_symbol_count),
                        int(symbol_redundancy),
                        redundancy_schedule,
                        pair_selection_mode,
                        mask1_mode,
                        mask2_mode,
                        float(embedding_strength),
                        channel_coding,
                        float(selection_scores.min().item()) if selection_scores.numel() > 0 else 0.0,
                        float(selection_scores.mean().item()) if selection_scores.numel() > 0 else 0.0,
                        float(group_plan_stats['time_overlap']),
                        float(group_plan_stats['space_overlap']),
                        float(group_plan_stats['channel_overlap']),
                        float(group_plan_stats['signature_overlap']))

                    save_dir = "message"
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(bit_message.cpu(), os.path.join(save_dir, "message.pt"))
                    latents = [final_latent]
                    break

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0], time_str, first_word


 



    def receive(self, video_path, input_prompt, size=(1280, 720), frame_num=81, shift=5.0,
                sample_solver='unipc', sampling_steps=50, guide_scale=5.0, n_prompt="", seed=-1, offload_model=True, val_mask1=0.32,
                 val_mask2=0.98,
                 add_cfg=16,
                 sector_margin_deg=15.0,
                 reference_mode='clean',
                 symbol_redundancy=1,
                 mask1_mode='clean',
                 mask1_attack_type='crf',
                 mask1_attack_crf=28,
                 mask2_mode='clean',
                 mask2_attack_type='crf',
                 mask2_attack_crf=28,
                 mask2_attack_weight=1.0,
                 mask2_attack_gate_quantile=0.2,
                 mask2_tiebreak_epsilon=1e-4,
                 embedding_strength=1.0,
                 channel_coding='repetition',
                 redundancy_schedule='uniform',
                 pair_selection_mode='full',
                 repetition_decoder_mode='hard_sum',
                 repetition_gap_scale=1.0,
                 base_replay_mode='match_mask1',
                 max_pairs=None,
                 k_mode=4):
        my_video, _, _ = io.read_video(video_path, pts_unit='pts')
        video = my_video.permute(3, 0, 1, 2).unsqueeze(0).to(self.device).float()
        video = (video / 127.5) - 1
        x_prime = self.vae.encode(video)
        if isinstance(x_prime, list):
            x_prime = x_prime[0]

        observed_frame_num = int(my_video.shape[0])
        target_shape = tuple(int(dim) for dim in x_prime.shape)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size
        if observed_frame_num != frame_num:
            logging.info(
                "Receiver observed frame_num=%d from attacked video; overriding requested frame_num=%d for latent alignment.",
                observed_frame_num,
                int(frame_num),
            )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        symbol_redundancy = max(int(symbol_redundancy), 1)
        max_pairs = None if max_pairs is None else max(int(max_pairs), 0)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latents = noise
            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = torch.stack([t])
                self.model.to(self.device)

                if i == sampling_steps - 1:
                    step_index = sample_scheduler._step_index
                    lower_order_nums = sample_scheduler.lower_order_nums

                    def generate_latent(noise_pred):
                        sample_scheduler._step_index = step_index
                        sample_scheduler.lower_order_nums = lower_order_nums
                        temp_x = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                        return temp_x.squeeze(0)

                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]
                    delta_v = noise_pred_cond - noise_pred_uncond
                    noise_pred_base = noise_pred_uncond + guide_scale * delta_v
                    latent_base = generate_latent(noise_pred_base)
                    break

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

        test_rev, abs_diff1, mask1 = self._build_mask1(
            latent_base,
            val_mask1,
            mask1_mode=mask1_mode,
            mask1_attack_type=mask1_attack_type,
            mask1_attack_crf=mask1_attack_crf)

        resolved_base_replay_mode = mask1_mode
        if base_replay_mode != 'match_mask1':
            resolved_base_replay_mode = base_replay_mode

        base_test_rev = test_rev
        if (reference_mode == 'base_cosine' and
                resolved_base_replay_mode != mask1_mode):
            base_test_rev = self._replay_latent_through_video(
                latent_base,
                mask1_mode=resolved_base_replay_mode,
                mask1_attack_type=mask1_attack_type,
                mask1_attack_crf=mask1_attack_crf)

        candidate_mask = mask1
        if pair_selection_mode == 'mask2_only':
            candidate_mask = torch.ones_like(mask1, dtype=torch.bool)

        pair_i, pair_j = build_pair_indices(candidate_mask, seed, self.device)
        recovered_bits = torch.empty(0, device=self.device, dtype=torch.long)
        selection_scores = torch.empty(0, device=self.device, dtype=torch.float32)

        if pair_i.numel() > 0:
            delta_v_flat = delta_v.flatten()
            delta_v_float = delta_v_flat.float()
            vi = delta_v_float[pair_i]
            vj = delta_v_float[pair_j]
            pair_radii = torch.sqrt(vi.square() + vj.square())
            scaled_pair_radii = embedding_strength * pair_radii
            orig_angles = torch.remainder(torch.atan2(vj, vi), 2 * math.pi)
            anchor_latents = []
            if k_mode == 2:
                target_angles_0 = orig_angles
                target_angles_1 = torch.remainder(orig_angles + math.pi, 2 * math.pi)
                for target_angles in [target_angles_0, target_angles_1]:
                    rotated_delta = delta_v_flat.clone()
                    rotated_delta[pair_i] = (
                        scaled_pair_radii * torch.cos(target_angles)).to(delta_v_flat.dtype)
                    rotated_delta[pair_j] = (
                        scaled_pair_radii * torch.sin(target_angles)).to(delta_v_flat.dtype)
                    rotated_delta = rotated_delta.view_as(delta_v)
                    anchor_noise = noise_pred_uncond + guide_scale * rotated_delta
                    anchor_latent = generate_latent(anchor_noise)
                    anchor_latents.append(anchor_latent)
            else:
                sector_ranges = build_sector_ranges(sector_margin_deg)
                for lower, upper in sector_ranges:
                    target_angles = nearest_safe_angles(orig_angles, lower, upper)
                    rotated_delta = delta_v_flat.clone()
                    rotated_delta[pair_i] = (
                        scaled_pair_radii * torch.cos(target_angles)).to(delta_v_flat.dtype)
                    rotated_delta[pair_j] = (
                        scaled_pair_radii * torch.sin(target_angles)).to(delta_v_flat.dtype)
                    rotated_delta = rotated_delta.view_as(delta_v)
                    anchor_noise = noise_pred_uncond + guide_scale * rotated_delta
                    anchor_latent = generate_latent(anchor_noise)
                    anchor_latents.append(anchor_latent)

            if pair_selection_mode == 'mask1_only':
                replay_anchor_pair_values = build_anchor_pair_values(
                    anchor_latents, pair_i, pair_j)
                selection_scores = compute_mask1_pair_scores(
                    abs_diff1, pair_i, pair_j)
                selection_eligible_mask = None
            else:
                selection_scores, replay_anchor_pair_values, selection_eligible_mask = self._compute_pair_selection_scores(
                    latent_base,
                    anchor_latents,
                    pair_i,
                    pair_j,
                    reference_mode=reference_mode,
                    mask2_mode=mask2_mode,
                    mask2_attack_type=mask2_attack_type,
                    mask2_attack_crf=mask2_attack_crf,
                    mask2_attack_weight=mask2_attack_weight,
                    mask2_attack_gate_quantile=mask2_attack_gate_quantile,
                    mask2_tiebreak_epsilon=mask2_tiebreak_epsilon,
                    current_mask1_mode=mask1_mode,
                    current_mask1_attack_type=mask1_attack_type,
                    current_mask1_attack_crf=mask1_attack_crf,
                    current_base_replay_latent=test_rev,
                    selection_keep_count=max_pairs)

            selected_pair_positions = select_pair_positions(
                selection_scores,
                pair_selection_mode,
                val_mask2,
                max_pairs,
                eligible_mask=selection_eligible_mask)
            pair_i = pair_i[selected_pair_positions]
            pair_j = pair_j[selected_pair_positions]
            selection_scores = selection_scores[selected_pair_positions]
            replay_anchor_pair_values = replay_anchor_pair_values[:, selected_pair_positions]
            symbol_group_ids = torch.empty(0, device=self.device, dtype=torch.long)
            symbol_group_sizes = torch.empty(0, device=self.device, dtype=torch.long)
            group_plan_stats = _empty_group_plan_stats()

            if pair_i.numel() > 0:
                grouped_pair_positions, symbol_group_ids, symbol_group_sizes, group_plan_stats = build_symbol_group_plan(
                    pair_i.numel(),
                    symbol_redundancy,
                    seed,
                    self.device,
                    redundancy_schedule=redundancy_schedule,
                    selection_scores=selection_scores,
                    pair_i=pair_i,
                    pair_j=pair_j,
                    latent_shape=target_shape)
                if grouped_pair_positions.numel() == 0:
                    pair_i = pair_i[:0]
                    pair_j = pair_j[:0]
                    selection_scores = selection_scores[:0]
                    replay_anchor_pair_values = replay_anchor_pair_values[:, :0]

            if pair_i.numel() > 0:
                usable_symbol_count = usable_channel_symbol_count(
                    symbol_group_sizes.numel(), channel_coding)
                if usable_symbol_count == 0:
                    pair_i = pair_i[:0]
                    pair_j = pair_j[:0]
                    selection_scores = selection_scores[:0]
                    replay_anchor_pair_values = replay_anchor_pair_values[:, :0]
                elif usable_symbol_count < symbol_group_sizes.numel():
                    keep_pair_count = int(
                        symbol_group_sizes[:usable_symbol_count].sum().item())
                    grouped_pair_positions = grouped_pair_positions[:keep_pair_count]
                    symbol_group_ids = symbol_group_ids[:keep_pair_count]
                    symbol_group_sizes = symbol_group_sizes[:usable_symbol_count]

            if pair_i.numel() > 0:
                pair_i = pair_i[grouped_pair_positions]
                pair_j = pair_j[grouped_pair_positions]
                selection_scores = selection_scores[grouped_pair_positions]
                replay_anchor_pair_values = replay_anchor_pair_values[:, grouped_pair_positions]

            recovered_bits = torch.empty(0, device=self.device, dtype=torch.long)
            payload_symbol_count = 0
            decision_gaps = torch.empty(0, device=self.device, dtype=torch.float32)

            if pair_i.numel() > 0:
                observed_pair_values = build_pair_values(x_prime, pair_i, pair_j)
                if reference_mode == 'base_cosine':
                    clean_base_pair_values = build_pair_values(
                        latent_base, pair_i, pair_j)
                    replay_base_pair_values = build_pair_values(
                        base_test_rev, pair_i, pair_j)
                    reference_offsets = (
                        replay_anchor_pair_values -
                        clean_base_pair_values.unsqueeze(0))
                    observed_offsets = (
                        observed_pair_values - replay_base_pair_values)

                    reference_offset_norms = torch.linalg.norm(
                        reference_offsets, dim=-1).clamp_min(1e-8)
                    observed_offset_norms = torch.linalg.norm(
                        observed_offsets, dim=-1).clamp_min(1e-8)
                    cosine_scores = torch.sum(
                        reference_offsets * observed_offsets.unsqueeze(0),
                        dim=-1) / (
                            reference_offset_norms *
                            observed_offset_norms.unsqueeze(0))
                    pair_scores = cosine_scores
                else:
                    pair_distances = torch.sum(
                        torch.abs(replay_anchor_pair_values - observed_pair_values.unsqueeze(0)),
                        dim=-1)
                    pair_scores = -pair_distances

                grouped_scores, pair_weights, grouped_bit_llrs = aggregate_channel_scores(
                    pair_scores,
                    symbol_group_ids,
                    symbol_group_sizes,
                    channel_coding,
                    repetition_decoder_mode=repetition_decoder_mode,
                    repetition_gap_scale=repetition_gap_scale)
                recovered_bits, payload_symbol_count, decision_gaps = decode_channel_scores(
                    grouped_scores,
                    channel_coding,
                    repetition_decoder_mode=repetition_decoder_mode,
                    grouped_bit_llrs=grouped_bit_llrs,
                    k=k_mode)

                pair_weight_mean = 1.0
                pair_weight_max = 1.0
                if pair_weights is not None and pair_weights.numel() > 0:
                    pair_weight_mean = float(pair_weights.mean().item())
                    pair_weight_max = float(pair_weights.max().item())

                logging.info(
                    "K=%d receiver mask1 dims=%d usable_pairs=%d recovered_bits=%d payload_symbols=%d redundancy=%d redundancy_schedule=%s selection_mode=%s mask1_mode=%s mask2_mode=%s embedding_strength=%.4f channel_coding=%s repetition_decoder_mode=%s repetition_gap_scale=%.4f base_replay_mode=%s min_score=%.6f mean_score=%.6f gap_mean=%.6f gap_p10=%.6f gap_min=%.6f pair_weight_mean=%.6f pair_weight_max=%.6f time_overlap=%.6f space_overlap=%.6f channel_overlap=%.6f signature_overlap=%.6f",
                    int(k_mode),
                    int(mask1.sum().item()),
                    int(pair_i.numel()),
                    int(recovered_bits.numel()),
                    int(payload_symbol_count),
                    int(symbol_redundancy),
                    redundancy_schedule,
                    pair_selection_mode,
                    mask1_mode,
                    mask2_mode,
                    float(embedding_strength),
                    channel_coding,
                    repetition_decoder_mode,
                    float(repetition_gap_scale),
                    resolved_base_replay_mode,
                    float(selection_scores.min().item()) if selection_scores.numel() > 0 else 0.0,
                    float(selection_scores.mean().item()) if selection_scores.numel() > 0 else 0.0,
                    float(decision_gaps.mean().item()),
                    float(torch.quantile(decision_gaps, 0.1).item()),
                    float(decision_gaps.min().item()),
                    pair_weight_mean,
                    pair_weight_max,
                    float(group_plan_stats['time_overlap']),
                    float(group_plan_stats['space_overlap']),
                    float(group_plan_stats['channel_overlap']),
                    float(group_plan_stats['signature_overlap']))

        return recovered_bits