"""Batch-aware multi-GPU support for StereoDETR.

PyTorch's default ``DataParallel`` scatter treats a Python list as a generic
container.  StereoDETR, however, passes targets as a batch-aligned list of
dictionaries.  Replicating that list on every GPU silently breaks the
relationship between images and targets.  This module keeps the existing
single-process training entry point while splitting every batch-aligned input
with the same chunk plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable, Sequence

import torch


def parse_gpu_ids(value: str | Iterable[int]) -> list[int]:
    """Parse and validate visible CUDA device indices."""
    if isinstance(value, str):
        fields = [field.strip() for field in value.split(",")]
        if not fields or any(not field for field in fields):
            raise ValueError("trainer.gpu_ids must contain CUDA indices")
        gpu_ids = [int(field) for field in fields]
    else:
        gpu_ids = [int(device_id) for device_id in value]
    if not gpu_ids:
        raise ValueError("at least one CUDA device is required")
    if any(device_id < 0 for device_id in gpu_ids):
        raise ValueError("CUDA device indices must be non-negative")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("trainer.gpu_ids must not contain duplicates")
    return gpu_ids


def balanced_chunk_sizes(batch_size: int, num_devices: int) -> list[int]:
    """Return non-empty, DataParallel-compatible batch chunks."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_devices <= 0:
        raise ValueError("num_devices must be positive")
    active_devices = min(batch_size, num_devices)
    quotient, remainder = divmod(batch_size, active_devices)
    return [
        quotient + (1 if index < remainder else 0)
        for index in range(active_devices)
    ]


def _move_to_device(value: Any, device_id: int | None) -> Any:
    if torch.is_tensor(value):
        if device_id is None:
            return value
        return value.to(
            device=torch.device("cuda", device_id),
            non_blocking=True,
        )
    if isinstance(value, Mapping):
        return type(value)(
            (key, _move_to_device(item, device_id))
            for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device_id) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device_id) for item in value]
    return value


def split_batch_value(
    value: Any,
    batch_size: int,
    chunk_sizes: Sequence[int],
    device_ids: Sequence[int | None],
) -> list[Any]:
    """Split one nested argument according to its leading batch dimension."""
    if len(chunk_sizes) != len(device_ids):
        raise ValueError("chunk_sizes and device_ids must have equal length")

    if torch.is_tensor(value):
        if value.ndim > 0 and value.shape[0] == batch_size:
            chunks = value.split(tuple(chunk_sizes), dim=0)
            return [
                _move_to_device(chunk, device_id)
                for chunk, device_id in zip(chunks, device_ids)
            ]
        return [_move_to_device(value, device_id) for device_id in device_ids]

    # Training targets are specifically a list[dict] with one dictionary per
    # image.  Slice the outer list first, then transfer each nested tensor.
    if (
        isinstance(value, list)
        and len(value) == batch_size
        and (not value or all(isinstance(item, Mapping) for item in value))
    ):
        chunks = []
        start = 0
        for chunk_size, device_id in zip(chunk_sizes, device_ids):
            stop = start + chunk_size
            chunks.append(_move_to_device(value[start:stop], device_id))
            start = stop
        return chunks

    if isinstance(value, Mapping):
        split_items = {
            key: split_batch_value(
                item,
                batch_size,
                chunk_sizes,
                device_ids,
            )
            for key, item in value.items()
        }
        return [
            type(value)(
                (key, split_items[key][device_index])
                for key in value
            )
            for device_index in range(len(device_ids))
        ]

    if isinstance(value, tuple):
        split_items = [
            split_batch_value(
                item,
                batch_size,
                chunk_sizes,
                device_ids,
            )
            for item in value
        ]
        return [
            tuple(item[device_index] for item in split_items)
            for device_index in range(len(device_ids))
        ]

    if isinstance(value, list):
        split_items = [
            split_batch_value(
                item,
                batch_size,
                chunk_sizes,
                device_ids,
            )
            for item in value
        ]
        return [
            [item[device_index] for item in split_items]
            for device_index in range(len(device_ids))
        ]

    return [value for _ in device_ids]


class StereoBatchDataParallel(torch.nn.DataParallel):
    """DataParallel that correctly partitions StereoDETR target structures."""

    def scatter(self, inputs, kwargs, device_ids):
        if not inputs or not torch.is_tensor(inputs[0]):
            return super().scatter(inputs, kwargs, device_ids)
        batch_size = int(inputs[0].shape[0])
        chunk_sizes = balanced_chunk_sizes(batch_size, len(device_ids))
        active_device_ids = list(device_ids[: len(chunk_sizes)])

        split_inputs = [
            split_batch_value(
                value,
                batch_size,
                chunk_sizes,
                active_device_ids,
            )
            for value in inputs
        ]
        scattered_inputs = tuple(
            tuple(parts[index] for parts in split_inputs)
            for index in range(len(active_device_ids))
        )

        split_kwargs = split_batch_value(
            kwargs or {},
            batch_size,
            chunk_sizes,
            active_device_ids,
        )
        return scattered_inputs, tuple(split_kwargs)


def build_parallel_model(
    model: torch.nn.Module,
    gpu_ids: Sequence[int],
) -> torch.nn.Module:
    """Move a model to one GPU or wrap it in batch-aware DataParallel."""
    gpu_ids = parse_gpu_ids(gpu_ids)
    primary_device = torch.device("cuda", gpu_ids[0])
    model = model.to(primary_device)
    if len(gpu_ids) == 1:
        return model
    return StereoBatchDataParallel(
        model,
        device_ids=list(gpu_ids),
        output_device=gpu_ids[0],
    )
