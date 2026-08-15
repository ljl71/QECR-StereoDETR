import os
import torch
import torch.nn as nn


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None, best_result=None, best_epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state, 'best_result': best_result, 'best_epoch': best_epoch}


def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def _has_allowed_prefix(name, prefixes):
    return any(name.startswith(prefix) for prefix in prefixes)


def load_checkpoint(
    model,
    optimizer,
    filename,
    map_location,
    logger=None,
    strict=True,
    allowed_missing_prefixes=None,
    allow_unexpected=True,
):
    if os.path.isfile(filename):
        if logger is not None:
            logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location)
        epoch = checkpoint.get('epoch', -1)
        best_result = checkpoint.get('best_result', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            incompatible = model.load_state_dict(
                checkpoint['model_state'], strict=strict
            )
            if logger is not None and not strict:
                if incompatible.missing_keys:
                    logger.info(
                        "Pretrain missing keys (expected for new heads): %s",
                        incompatible.missing_keys,
                    )
                if incompatible.unexpected_keys:
                    logger.info(
                        "Pretrain unexpected keys: %s",
                        incompatible.unexpected_keys,
                    )
            if not strict and allowed_missing_prefixes is not None:
                allowed_missing_prefixes = tuple(
                    str(prefix) for prefix in allowed_missing_prefixes
                )
                invalid_missing = [
                    key for key in incompatible.missing_keys
                    if not _has_allowed_prefix(
                        key, allowed_missing_prefixes
                    )
                ]
                if invalid_missing:
                    raise RuntimeError(
                        "checkpoint has disallowed missing keys: {}".format(
                            invalid_missing
                        )
                    )
                if incompatible.unexpected_keys and not allow_unexpected:
                    raise RuntimeError(
                        "checkpoint has unexpected keys: {}".format(
                            incompatible.unexpected_keys
                        )
                    )
                if logger is not None:
                    logger.info(
                        "Checkpoint compatibility whitelist passed; "
                        "allowed missing prefixes: %s",
                        allowed_missing_prefixes,
                    )
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if logger is not None:
            logger.info("==> Done")
    else:
        raise FileNotFoundError

    return epoch, best_result, best_epoch
