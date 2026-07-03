from collections import OrderedDict
import os.path as osp

import torch
import torch.nn.functional as F


def _is_wrapped(module):
    return hasattr(module, "module") and module.__class__.__name__ in {"DataParallel", "DistributedDataParallel"}


def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(child_module, prefix=""):
        if _is_wrapped(child_module):
            child_module = child_module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        child_module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, submodule in child_module._modules.items():
            if submodule is not None:
                load(submodule, prefix + name + ".")

    load(module)
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append("unexpected key in source state_dict: {}\n".format(", ".join(unexpected_keys)))
    if missing_keys:
        err_msg.append("missing keys in source state_dict: {}\n".format(", ".join(missing_keys)))

    if strict and err_msg:
        message = "The model and loaded state dict do not match exactly\n" + "\n".join(err_msg)
        if logger is not None:
            logger.warning(message)
        raise RuntimeError(message)


def _load_checkpoint(filename, map_location="cpu"):
    if not osp.isfile(filename):
        raise IOError(f"{filename} is not a checkpoint file")
    return torch.load(filename, map_location=map_location)


def _warn(logger, message):
    if logger is not None:
        logger.warning(message)
    else:
        print(message)


def load_checkpoint(model, filename, map_location="cpu", strict=False, logger=None):
    checkpoint = _load_checkpoint(filename, map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("backbone."):
        print("Start stripping upper net pre-fix and loading backbone weights to our swin encoder")
        state_dict = {k.replace("backbone.", "", 1): v for k, v in state_dict.items() if k.startswith("backbone.")}
    first_key = sorted(state_dict.keys())[0]
    if first_key.startswith("encoder"):
        state_dict = {k.replace("encoder.", "", 1): v for k, v in state_dict.items() if k.startswith("encoder.")}

    if state_dict.get("absolute_pos_embed") is not None and hasattr(model, "absolute_pos_embed"):
        absolute_pos_embed = state_dict["absolute_pos_embed"]
        n1, length, c1 = absolute_pos_embed.size()
        n2, c2, h, w = model.absolute_pos_embed.size()
        if n1 != n2 or c1 != c2 or length != h * w:
            _warn(logger, "Error in loading absolute_pos_embed, pass")
        else:
            state_dict["absolute_pos_embed"] = absolute_pos_embed.view(n2, h, w, c2).permute(0, 3, 1, 2)

    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    current_state = model.state_dict()
    for table_key in relative_position_bias_table_keys:
        if table_key not in current_state:
            continue
        table_pretrained = state_dict[table_key]
        table_current = current_state[table_key]
        l1, n_h1 = table_pretrained.size()
        l2, n_h2 = table_current.size()
        if n_h1 != n_h2:
            _warn(logger, f"Error in loading {table_key}, pass")
            continue
        if l1 != l2:
            s1 = int(l1 ** 0.5)
            s2 = int(l2 ** 0.5)
            resized = F.interpolate(
                table_pretrained.permute(1, 0).view(1, n_h1, s1, s1),
                size=(s2, s2),
                mode="bicubic",
            )
            state_dict[table_key] = resized.view(n_h2, l2).permute(1, 0)

    load_state_dict(model, state_dict, strict=strict, logger=logger)
    return checkpoint


def weights_to_cpu(state_dict):
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu
