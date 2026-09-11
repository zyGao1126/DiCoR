import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from args import offline_bank_parser
from data.dataloader_util import colllate_fn_custom
from engine import (
    batch_to_device,
    foreground_iou_from_logits,
    load_model_weights,
    make_dataset,
    model_cfg,
    parse_epochs,
    resolve_device,
)
from lib import segmentation


class DLGInputDataset(Dataset):
    """Image-and-text inputs used to extract the frozen DLG features."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.get_dlg_input(index)


class DLGFeatureWriter:
    def __init__(self, output_dir: Path, shard_size: int):
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.output_dir = output_dir
        self.shard_size = int(shard_size)
        self.pending = []
        self.pending_rows = 0
        self.shard_count = 0

        output_dir.mkdir(parents=True, exist_ok=True)
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use a new directory to avoid mixing different cache versions."
            )

    def add(self, features: torch.Tensor) -> None:
        features = features.detach().to(device="cpu", dtype=torch.float16).contiguous()
        self.pending.append(features)
        self.pending_rows += int(features.shape[0])
        while self.pending_rows >= self.shard_size:
            self._flush(self.shard_size)

    def _flush(self, rows: int) -> None:
        merged = torch.cat(self.pending, dim=0)
        shard = merged[:rows].contiguous()
        self.pending = [merged[rows:].contiguous()] if merged.shape[0] > rows else []
        self.pending_rows -= rows

        filename = f"shard_{self.shard_count:04d}.pt"
        path = self.output_dir / filename
        temporary = path.with_suffix(".pt.tmp")
        torch.save(shard, temporary)
        os.replace(temporary, path)
        self.shard_count += 1

    def finish(self):
        if self.pending_rows:
            self._flush(self.pending_rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def file_record(path: str) -> dict:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def snapshot_path(coarse_dir: str, epoch: int) -> str:
    path = os.path.join(coarse_dir, f"coarse_ep{epoch}.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing coarse snapshot: {path}")
    return path


def load_exact_weights(model, checkpoint: str, label: str) -> None:
    incompatible = load_model_weights(model, checkpoint, label=label)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{label} does not match the current model: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def model_record(args) -> dict:
    return {
        "dataset": args.dataset,
        "img_size": int(args.img_size),
        "swin_type": args.swin_type,
        "window12": bool(args.window12),
        "num_vmsf_blocks": int(args.num_vmsf_blocks),
        "num_heads_fusion": int(args.num_heads_fusion),
        "visual_fusion": args.visual_fusion,
    }


@torch.no_grad()
def write_lcr_snapshot(model, loader, device, output_path: Path, iou_range, label: str) -> dict:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    probability_map = None
    map_shape = None
    kept = 0

    try:
        for data in tqdm(loader, desc=f"[OfflineBank] {label}", dynamic_ncols=True):
            batch = batch_to_device(data, device)
            output = model(batch["image"], batch["text"], l_mask=batch["l_mask"])
            logits = output["coarse_logits_120"]

            if probability_map is None:
                map_shape = tuple(int(value) for value in logits.shape[-2:])
                probability_map = np.memmap(
                    temporary,
                    mode="w+",
                    dtype=np.uint8,
                    shape=(len(loader.dataset), *map_shape),
                )

            probability = torch.softmax(logits, dim=1)[:, 1]
            iou = foreground_iou_from_logits(logits, batch["target"], map_shape)
            valid = (iou >= iou_range[0]) & (iou <= iou_range[1])
            probability = probability * valid.float().view(-1, 1, 1)

            indices = data["index"].long().cpu().numpy()
            values = np.clip(np.round(probability.detach().cpu().numpy() * 255.0), 0, 255.0).astype(np.uint8)
            probability_map[indices] = values
            kept += int(valid.sum().item())

        probability_map.flush()
        del probability_map
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    print(f"[OfflineBank] {label}: kept {kept}/{len(loader.dataset)}, wrote {output_path}")
    return {
        "file": output_path.name,
        "kept": kept,
        "height": map_shape[0],
        "width": map_shape[1],
    }


def build_lcr_bank(args, device: torch.device) -> None:
    if not args.coarse_dir:
        raise ValueError("--coarse-dir is required")

    output_dir = Path(args.output_dir).resolve() / "LCR"
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new directory to avoid mixing different cache versions."
        )

    dataset = make_dataset(args, "train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=colllate_fn_custom,
    )
    model = segmentation.dicor_coarse(
        pretrained=args.pretrained_swin_weights,
        pretrained_refineHead="",
        args=args,
        cfg=model_cfg(visual_fusion=args.visual_fusion),
    ).to(device)
    model.eval()

    snapshots = []
    for epoch in parse_epochs(args.snapshot_epochs):
        checkpoint = snapshot_path(args.coarse_dir, epoch)
        load_exact_weights(model, checkpoint, label=f"LCR snapshot ep{epoch}")
        record = write_lcr_snapshot(
            model,
            loader,
            device,
            output_dir / f"ep{epoch}.mmap",
            args.lcr_iou_range,
            label=f"LCR ep{epoch}",
        )
        map_height = record.pop("height")
        map_width = record.pop("width")
        record.update({"epoch": epoch, "checkpoint": file_record(checkpoint)})
        snapshots.append(record)

    if not snapshots:
        raise ValueError("--snapshot-epochs must contain at least one epoch")

    manifest = {
        "kind": "refiner_probability",
        "sample_count": len(dataset),
        "height": map_height,
        "width": map_width,
        "iou_range": list(args.lcr_iou_range),
        "annotation_file": file_record(dataset.ann_path),
        "model": model_record(args),
        "snapshots": snapshots,
    }
    write_json(output_dir / "manifest.json", manifest)


@torch.no_grad()
def write_dlg_features(model, loader, writer, device: torch.device, label: str) -> int:
    cached_rows = 0

    for data in tqdm(loader, desc=f"[OfflineBank] DLG {label}", dynamic_ncols=True):
        indices = data["index"].long().cpu()
        expected = torch.arange(cached_rows, cached_rows + indices.numel(), dtype=indices.dtype)
        if not torch.equal(indices, expected):
            raise RuntimeError(
                "Dataset order changed while caching DLG: expected rows "
                f"{cached_rows}-{cached_rows + indices.numel() - 1}, got {indices.tolist()}"
            )

        image = data["image"].to(device=device, non_blocking=True)
        text = data["tensor_embeddings"].to(device=device, non_blocking=True).squeeze(1)
        language_mask = data["attention_mask"].to(device=device, non_blocking=True)
        output = model(image, text, l_mask=language_mask)
        features = output["x_pre_c3_star"]

        writer.add(features)
        cached_rows += int(features.shape[0])

    print(f"[OfflineBank] DLG {label}: cached {cached_rows} samples")
    return cached_rows


def build_dlg_bank(args, device: torch.device) -> None:
    if not os.path.isfile(args.coarse_ckpt):
        raise FileNotFoundError(f"Missing coarse checkpoint: {args.coarse_ckpt}")

    checkpoints = (
        ("best", args.coarse_ckpt),
        ("ep30", snapshot_path(args.coarse_dir, 30)),
    )

    dataset = make_dataset(args, "train")
    input_dataset = DLGInputDataset(dataset)
    loader = DataLoader(
        input_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(input_dataset),
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=colllate_fn_custom,
    )
    output_dir = Path(args.output_dir).resolve() / "DLG"
    writer = DLGFeatureWriter(output_dir, 256)

    model = segmentation.dicor_coarse(
        pretrained=args.pretrained_swin_weights,
        pretrained_refineHead="",
        args=args,
        cfg=model_cfg(visual_fusion=args.visual_fusion),
    ).to(device)
    model.eval()
    model.requires_grad_(False)

    snapshots = []
    row_start = 0
    for name, checkpoint in checkpoints:
        load_exact_weights(model, checkpoint, label=f"DLG {name}")
        row_stop = row_start + write_dlg_features(model, loader, writer, device, name)
        snapshots.append(
            {
                "name": name,
                "row_start": row_start,
                "row_stop": row_stop,
                "checkpoint": file_record(checkpoint),
            }
        )
        row_start = row_stop

    writer.finish()
    manifest = {
        "kind": "dlg_feature",
        "sample_count": len(dataset),
        "total_rows": row_start,
        "annotation_file": file_record(dataset.ann_path),
        "model": model_record(args),
        "snapshots": snapshots,
    }
    write_json(output_dir / "manifest.json", manifest)


def main() -> None:
    args = offline_bank_parser().parse_args()
    device = resolve_device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    start = time.time()
    if args.bank_type in ("dlg", "all"):
        build_dlg_bank(args, device)
    if args.bank_type in ("lcr", "all"):
        build_lcr_bank(args, device)
    print(f"[OfflineBank] finished in {(time.time() - start) / 3600:.2f}h")


if __name__ == "__main__":
    main()
