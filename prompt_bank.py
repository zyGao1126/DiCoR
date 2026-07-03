import os
import re

import numpy as np
import torch


PROMPT_BANK_SCALE = 255.0
DEFAULT_PROMPT_HW = (120, 120)


class PromptBank:
    """Uint8 offline prompt bank backed by memmap snapshots."""

    def __init__(self, bank_dir: str, split: str):
        self.bank_dir = os.path.join(bank_dir, split)
        self.split = split
        self.N, self.H, self.W, self.snapshots = self._scan_snapshots()
        self._mms = [None] * len(self.snapshots)

        if len(self.snapshots) == 0:
            raise ValueError(f"PromptBank split '{split}' has no snapshots")

    def _scan_snapshots(self):
        files = []
        for name in os.listdir(self.bank_dir):
            if name.startswith("ep") and name.endswith(".mmap"):
                match = re.search(r"ep(\d+)", name)
                epoch = int(match.group(1)) if match else 0
                files.append((epoch, name))

        files.sort(key=lambda item: (item[0], item[1]))
        if not files:
            raise FileNotFoundError(f"No prompt snapshots found in {self.bank_dir}; expected files like ep10.mmap")

        h, w = DEFAULT_PROMPT_HW
        sample_count = None
        snapshots = []
        for epoch, name in files:
            path = os.path.join(self.bank_dir, name)
            size = os.path.getsize(path)
            denom = h * w * np.dtype(np.uint8).itemsize
            if size % denom != 0:
                raise ValueError(f"Cannot infer shape for {path}: file size {size} is not divisible by {h}*{w}")
            n = size // denom
            if sample_count is None:
                sample_count = n
            elif n != sample_count:
                raise ValueError(f"PromptBank mmap size mismatch: {name} has N={n}, expected N={sample_count}")
            snapshots.append({"name": f"ep{epoch}", "file": name, "kept": None})

        print(f"[PromptBank] loaded {self.split}: {len(snapshots)} snapshots from {self.bank_dir}, shape=({sample_count}, {h}, {w})")
        return int(sample_count), h, w, snapshots

    def __len__(self) -> int:
        return len(self.snapshots)

    def _open_mm(self, sid: int):
        if self._mms[sid] is not None:
            return self._mms[sid]

        path = os.path.join(self.bank_dir, self.snapshots[sid]["file"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"PromptBank snapshot file missing: {path}")

        mm = np.memmap(path, mode="r", dtype=np.uint8, shape=(self.N, self.H, self.W))
        self._mms[sid] = mm
        return mm

    @torch.no_grad()
    def get_batch(self, indices: torch.Tensor, sids: torch.Tensor, device: torch.device) -> torch.Tensor:
        idx_cpu = indices.detach().cpu().long()
        sid_cpu = sids.detach().cpu().long()
        out = torch.empty((idx_cpu.numel(), 1, self.H, self.W), dtype=torch.float32)

        for sid in torch.unique(sid_cpu).tolist():
            mm = self._open_mm(int(sid))
            sel = sid_cpu == int(sid)
            arr = mm[idx_cpu[sel].numpy()]
            out[sel] = torch.from_numpy(arr.astype(np.float32) / PROMPT_BANK_SCALE).unsqueeze(1)

        return out.to(device=device, non_blocking=True)

    def has_valid_prompts(self) -> bool:
        for sid in range(len(self.snapshots)):
            mm = self._open_mm(sid)
            if np.any(np.asarray(mm).reshape(self.N, -1).sum(axis=1) > 0):
                return True
        return False
