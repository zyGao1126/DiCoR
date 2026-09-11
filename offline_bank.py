from pathlib import Path

import numpy as np
import torch


class LCRProbabilityBank:
    """Memory-mapped coarse probabilities used to train the LCR refiner."""

    def __init__(self, offline_bank_dir: str, map_size: int):
        self.root = Path(offline_bank_dir).resolve() / "LCR"
        self.snapshots = tuple(sorted(self.root.glob("*.mmap")))
        if not self.snapshots:
            raise FileNotFoundError(f"No LCR probability maps found in {self.root}")

        self.height = self.width = int(map_size)
        self.sample_count = self.snapshots[0].stat().st_size // (self.height * self.width)
        self._maps = [None] * len(self.snapshots)
        print(
            f"[LCR] loaded {len(self.snapshots)} probability snapshots from "
            f"{self.root}, shape=({self.sample_count}, {self.height}, {self.width})"
        )

    @property
    def num_snapshots(self) -> int:
        return len(self.snapshots)

    def _open_snapshot(self, snapshot_index: int):
        if self._maps[snapshot_index] is None:
            self._maps[snapshot_index] = np.memmap(
                self.snapshots[snapshot_index],
                mode="r",
                dtype=np.uint8,
                shape=(self.sample_count, self.height, self.width),
            )
        return self._maps[snapshot_index]

    @torch.no_grad()
    def load_probabilities(self, sample_indices: torch.Tensor, snapshot_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        sample_indices = sample_indices.detach().cpu().long()
        snapshot_indices = snapshot_indices.detach().cpu().long()
        probabilities = torch.empty(
            (sample_indices.numel(), 1, self.height, self.width), dtype=torch.float32
        )
        for snapshot_index in torch.unique(snapshot_indices).tolist():
            selected = snapshot_indices == snapshot_index
            values = self._open_snapshot(snapshot_index)[sample_indices[selected].numpy()]
            probabilities[selected] = torch.from_numpy(values.astype(np.float32) / 255.0).unsqueeze(1)
        return probabilities.to(device=device, non_blocking=True)


class DLGFeatureBank:
    """Shard-backed coarse visual features used to train DLG."""

    def __init__(self, offline_bank_dir: str, sample_count: int):
        self.root = Path(offline_bank_dir).resolve() / "DLG"
        self.shards = tuple(sorted(self.root.glob("shard_*.pt")))
        if not self.shards:
            raise FileNotFoundError(f"No DLG feature shards found in {self.root}")

        first_shard = self.load_shard(self.shards[0])
        self.shard_size = int(first_shard.shape[0])
        self.feature_shape = tuple(int(value) for value in first_shard.shape[1:])
        del first_shard

        last_shard_rows = self.shard_size
        if len(self.shards) > 1:
            last_shard_rows = int(self.load_shard(self.shards[-1]).shape[0])
        self.total_rows = self.shard_size * (len(self.shards) - 1) + last_shard_rows
        self.sample_count = int(sample_count)
        if self.total_rows % self.sample_count:
            raise ValueError(f"DLG contains {self.total_rows} rows for {self.sample_count} training samples")
        self.num_snapshots = self.total_rows // self.sample_count
        self.snapshot_offsets = tuple(
            index * self.sample_count
            for index in range(self.num_snapshots)
        )

        print(
            f"[DLG] loaded {self.num_snapshots} snapshots and {self.total_rows} feature maps "
            f"from {self.root}, shape={self.feature_shape}"
        )

    @staticmethod
    def load_shard(shard: Path) -> torch.Tensor:
        return torch.load(shard, map_location="cpu", weights_only=True)
