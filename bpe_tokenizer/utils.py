import os
from time import time
from dataclasses import dataclass, field


def find_chunk_boundaries(file, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


@dataclass
class MergeStats:
    merge_index: int
    pair: tuple[int, int]
    new_token: bytes
    pair_count: int
    merge_time: float
    total_elapsed: float
    vocab_size: int
    num_active_pairs: int


class TrainingTracker:
    def __init__(self, wandb_run=None):
        self.wandb_run = wandb_run
        self.start_time: float = 0.0
        self.merge_start_time: float = 0.0

    def on_train_start(self):
        self.start_time = time()

    def on_merge_start(self):
        self.merge_start_time = time()

    def on_merge_end(self, stats: MergeStats):
        if self.wandb_run is None:
            return
        self.wandb_run.log({
            "merge/index": stats.merge_index,
            "merge/pair_count": stats.pair_count,
            "merge/time_seconds": stats.merge_time,
            "merge/total_elapsed_seconds": stats.total_elapsed,
            "merge/vocab_size": stats.vocab_size,
            "merge/active_pairs": stats.num_active_pairs,
            "merge/new_token": stats.new_token.decode("utf-8", errors="replace"),
        })

    def on_train_end(self, total_merges: int):
        total_time = time() - self.start_time
        if self.wandb_run is not None:
            self.wandb_run.log({
                "train/total_time_seconds": total_time,
                "train/total_merges": total_merges,
                "train/avg_time_per_merge": total_time / max(total_merges, 1),
            })

    def elapsed(self) -> float:
        return time() - self.start_time
