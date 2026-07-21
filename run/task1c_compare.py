"""Task 1c - Naive vs Paged on the exact same 200-request workload (seed=42)."""
import _bootstrap  # noqa: F401

from paged_attention.simulation import run_naive, run_paged
from paged_attention.workload import generate_workload

BLOCK_SIZE = 16
NUM_BLOCKS = 256
TOTAL_SLOTS = NUM_BLOCKS * BLOCK_SIZE  # identical capacity for a fair fight


def main() -> None:
    naive = run_naive(generate_workload(200, 42), TOTAL_SLOTS)
    paged = run_paged(generate_workload(200, 42), NUM_BLOCKS, BLOCK_SIZE)

    naive_peak_blocks = -(-naive["peak_slots"] // BLOCK_SIZE)

    print(f"Same workload: 200 requests, seed=42, {TOTAL_SLOTS} token capacity "
          f"({NUM_BLOCKS} blocks x {BLOCK_SIZE})\n")
    header = f"{'allocator':<10}{'peak blocks':>13}{'rejected':>11}{'internal%':>12}{'external%':>12}"
    print(header)
    print("-" * len(header))
    print(f"{'Naive':<10}{naive_peak_blocks:>13}{naive['rejections']:>11}"
          f"{naive['internal_pct']:>12.1f}{naive['external_pct']:>12.1f}")
    print(f"{'Paged':<10}{paged['peak_blocks']:>13}{paged['rejections']:>11}"
          f"{paged['internal_pct']:>12.1f}{paged['external_pct']:>12.1f}")


if __name__ == "__main__":
    main()
