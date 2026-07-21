"""Task 1a - Naive contiguous allocator on a 200-request Poisson workload."""
import _bootstrap  # noqa: F401

from paged_attention.simulation import run_naive
from paged_attention.workload import generate_workload

TOTAL_SLOTS = 4096
BLOCK_SIZE = 16  # only used to express peak usage in blocks


def main() -> None:
    requests = generate_workload(n=200, seed=42)
    result = run_naive(requests, TOTAL_SLOTS, record_maps=True)

    print(f"Naive contiguous allocator  |  {TOTAL_SLOTS} token slots\n")
    print("step  used  int%   ext%  act  memory map")
    for s in result["steps"]:
        print(
            f"{s['t']:>4}  {s['used']:>4}  {s['internal']:>4.1f}  "
            f"{s['external']:>4.1f}  {s['active']:>3}  {s['map']}"
        )

    print("\n--- final ---")
    print(f"rejected requests        : {result['rejections']}")
    print(f"peak slots used          : {result['peak_slots']}  "
          f"({-(-result['peak_slots'] // BLOCK_SIZE)} blocks of {BLOCK_SIZE})")
    print(f"avg internal fragmentation: {result['internal_pct']:.1f}%")
    print(f"avg external fragmentation: {result['external_pct']:.1f}%")


if __name__ == "__main__":
    main()
