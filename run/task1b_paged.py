"""Task 1b - PagedAllocator correctness: on-demand allocation, zero leaks.

Runs 10,000 requests across 5 seeds for every block size in {4, 8, 16, 32}.
After each request completes we assert the pool is whole again, and after each
full run we assert not a single block leaked.
"""
import _bootstrap  # noqa: F401

from paged_attention.paged_allocator import PagedAllocator
from paged_attention.workload import generate_workload

BLOCK_SIZES = [4, 8, 16, 32]
SEEDS = [1, 2, 3, 4, 5]
N = 10_000
NUM_BLOCKS = 2048


def run_once(block_size: int, seed: int) -> None:
    alloc = PagedAllocator(NUM_BLOCKS, block_size)
    requests = generate_workload(N, seed, max_len_range=(10, 512))

    for req in requests:
        req.actual_length = min(req.actual_length, NUM_BLOCKS * block_size)
        alloc.admit(req)
        while req.tokens_generated < req.actual_length:
            before = len(alloc.page_tables[req.id])
            alloc.append_token(req)
            # A block is added exactly when the new token overflows the last one.
            expected = -(-req.tokens_generated // block_size)  # ceil
            assert len(alloc.page_tables[req.id]) == expected, "off-by-one in growth"
            assert len(alloc.page_tables[req.id]) in (before, before + 1)

        freed = alloc.release(req)
        assert len(set(freed)) == len(freed), "duplicate block ids in one request"
        assert alloc.num_used == 0, f"leak after request {req.id}"

    assert alloc.num_free == NUM_BLOCKS, "pool not fully restored"


def main() -> None:
    for block_size in BLOCK_SIZES:
        for seed in SEEDS:
            run_once(block_size, seed)
            print(f"block_size={block_size:>2}  seed={seed}  "
                  f"{N} requests  ->  no leaks, no off-by-one")
    print(f"\nAll {len(BLOCK_SIZES) * len(SEEDS)} runs passed "
          f"({len(BLOCK_SIZES) * len(SEEDS) * N:,} requests total).")


if __name__ == "__main__":
    main()
