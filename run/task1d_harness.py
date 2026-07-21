"""Task 1d - run the harness: invariant checks every step + three injected bugs."""
import _bootstrap  # noqa: F401

from paged_attention.harness import CorrectnessHarness
from paged_attention.paged_allocator import PagedAllocator
from paged_attention.workload import generate_workload

BLOCK_SIZE = 8
NUM_BLOCKS = 512


def clean_run() -> None:
    """A normal workload with the disjointness invariant checked every step."""
    alloc = PagedAllocator(NUM_BLOCKS, BLOCK_SIZE)
    harness = CorrectnessHarness(alloc)
    requests = generate_workload(60, seed=7)
    schedule: dict[int, list] = {}
    for r in requests:
        schedule.setdefault(r.arrival, []).append(r)

    active, t, remaining = [], 0, len(requests)
    last = max(schedule)
    while remaining > 0:
        for r in schedule.get(t, []):
            alloc.admit(r)
            active.append(r)
        keep = []
        for r in active:
            alloc.append_token(r)
            if r.tokens_generated >= r.actual_length:
                alloc.release(r)
                remaining -= 1
            else:
                keep.append(r)
        active = keep
        harness.assert_disjoint()   # invariant enforced at every step
        t += 1
        if t > last and not active:
            break
    print(f"clean run: {len(requests)} requests, disjointness held every step\n")


def injected_failures() -> None:
    print("injected failures (each must be caught, no crash):")

    # 1. Double free: release a request's blocks, then free one again.
    alloc = PagedAllocator(NUM_BLOCKS, BLOCK_SIZE)
    harness = CorrectnessHarness(alloc)  # one harness tallies all three
    req = generate_workload(1, seed=1)[0]
    req.actual_length = 20
    alloc.admit(req)
    for _ in range(20):
        alloc.append_token(req)
    stale_block = alloc.page_tables[req.id].physical_blocks[0]
    alloc.release(req)
    harness.guard("double-free",
                  lambda: alloc.blocks.free(stale_block, request_id=req.id))

    # 2. Use-after-free: free the underlying block but read through a page table
    #    entry that still points at it.
    alloc = PagedAllocator(NUM_BLOCKS, BLOCK_SIZE)
    req = generate_workload(1, seed=2)[0]
    req.actual_length = 20
    alloc.admit(req)
    for _ in range(20):
        alloc.append_token(req)
    live_block = alloc.page_tables[req.id].physical_blocks[1]
    alloc.blocks.free(live_block, request_id=req.id)  # premature free
    harness.guard("use-after-free", lambda: alloc.read(req, token_index=10))

    # 3. Boundary miss: read a token whose logical block was never allocated.
    alloc = PagedAllocator(NUM_BLOCKS, BLOCK_SIZE)
    req = generate_workload(1, seed=3)[0]
    req.actual_length = 10
    alloc.admit(req)
    for _ in range(10):
        alloc.append_token(req)
    harness.guard("boundary-miss", lambda: alloc.read(req, token_index=999))

    print(f"\n{harness.violations}/3 failures caught and reported.")


def main() -> None:
    clean_run()
    injected_failures()


if __name__ == "__main__":
    main()
