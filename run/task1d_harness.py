"""Task 1d - the harness rides a live run: invariants every step, faults injected mid-run.

The three classic bugs are injected while the simulation is running, not in
isolated scenarios afterwards. Each one is caught and reported, and the run
carries on to completion with the disjointness invariant still holding.
"""
import _bootstrap  # noqa: F401

from paged_attention.harness import CorrectnessHarness
from paged_attention.paged_allocator import PagedAllocator
from paged_attention.workload import generate_workload

BLOCK_SIZE = 8
NUM_BLOCKS = 512
BOUNDARY_MISS_STEP = 12
USE_AFTER_FREE_STEP = 20


def retire(alloc: PagedAllocator, req, skip_block: int) -> None:
    """Drop a request whose blocks we deliberately corrupted, without double-freeing."""
    page_table = alloc.page_tables.pop(req.id)
    for block_id in page_table.physical_blocks:
        if block_id != skip_block:
            alloc.blocks.free(block_id, request_id=req.id)


def main() -> None:
    alloc = PagedAllocator(NUM_BLOCKS, BLOCK_SIZE)
    harness = CorrectnessHarness(alloc)
    requests = generate_workload(60, seed=7)
    schedule: dict[int, list] = {}
    for r in requests:
        schedule.setdefault(r.arrival, []).append(r)

    active, t, remaining = [], 0, len(requests)
    last = max(schedule)
    double_free_injected = False

    print("live run, faults injected mid-simulation:")
    while remaining > 0:
        for r in schedule.get(t, []):
            alloc.admit(r)
            active.append(r)

        keep = []
        for r in active:
            alloc.append_token(r)
            if r.tokens_generated >= r.actual_length:
                freed = alloc.release(r)
                remaining -= 1
                # 1. Double free: the pool has just taken these blocks back, so
                #    hand one of them back a second time in the same step.
                if not double_free_injected and freed:
                    print(f"  step {t:>3}: inject double-free on request {r.id}")
                    harness.guard("double-free",
                                  lambda b=freed[0], rid=r.id:
                                  alloc.blocks.free(b, request_id=rid))
                    double_free_injected = True
            else:
                keep.append(r)
        active = keep

        # 2. Boundary miss: read a token whose logical block was never allocated.
        if t == BOUNDARY_MISS_STEP and active:
            victim = active[0]
            print(f"  step {t:>3}: inject boundary-miss on request {victim.id}")
            harness.guard("boundary-miss",
                          lambda v=victim: alloc.read(v, token_index=99_999))

        # 3. Use-after-free: free a live block, then read through the page table
        #    entry that still points at it. The request is retired afterwards.
        if t == USE_AFTER_FREE_STEP:
            victim = next((r for r in active
                           if len(alloc.page_tables[r.id]) >= 2), None)
            if victim is not None:
                stale = alloc.page_tables[victim.id].physical_blocks[1]
                alloc.blocks.free(stale, request_id=victim.id)
                print(f"  step {t:>3}: inject use-after-free on request {victim.id}")
                harness.guard("use-after-free",
                              lambda v=victim: alloc.read(v, token_index=BLOCK_SIZE))
                retire(alloc, victim, stale)
                active.remove(victim)
                remaining -= 1

        harness.assert_disjoint()   # invariant enforced at every step
        t += 1
        if t > last and not active:
            break

    print(f"\n{len(requests)} requests, {t} steps, disjointness held every step")
    print(f"{harness.violations}/3 injected faults caught and reported, no crash")
    assert harness.violations == 3, "an injected fault went undetected"
    assert alloc.num_used == 0, "blocks leaked"
    print("\nOK: harness caught every fault mid-run and the simulation survived.")


if __name__ == "__main__":
    main()
