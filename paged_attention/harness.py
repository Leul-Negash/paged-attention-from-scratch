"""Task 1d - a correctness harness that rides alongside the PagedAllocator.

It does two jobs:
  1. assert_disjoint() - every step, no two active requests may share a block.
  2. guard() - run an operation and report any allocator error instead of
     letting it propagate, so an injected fault never crashes the run.

The faults themselves are injected mid-run by run/task1d_harness.py.
"""

from .exceptions import AllocatorError
from .paged_allocator import PagedAllocator


class CorrectnessHarness:
    def __init__(self, allocator: PagedAllocator):
        self.alloc = allocator
        self.violations = 0

    def assert_disjoint(self) -> None:
        """No physical block may appear in two page tables at once."""
        seen: dict[int, int] = {}  # block id -> owning request id
        for req_id, pt in self.alloc.page_tables.items():
            for block_id in pt.physical_blocks:
                if block_id in seen:
                    raise AllocatorError(
                        f"block {block_id} shared by requests "
                        f"{seen[block_id]} and {req_id}",
                        request_id=req_id,
                        block_id=block_id,
                    )
                seen[block_id] = req_id

    def guard(self, label: str, action) -> bool:
        """Run action; report (don't propagate) any allocator error."""
        try:
            action()
        except AllocatorError as err:
            self.violations += 1
            print(f"  CAUGHT {type(err).__name__:<18} request={err.request_id}  "
                  f"block={err.block_id}  ({err})")
            return True
        print(f"  MISSED {label}: no error raised")
        return False
