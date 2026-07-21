"""Task 2a - preemption under permanent oversubscription.

The pool is far too small for the offered load, so allocate() runs dry
constantly. When it does we evict a victim (the request that has generated the
fewest tokens - cheapest to redo) using one of two strategies:

  recompute (tokens_generated <  T): throw the work away, re-queue from token 0.
  swap      (tokens_generated >= T): serialize the token sequence to a dict,
                                      free the blocks, reload and resume later.

Because the MockLLM is deterministic in (request_id, position), a request that
is recomputed or swapped-and-resumed produces byte-identical output to one that
was never touched. We verify that by hashing.
"""

import hashlib
from collections import deque

from .exceptions import OutOfBlocksError
from .paged_allocator import PagedAllocator
from .request import Request

STEP_CAP = 1_000_000


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _hash_tokens(tokens) -> str:
    h = hashlib.sha256()
    for tok in tokens:
        h.update(tok.to_bytes(8, "little"))
    return h.hexdigest()


class PreemptionEngine:
    def __init__(self, num_blocks: int, block_size: int, threshold_T: int):
        self.alloc = PagedAllocator(num_blocks, block_size)
        self.block_size = block_size
        self.threshold_T = threshold_T
        self.capacity_tokens = num_blocks * block_size

        self.waiting: deque[Request] = deque()
        self.running: dict[int, Request] = {}
        self.swap_store: dict[int, list[int]] = {}   # id -> saved token sequence
        self.swapped: set[int] = set()               # waiting ids that are swapped

        self.recompute_evictions = 0
        self.swap_evictions = 0
        self.mismatches = 0
        self.completed = 0
        self.peak_demand = 0

    # -- reference output for the byte-identical check ---------------------

    def _expected_hash(self, req: Request) -> str:
        return _hash_tokens(self.alloc.llm.token(req.id, i)
                            for i in range(req.target_length))

    # -- admission ---------------------------------------------------------

    def _can_admit(self, req: Request) -> bool:
        if req.id in self.swapped:
            return self.alloc.num_free >= _ceil_div(req.tokens_generated, self.block_size)
        return self.alloc.num_free >= 1

    def _admit(self, req: Request) -> None:
        if req.id in self.swapped:
            tokens = self.swap_store.pop(req.id)
            req.tokens = list(tokens)
            req.tokens_generated = len(tokens)
            self.alloc.admit(req)
            for _ in range(_ceil_div(len(tokens), self.block_size)):
                block = self.alloc.blocks.allocate()
                self.alloc.page_tables[req.id].append(block)
            self.swapped.discard(req.id)
        else:
            self.alloc.admit(req)  # fresh, or a recompute victim already reset
        self.running[req.id] = req

    def _drain_admissions(self) -> None:
        # FIFO, so requests that have waited longest go first.
        held: deque[Request] = deque()
        while self.waiting:
            req = self.waiting.popleft()
            if self._can_admit(req):
                self._admit(req)
            else:
                held.append(req)
        self.waiting = held

    # -- eviction ----------------------------------------------------------

    def _pick_victim(self, exclude_id: int) -> Request | None:
        candidates = [r for rid, r in self.running.items() if rid != exclude_id]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.tokens_generated)

    def _evict(self, victim: Request) -> None:
        del self.running[victim.id]
        self.alloc.release(victim)
        if victim.tokens_generated < self.threshold_T:
            victim.reset()
            self.recompute_evictions += 1
            self.waiting.append(victim)
        else:
            self.swap_store[victim.id] = list(victim.tokens)
            self.swapped.add(victim.id)
            self.swap_evictions += 1
            self.waiting.appendleft(victim)  # resume soon, keep the saved work warm

    def _grow(self, req: Request) -> None:
        while True:
            try:
                self.alloc.append_token(req)
                return
            except OutOfBlocksError:
                victim = self._pick_victim(exclude_id=req.id)
                if victim is None:
                    raise
                self._evict(victim)

    # -- main loop ---------------------------------------------------------

    def run(self, requests: list[Request]) -> dict:
        pending = sorted(requests, key=lambda r: r.arrival)
        idx = 0
        t = 0

        while idx < len(pending) or self.waiting or self.running:
            while idx < len(pending) and pending[idx].arrival <= t:
                self.waiting.append(pending[idx])
                idx += 1

            demand = sum(r.target_length - r.tokens_generated
                         for r in self.running.values())
            demand += sum(r.target_length - r.tokens_generated for r in self.waiting)
            demand += sum(len(self.swap_store[i]) for i in self.swapped)
            self.peak_demand = max(self.peak_demand, demand)

            self._drain_admissions()

            for req in list(self.running.values()):
                if req.id not in self.running:
                    continue  # got evicted while another request was growing
                self._grow(req)
                if req.done:
                    if _hash_tokens(req.tokens) != self._expected_hash(req):
                        self.mismatches += 1
                    self.alloc.release(req)
                    del self.running[req.id]
                    self.completed += 1

            t += 1
            if t > STEP_CAP:
                raise RuntimeError("did not converge")

        assert self.alloc.num_used == 0, "leaked blocks after draining"
        return {
            "completed": self.completed,
            "recompute_evictions": self.recompute_evictions,
            "swap_evictions": self.swap_evictions,
            "mismatches": self.mismatches,
            "peak_demand": self.peak_demand,
            "capacity_tokens": self.capacity_tokens,
            "oversubscription": self.peak_demand / self.capacity_tokens,
            "steps": t,
        }
