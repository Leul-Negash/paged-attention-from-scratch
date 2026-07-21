"""Task 2b - an SLA scheduler with deadlines and token budgets.

Two guarantees per request:
  deadline     - the most steps it may sit in the queue before being served.
  token_budget - a hard cap; generation is cut at exactly that many tokens.

Scheduling rule: highest priority first (1 > 2 > 3), earliest deadline breaks
ties. Capacity is reserved on admission (blocks_needed for the request's target),
so a request never dies mid-run for lack of memory and a deadline is only ever
missed when the pool is genuinely full at that moment.
"""

from .paged_allocator import PagedAllocator
from .request import Request


class SLAScheduler:
    def __init__(self, num_blocks: int, block_size: int):
        self.alloc = PagedAllocator(num_blocks, block_size)
        self.num_blocks = num_blocks
        self.reserved = 0  # blocks committed to running requests

        self.deadline_misses = 0
        self.budget_terminations = 0
        self.completed = 0
        self.orphaned_blocks = 0

    def _need(self, req: Request) -> int:
        return self.alloc.blocks_needed(req.target_length)

    def _terminate(self, req: Request, by_budget: bool) -> None:
        freed = self.alloc.release(req)
        self.reserved -= self._need(req)
        if len(freed) != self._need(req):
            self.orphaned_blocks += self._need(req) - len(freed)
        if by_budget:
            self.budget_terminations += 1
        self.completed += 1

    def run(self, requests: list[Request]) -> dict:
        pending = sorted(requests, key=lambda r: r.arrival)
        idx = 0
        waiting: list[Request] = []
        running: list[Request] = []
        missed: set[int] = set()
        t = 0

        while idx < len(pending) or waiting or running:
            while idx < len(pending) and pending[idx].arrival <= t:
                waiting.append(pending[idx])
                idx += 1

            # Highest priority first, earliest deadline breaks ties.
            waiting.sort(key=lambda r: (r.priority, r.deadline))

            still_waiting = []
            for req in waiting:
                if self.num_blocks - self.reserved >= self._need(req):
                    self.reserved += self._need(req)
                    self.alloc.admit(req)
                    req.admitted_at = t
                    running.append(req)
                else:
                    still_waiting.append(req)
            waiting = still_waiting

            # Anything left waiting past its deadline is a miss (capacity was
            # full this step, or it would have been admitted above).
            for req in waiting:
                if t - req.arrival > req.deadline and req.id not in missed:
                    missed.add(req.id)
                    self.deadline_misses += 1

            keep = []
            for req in running:
                self.alloc.append_token(req)
                if req.tokens_generated >= req.target_length:
                    by_budget = req.token_budget is not None and \
                        req.token_budget <= req.actual_length and \
                        req.tokens_generated == req.token_budget
                    self._terminate(req, by_budget)
                else:
                    keep.append(req)
            running = keep

            t += 1

        # Nothing may be left allocated once every request has terminated.
        self.orphaned_blocks += self.alloc.num_used
        return {
            "completed": self.completed,
            "deadline_misses": self.deadline_misses,
            "budget_terminations": self.budget_terminations,
            "orphaned_blocks": self.orphaned_blocks,
            "steps": t,
        }
