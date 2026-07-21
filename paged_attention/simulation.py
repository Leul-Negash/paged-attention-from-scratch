"""Discrete-step simulators for the naive and paged allocators.

Both run the same clock: at each step new arrivals are admitted, every active
request generates one token, and finished requests free their memory. The two
differ only in how memory is handed out, which is the whole comparison.
"""

from .exceptions import OutOfBlocksError
from .naive_allocator import NaiveContiguousAllocator
from .paged_allocator import PagedAllocator
from .request import Request


def _by_arrival(requests: list[Request]) -> dict[int, list[Request]]:
    schedule: dict[int, list[Request]] = {}
    for r in requests:
        schedule.setdefault(r.arrival, []).append(r)
    return schedule


def run_naive(requests: list[Request], total_slots: int, record_maps: bool = False) -> dict:
    alloc = NaiveContiguousAllocator(total_slots)
    schedule = _by_arrival(requests)
    active: list[Request] = []
    rejections = 0
    peak_slots = 0
    internal_series: list[float] = []
    external_series: list[float] = []
    steps: list[dict] = []

    t = 0
    remaining = len(requests)
    last_arrival = max(schedule) if schedule else 0

    while remaining > 0:
        rejected_need = None
        for req in schedule.get(t, []):
            if alloc.place(req):
                req.admitted_at = t
                active.append(req)
            else:
                rejections += 1
                remaining -= 1
                if rejected_need is None:
                    rejected_need = req.max_length

        still_active = []
        for req in active:
            req.tokens_generated += 1
            if req.tokens_generated >= req.actual_length:
                alloc.release(req)
                remaining -= 1
            else:
                still_active.append(req)
        active = still_active

        peak_slots = max(peak_slots, alloc.slots_used)
        internal = alloc.internal_fragmentation_pct()
        external = alloc.external_fragmentation_pct(rejected_need)
        internal_series.append(internal)
        external_series.append(external)
        if record_maps:
            steps.append({
                "t": t,
                "map": alloc.memory_map(),
                "internal": internal,
                "external": external,
                "active": len(active),
                "used": alloc.slots_used,
            })

        t += 1
        if t > last_arrival and not active:
            break

    return {
        "rejections": rejections,
        "peak_slots": peak_slots,
        "internal_pct": _avg(internal_series),
        "external_pct": _avg(external_series),
        "steps": steps,
    }


def run_paged(requests: list[Request], num_blocks: int, block_size: int) -> dict:
    alloc = PagedAllocator(num_blocks, block_size)
    schedule = _by_arrival(requests)
    active: list[Request] = []
    rejections = 0
    peak_blocks = 0
    internal_series: list[float] = []
    total_slots = num_blocks * block_size

    t = 0
    remaining = len(requests)
    last_arrival = max(schedule) if schedule else 0

    while remaining > 0:
        for req in schedule.get(t, []):
            alloc.admit(req)
            active.append(req)
            req.admitted_at = t

        still_active = []
        for req in active:
            try:
                alloc.append_token(req)
            except OutOfBlocksError:
                # No preemption in Task 1: a request that cannot grow is dropped.
                alloc.release(req)
                rejections += 1
                remaining -= 1
                continue
            if req.tokens_generated >= req.actual_length:
                alloc.release(req)
                remaining -= 1
            else:
                still_active.append(req)
        active = still_active

        peak_blocks = max(peak_blocks, alloc.num_used)
        idle = sum(
            len(alloc.page_tables[r.id]) * block_size - r.tokens_generated
            for r in active
        )
        internal_series.append(100.0 * idle / total_slots)

        t += 1
        if t > last_arrival and not active:
            break

    # Every block must be back in the pool.
    assert alloc.num_used == 0, "paged allocator leaked blocks"

    return {
        "rejections": rejections,
        "peak_blocks": peak_blocks,
        "internal_pct": _avg(internal_series),
        "external_pct": 0.0,  # paging cannot strand free blocks by position
        "steps": [],
    }


def _avg(series: list[float]) -> float:
    return sum(series) / len(series) if series else 0.0
