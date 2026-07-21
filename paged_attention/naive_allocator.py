from .request import Request


class Reservation:
    __slots__ = ("start", "size", "request")

    def __init__(self, start: int, size: int, request: Request):
        self.start = start
        self.size = size
        self.request = request

    @property
    def end(self) -> int:
        return self.start + self.size


class NaiveContiguousAllocator:
    """The allocator PagedAttention replaces.

    Every request reserves one contiguous slot equal to its *maximum* possible
    length and holds it from admission to completion, no matter how few tokens
    it actually generates. This is where internal and external fragmentation
    come from.
    """

    def __init__(self, total_slots: int):
        self.total_slots = total_slots
        self.reservations: list[Reservation] = []  # kept sorted by start

    def _free_gaps(self) -> list[tuple[int, int]]:
        """Return (start, size) of every free gap between reservations."""
        gaps = []
        cursor = 0
        for r in self.reservations:
            if r.start > cursor:
                gaps.append((cursor, r.start - cursor))
            cursor = r.end
        if cursor < self.total_slots:
            gaps.append((cursor, self.total_slots - cursor))
        return gaps

    def place(self, request: Request) -> bool:
        """First-fit a contiguous slot of size max_length. False if none fits."""
        size = request.max_length
        for start, gap_size in self._free_gaps():
            if gap_size >= size:
                res = Reservation(start, size, request)
                self.reservations.append(res)
                self.reservations.sort(key=lambda r: r.start)
                return True
        return False

    def release(self, request: Request) -> None:
        self.reservations = [r for r in self.reservations if r.request.id != request.id]

    # -- fragmentation metrics --------------------------------------------

    def internal_fragmentation_pct(self) -> float:
        """Reserved-but-idle token slots inside active reservations."""
        idle = sum(r.size - r.request.tokens_generated for r in self.reservations)
        return 100.0 * idle / self.total_slots

    def external_fragmentation_pct(self, next_need: int | None) -> float:
        """Free slots stuck in gaps too small for the next waiting request."""
        if not next_need:
            return 0.0
        stranded = sum(size for _, size in self._free_gaps() if size < next_need)
        return 100.0 * stranded / self.total_slots

    def memory_map(self, width: int = 100) -> str:
        """A one-line picture of memory: '#' reserved, '.' free."""
        cells = ["."] * width
        for r in self.reservations:
            lo = r.start * width // self.total_slots
            hi = max(lo + 1, r.end * width // self.total_slots)
            for i in range(lo, min(hi, width)):
                cells[i] = "#"
        return "".join(cells)

    @property
    def slots_used(self) -> int:
        return sum(r.size for r in self.reservations)
