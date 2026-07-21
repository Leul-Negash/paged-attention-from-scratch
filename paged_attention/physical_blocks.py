from .exceptions import DoubleFreeError


class PhysicalBlockAllocator:
    """The global free pool of physical block ids.

    allocate() pops a block, free() returns one. Nothing here knows about
    requests or page tables; it just hands out integers and takes them back.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        # Treated as a stack so ids come back in a predictable order.
        self._free = list(reversed(range(num_blocks)))
        self._allocated: set[int] = set()

    def allocate(self) -> int | None:
        if not self._free:
            return None
        block_id = self._free.pop()
        self._allocated.add(block_id)
        return block_id

    def free(self, block_id: int, request_id=None) -> None:
        if block_id not in self._allocated:
            raise DoubleFreeError(
                f"block {block_id} freed but not currently allocated",
                request_id=request_id,
                block_id=block_id,
            )
        self._allocated.discard(block_id)
        self._free.append(block_id)

    def is_allocated(self, block_id: int) -> bool:
        return block_id in self._allocated

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return len(self._allocated)
