"""Typed failures the correctness harness (Task 1d) is expected to catch."""


class AllocatorError(Exception):
    def __init__(self, message: str, request_id=None, block_id=None):
        super().__init__(message)
        self.request_id = request_id
        self.block_id = block_id


class DoubleFreeError(AllocatorError):
    pass


class UseAfterFreeError(AllocatorError):
    pass


class BoundaryMissError(AllocatorError):
    pass


class OutOfBlocksError(AllocatorError):
    """Raised when the free pool is empty. Task 2 turns this into preemption."""
    pass
