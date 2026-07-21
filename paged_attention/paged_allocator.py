from .exceptions import OutOfBlocksError, UseAfterFreeError
from .mock_llm import MockLLM
from .page_table import PageTable
from .physical_blocks import PhysicalBlockAllocator
from .request import Request


class PagedAllocator:
    """PagedAttention memory manager.

    Owns the global block pool and one page table per admitted request.
    Blocks are allocated strictly on demand, the moment a new token would
    overflow the request's last block.
    """

    def __init__(self, num_blocks: int, block_size: int, llm: MockLLM | None = None):
        assert block_size >= 1 and (block_size & (block_size - 1)) == 0, \
            "block_size must be a power of 2"
        self.block_size = block_size
        self.blocks = PhysicalBlockAllocator(num_blocks)
        self.llm = llm or MockLLM()
        self.page_tables: dict[int, PageTable] = {}

    # -- lifecycle ---------------------------------------------------------

    def admit(self, request: Request) -> None:
        self.page_tables[request.id] = PageTable(self.block_size, request.id)

    def append_token(self, request: Request) -> int:
        """Generate and store one token, growing the page table if needed.

        Raises OutOfBlocksError when a new block is required but the pool is
        empty. Task 2 catches that and preempts a victim.
        """
        pt = self.page_tables[request.id]
        index = request.tokens_generated
        if pt.needs_new_block(index):
            block_id = self.blocks.allocate()
            if block_id is None:
                raise OutOfBlocksError("free pool empty", request_id=request.id)
            pt.append(block_id)
        token = self.llm.token(request.id, index)
        request.tokens.append(token)
        request.tokens_generated += 1
        return token

    def read(self, request: Request, token_index: int) -> int:
        """Resolve a token's physical block. Used by the correctness harness."""
        pt = self.page_tables[request.id]
        block_id = pt.logical_to_physical(token_index)  # may raise BoundaryMiss
        if not self.blocks.is_allocated(block_id):
            raise UseAfterFreeError(
                f"read of token {token_index} hit freed block {block_id}",
                request_id=request.id,
                block_id=block_id,
            )
        return block_id

    def release(self, request: Request) -> list[int]:
        """Free every block owned by a request and drop its page table."""
        pt = self.page_tables.pop(request.id)
        freed = pt.physical_blocks
        for block_id in freed:
            self.blocks.free(block_id, request_id=request.id)
        return freed

    # -- introspection -----------------------------------------------------

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def active_ids(self) -> list[int]:
        return list(self.page_tables)

    @property
    def num_free(self) -> int:
        return self.blocks.num_free

    @property
    def num_used(self) -> int:
        return self.blocks.num_used
