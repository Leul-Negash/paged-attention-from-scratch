"""Task 3a - copy-on-write with reference counting.

One prompt fans out into N parallel completions. While they only read the
prompt they all point at the same physical blocks (ref_count = N). The instant
a completion writes into a shared block it clones that block first, so no two
completions ever observe each other's generated tokens.
"""

from .mock_llm import MockLLM


class Block:
    __slots__ = ("data", "ref_count")

    def __init__(self, block_size: int):
        self.data: list[int | None] = [None] * block_size
        self.ref_count = 0


class Sequence:
    def __init__(self, seq_id: int):
        self.id = seq_id
        self.page_table: list[int] = []
        self.length = 0


class CoWAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(block_size) for _ in range(num_blocks)]
        self.free = list(reversed(range(num_blocks)))
        self.llm = MockLLM()
        self.clones = 0

    def _alloc(self) -> int:
        block_id = self.free.pop()
        self.blocks[block_id] = Block(self.block_size)
        self.blocks[block_id].ref_count = 1
        return block_id

    def _release(self, block_id: int) -> None:
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free.append(block_id)

    # -- sharing -----------------------------------------------------------

    def fork(self, prompt_tokens: list[int], n: int) -> list[Sequence]:
        """Lay the prompt into shared blocks and hand out N completions."""
        page_table = []
        for i, token in enumerate(prompt_tokens):
            if i % self.block_size == 0:
                page_table.append(self._alloc())
            self.blocks[page_table[-1]].data[i % self.block_size] = token
        for block_id in page_table:
            self.blocks[block_id].ref_count = n  # N completions share each block

        seqs = []
        for k in range(n):
            s = Sequence(seq_id=k)
            s.page_table = list(page_table)
            s.length = len(prompt_tokens)
            seqs.append(s)
        return seqs

    # -- the write path ----------------------------------------------------

    def append(self, seq: Sequence, token: int) -> int:
        index = seq.length
        logical, offset = divmod(index, self.block_size)

        if logical >= len(seq.page_table):
            seq.page_table.append(self._alloc())  # brand new, private block

        block_id = seq.page_table[logical]
        if self.blocks[block_id].ref_count > 1:
            block_id = self._copy_on_write(seq, logical, block_id)

        self.blocks[block_id].data[offset] = token
        seq.length += 1
        return block_id

    def _copy_on_write(self, seq: Sequence, logical: int, block_id: int) -> int:
        clone_id = self._alloc()
        self.blocks[clone_id].data = list(self.blocks[block_id].data)
        self.blocks[block_id].ref_count -= 1  # this seq no longer shares the original
        seq.page_table[logical] = clone_id
        self.clones += 1
        return clone_id

    def cancel(self, seq: Sequence) -> None:
        for block_id in seq.page_table:
            self._release(block_id)
        seq.page_table = []

    # -- helpers -----------------------------------------------------------

    @property
    def num_used(self) -> int:
        return sum(1 for b in self.blocks if b.ref_count > 0)
