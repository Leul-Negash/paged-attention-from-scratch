"""Task 3b - automatic prefix sharing.

Physical blocks are content-addressed: the cache maps the hash of a block's
token content to the physical block that already stores it. When a new request
arrives we walk its prompt block by block; every leading block whose content is
already cached is reused (ref_count bumped, nothing allocated), as long as the
matched prefix is at least MIN_PREFIX tokens long.
"""

import hashlib

MIN_PREFIX = 8  # a shared prefix shorter than this is not worth sharing


def _content_hash(tokens: tuple[int, ...]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for tok in tokens:
        h.update(tok.to_bytes(8, "little"))
    return h.hexdigest()


class Block:
    __slots__ = ("data", "ref_count")

    def __init__(self, tokens: list[int]):
        self.data = tokens
        self.ref_count = 1


class PrefixCachingAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: dict[int, Block] = {}
        self.free = list(reversed(range(num_blocks)))
        self.cache: dict[str, int] = {}  # content hash -> physical block id

    def _full_blocks(self, tokens: list[int]) -> list[list[int]]:
        bs = self.block_size
        return [tokens[i:i + bs] for i in range(0, len(tokens) - bs + 1, bs)]

    def admit(self, prompt_tokens: list[int]) -> dict:
        """Return the page table and how many blocks were newly allocated."""
        chunks = self._full_blocks(prompt_tokens)

        # How many leading blocks are already cached?
        matched = 0
        for chunk in chunks:
            if _content_hash(tuple(chunk)) in self.cache:
                matched += 1
            else:
                break
        if matched * self.block_size < MIN_PREFIX:
            matched = 0  # too short a prefix to bother sharing

        page_table = []
        new_allocations = 0
        for i, chunk in enumerate(chunks):
            key = _content_hash(tuple(chunk))
            if i < matched:
                block_id = self.cache[key]
                self.blocks[block_id].ref_count += 1
            else:
                block_id = self.free.pop()
                self.blocks[block_id] = Block(list(chunk))
                self.cache.setdefault(key, block_id)
                new_allocations += 1
            page_table.append(block_id)

        return {
            "page_table": page_table,
            "new_allocations": new_allocations,
            "reused_blocks": matched,
        }
