from .exceptions import BoundaryMissError


class PageTable:
    """Per-request map from logical block index to physical block id.

    The request only ever thinks in logical terms (token 0, 1, 2, ...). This
    table is the translation layer to whatever physical blocks it was given.
    """

    def __init__(self, block_size: int, request_id=None):
        self.block_size = block_size
        self.request_id = request_id
        self._map: list[int] = []

    def __len__(self) -> int:
        return len(self._map)

    def needs_new_block(self, token_index: int) -> bool:
        # The one off-by-one that matters: token 8 with block_size 4 is
        # logical block 2, and len is 2 when blocks 0-1 hold tokens 0-7.
        return token_index // self.block_size >= len(self._map)

    def append(self, physical_block_id: int) -> None:
        self._map.append(physical_block_id)

    def logical_to_physical(self, token_index: int) -> int:
        logical = token_index // self.block_size
        if logical >= len(self._map):
            raise BoundaryMissError(
                f"token {token_index} maps to logical block {logical} "
                f"but only {len(self._map)} blocks are allocated",
                request_id=self.request_id,
            )
        return self._map[logical]

    @property
    def physical_blocks(self) -> list[int]:
        return list(self._map)

    def set(self, logical_index: int, physical_block_id: int) -> None:
        """Used by copy-on-write and prefix caching to repoint an entry."""
        self._map[logical_index] = physical_block_id
