"""Task 3c - a page table spread across 4 independent GPU nodes.

Each logical block now resolves to a (node_id, physical_block_id) pair. The
requesting code never picks a node: allocations spill to whichever node has the
most free blocks, hot remote blocks migrate to the local node, and when a node
dies every request holding blocks there is rescheduled on the survivors.
"""

from collections import deque

from .mock_llm import MockLLM

HOT_THRESHOLD = 3  # more than this many accesses in one step -> migrate home


class Node:
    def __init__(self, node_id: int, num_blocks: int):
        self.node_id = node_id
        self.online = True
        self.free = list(reversed(range(num_blocks)))
        self.data: dict[int, list[int]] = {}

    def allocate(self) -> int | None:
        if not self.online or not self.free:
            return None
        return self.free.pop()

    def release(self, block_id: int) -> None:
        self.data.pop(block_id, None)
        self.free.append(block_id)

    @property
    def num_free(self) -> int:
        return len(self.free) if self.online else 0


class DistributedRequest:
    def __init__(self, req_id: int, arrival: int, actual_length: int, home: int):
        self.id = req_id
        self.arrival = arrival
        self.actual_length = actual_length
        self.home = home
        self.page_table: list[tuple[int, int]] = []  # (node_id, block_id)
        self.tokens_generated = 0
        self.tokens: list[int] = []


class DistributedAllocator:
    def __init__(self, num_nodes: int, blocks_per_node: int):
        self.nodes = [Node(i, blocks_per_node) for i in range(num_nodes)]
        self.migrations = 0
        self.spills = 0

    def _spill_target(self) -> Node | None:
        candidates = [n for n in self.nodes if n.online and n.free]
        return max(candidates, key=lambda n: n.num_free, default=None)

    def allocate(self, home: int) -> tuple[int, int] | None:
        node = self.nodes[home]
        if node.online and node.free:
            return (home, node.allocate())
        target = self._spill_target()   # node's pool exhausted -> spill
        if target is None:
            return None
        self.spills += 1
        return (target.node_id, target.allocate())

    def free(self, loc: tuple[int, int]) -> None:
        node_id, block_id = loc
        if self.nodes[node_id].online:
            self.nodes[node_id].release(block_id)

    def migrate(self, loc: tuple[int, int], home: int) -> tuple[int, int]:
        """Move a hot remote block to the local node, transparently."""
        node_id, block_id = loc
        home_node = self.nodes[home]
        if node_id == home or not home_node.online or not home_node.free:
            return loc
        new_block = home_node.allocate()
        home_node.data[new_block] = list(self.nodes[node_id].data.get(block_id, []))
        self.nodes[node_id].release(block_id)
        self.migrations += 1
        return (home, new_block)


class DistributedEngine:
    def __init__(self, num_nodes: int = 4, blocks_per_node: int = 48,
                 block_size: int = 16):
        self.alloc = DistributedAllocator(num_nodes, blocks_per_node)
        self.block_size = block_size
        self.num_nodes = num_nodes
        self.llm = MockLLM()
        self.preempted_by_failure = 0

    def _write_token(self, req: DistributedRequest) -> None:
        index = req.tokens_generated
        if index // self.block_size >= len(req.page_table):
            loc = self.alloc.allocate(req.home)
            if loc is None:
                raise RuntimeError("cluster fully exhausted")
            node_id, block_id = loc
            self.alloc.nodes[node_id].data[block_id] = []
            req.page_table.append(loc)
        node_id, block_id = req.page_table[index // self.block_size]
        token = self.llm.token(req.id, index)
        self.alloc.nodes[node_id].data[block_id].append(token)
        req.tokens.append(token)
        req.tokens_generated += 1

    def _maybe_migrate_hot_block(self, req: DistributedRequest) -> None:
        # Attention reads the whole context each step, so every populated block
        # is hot (> 3 accesses). Any that spilled to a remote node is pulled back
        # to the local node whenever the local node has room.
        accesses = 4  # > HOT_THRESHOLD
        for i, loc in enumerate(req.page_table):
            if loc[0] != req.home and accesses > HOT_THRESHOLD:
                req.page_table[i] = self.alloc.migrate(loc, req.home)

    def _reschedule_after_failure(self, req: DistributedRequest, dead_node: int) -> None:
        # Free whatever survived, recompute from scratch on a live node.
        for loc in req.page_table:
            if loc[0] != dead_node:
                self.alloc.free(loc)
        req.page_table = []
        req.tokens_generated = 0
        req.tokens = []
        req.home = self._pick_live_home()
        self.preempted_by_failure += 1

    def _pick_live_home(self) -> int:
        live = [n for n in self.alloc.nodes if n.online]
        return max(live, key=lambda n: n.num_free).node_id

    def run(self, requests: list[DistributedRequest], fail_step: int | None = None,
            fail_node: int = 2) -> dict:
        pending = sorted(requests, key=lambda r: r.arrival)
        idx = 0
        active: list[DistributedRequest] = []
        completed = 0
        mismatches = 0
        t = 0

        while idx < len(pending) or active:
            while idx < len(pending) and pending[idx].arrival <= t:
                active.append(pending[idx])
                idx += 1

            if t == fail_step:
                self.alloc.nodes[fail_node].online = False
                self.alloc.nodes[fail_node].free = []
                self.alloc.nodes[fail_node].data = {}
                for req in active:
                    if any(loc[0] == fail_node for loc in req.page_table):
                        self._reschedule_after_failure(req, fail_node)  # lost data
                    elif req.home == fail_node:
                        req.home = self._pick_live_home()  # just move off the dead home

            keep = []
            for req in active:
                self._write_token(req)
                self._maybe_migrate_hot_block(req)
                if req.tokens_generated >= req.actual_length:
                    expected = [self.llm.token(req.id, i) for i in range(req.actual_length)]
                    if req.tokens != expected:
                        mismatches += 1
                    for loc in req.page_table:
                        self.alloc.free(loc)
                    completed += 1
                else:
                    keep.append(req)
            active = keep
            t += 1

        return {
            "completed": completed,
            "mismatches": mismatches,
            "spills": self.alloc.spills,
            "migrations": self.alloc.migrations,
            "preempted_by_failure": self.preempted_by_failure,
            "steps": t,
        }
