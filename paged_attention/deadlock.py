"""Task 2c - deadlock detection and recovery via a wait-for graph.

When the pool is empty, two requests can each hold a block the other needs and
neither can proceed. We model this as a directed wait-for graph (edge A -> B
means "A is waiting for a block that B holds"), look for a cycle with DFS every
step, and break it by evicting the request in the cycle that holds the most
blocks (fewest generated tokens breaks ties).
"""


def find_cycle(adj: dict[int, list[int]]) -> list[int] | None:
    """Return one directed cycle as a node list, or None. O(V + E) via DFS colors."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in adj}
    parent: dict[int, int] = {}

    def visit(start: int) -> list[int] | None:
        stack = [(start, iter(adj[start]))]
        color[start] = GRAY
        while stack:
            node, neighbours = stack[-1]
            advanced = False
            for nxt in neighbours:
                if color[nxt] == WHITE:
                    color[nxt] = GRAY
                    parent[nxt] = node
                    stack.append((nxt, iter(adj[nxt])))
                    advanced = True
                    break
                if color[nxt] == GRAY:  # back edge -> reconstruct the cycle
                    cycle = [node]
                    cur = node
                    while cur != nxt:
                        cur = parent[cur]
                        cycle.append(cur)
                    cycle.reverse()
                    return cycle
            if not advanced:
                color[node] = BLACK
                stack.pop()
        return None

    for node in adj:
        if color[node] == WHITE:
            found = visit(node)
            if found:
                return found
    return None


class DeadlockManager:
    """Tracks who holds which blocks and who is waiting for what."""

    def __init__(self):
        self.holds: dict[int, set[int]] = {}
        self.waiting_for: dict[int, int | None] = {}
        self.tokens: dict[int, int] = {}
        self.free: set[int] = set()

    def add_request(self, rid: int, holds: set[int], waiting_for: int | None = None,
                    tokens: int = 0) -> None:
        self.holds[rid] = set(holds)
        self.waiting_for[rid] = waiting_for
        self.tokens[rid] = tokens

    def _holder_of(self) -> dict[int, int]:
        return {block: rid for rid, blocks in self.holds.items() for block in blocks}

    def build_wait_for_graph(self) -> dict[int, list[int]]:
        holder = self._holder_of()
        adj = {rid: [] for rid in self.holds}
        for rid, block in self.waiting_for.items():
            if block is not None and block in holder and holder[block] != rid:
                adj[rid].append(holder[block])
        return adj

    def detect(self) -> list[int] | None:
        return find_cycle(self.build_wait_for_graph())

    def _evict(self, victim: int) -> set[int]:
        freed = self.holds.pop(victim)
        self.waiting_for.pop(victim)
        self.tokens.pop(victim)
        self.free |= freed
        # Anyone waiting on a just-freed block can now take it and proceed.
        for rid, block in self.waiting_for.items():
            if block in freed:
                self.holds[rid].add(block)
                self.free.discard(block)
                self.waiting_for[rid] = None
        return freed

    def recover_step(self) -> list[dict]:
        """Detect and break every cycle within a single time step."""
        actions = []
        while True:
            cycle = self.detect()
            if cycle is None:
                break
            # Most blocks held wins; fewest tokens generated breaks the tie.
            victim = max(cycle, key=lambda r: (len(self.holds[r]), -self.tokens[r]))
            freed = self._evict(victim)
            actions.append({"cycle": cycle, "victim": victim, "freed": sorted(freed)})
        return actions
