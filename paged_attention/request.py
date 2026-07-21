from dataclasses import dataclass, field
from enum import IntEnum


class Priority(IntEnum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class Request:
    id: int
    arrival: int
    max_length: int          # largest number of tokens this request could reach
    actual_length: int       # how many it would generate if left alone

    # Scheduling attributes (Task 2). Defaults keep Task 1 code untouched.
    priority: Priority = Priority.MEDIUM
    deadline: int | None = None       # max steps it may wait in the queue
    token_budget: int | None = None   # hard cap on generated tokens

    # Runtime state.
    tokens_generated: int = 0
    tokens: list[int] = field(default_factory=list)
    admitted_at: int | None = None

    @property
    def target_length(self) -> int:
        """Where this request actually stops: its own length or its budget."""
        if self.token_budget is None:
            return self.actual_length
        return min(self.actual_length, self.token_budget)

    @property
    def done(self) -> bool:
        return self.tokens_generated >= self.target_length

    def reset(self) -> None:
        """Wipe generation state (recompute preemption starts a request over)."""
        self.tokens_generated = 0
        self.tokens = []
